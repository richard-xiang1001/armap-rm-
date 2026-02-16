#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def _run(cmd: List[str], env: Dict[str, str] | None = None) -> str:
    print("[run]", " ".join(cmd))
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.stderr.strip():
        print(proc.stderr.strip())
    return proc.stdout


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _last_jsonl(path: Path) -> Dict:
    last = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    if last is None:
        raise RuntimeError(f"No JSONL content found in {path}")
    return last


def _parse_last_dict(output: str) -> Dict:
    candidates = [line.strip() for line in output.splitlines() if line.strip().startswith("{") and line.strip().endswith("}")]
    if not candidates:
        raise RuntimeError(f"Cannot parse dict from output:\n{output}")
    return ast.literal_eval(candidates[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Run v0.2.1 acceptance: 5000/500 + gate + CPU/GPU parity.")
    ap.add_argument("--raw_path", type=str, default="data/raw/episodes_real.jsonl")
    ap.add_argument("--adapter", type=str, default="generic_jsonl", choices=["generic_jsonl", "armap_style"])
    ap.add_argument("--work_dir", type=str, default="data/real")
    ap.add_argument("--results_dir", type=str, default="results/v021_real")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_pairs", type=int, default=5500)
    ap.add_argument("--target_train_size", type=int, default=5000)
    ap.add_argument("--target_valid_size", type=int, default=500)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--amp_dtype", type=str, default="bf16", choices=["none", "bf16", "fp16"])
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--pin_memory", type=str, default="auto", choices=["auto", "true", "false"])
    ap.add_argument("--persistent_workers", type=str, default="auto", choices=["auto", "true", "false"])
    ap.add_argument("--last_k_steps", type=int, default=3)
    ap.add_argument("--truncation_threshold", type=float, default=0.05)
    ap.add_argument("--min_pair_accuracy", type=float, default=0.55)
    ap.add_argument("--min_gap_p50", type=float, default=0.0)
    ap.add_argument("--parity_threshold", type=float, default=0.03)
    ap.add_argument("--report_path", type=str, default="", help="Default: <work_dir>/reports/v021_acceptance.json")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    work_dir = Path(args.work_dir)
    results_dir = Path(args.results_dir)
    reports_dir = work_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_path = Path(args.raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw episodes file not found: {raw_path}")

    pipeline_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_v020_pipeline.py"),
        "--raw_path",
        str(raw_path),
        "--adapter",
        args.adapter,
        "--work_dir",
        str(work_dir),
        "--results_dir",
        str(results_dir),
        "--seed",
        str(args.seed),
        "--max_pairs",
        str(args.max_pairs),
        "--valid_size",
        str(args.target_valid_size),
        "--max_len",
        str(args.max_len),
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--max_steps",
        str(args.max_steps),
        "--lr",
        str(args.lr),
        "--device",
        args.device,
        "--amp_dtype",
        args.amp_dtype,
        "--num_workers",
        str(args.num_workers),
        "--grad_accum_steps",
        str(args.grad_accum_steps),
        "--prefetch_factor",
        str(args.prefetch_factor),
        "--pin_memory",
        args.pin_memory,
        "--persistent_workers",
        args.persistent_workers,
        "--last_k_steps",
        str(args.last_k_steps),
        "--max_truncation_risk_last_k_steps",
        str(args.truncation_threshold),
    ]
    _run(pipeline_cmd)

    export_stats = _load_json(reports_dir / "export_stats.json")
    split_stats = _load_json(reports_dir / "split_stats.json")
    lint_stats = _load_json(reports_dir / "train_lint.json")
    train_metrics = _last_jsonl(results_dir / "metrics.jsonl")

    gpu_eval_out = _run(
        [
            sys.executable,
            "-m",
            "rm.eval",
            "--ckpt",
            str(results_dir / "rm.pt"),
            "--valid_path",
            str(work_dir / "valid.jsonl"),
            "--max_len",
            str(args.max_len),
            "--num_workers",
            "0",
        ]
    )
    gpu_eval = _parse_last_dict(gpu_eval_out)

    cpu_env = os.environ.copy()
    cpu_env["CUDA_VISIBLE_DEVICES"] = ""
    cpu_eval_out = _run(
        [
            sys.executable,
            "-m",
            "rm.eval",
            "--ckpt",
            str(results_dir / "rm.pt"),
            "--valid_path",
            str(work_dir / "valid.jsonl"),
            "--max_len",
            str(args.max_len),
            "--num_workers",
            "0",
        ],
        env=cpu_env,
    )
    cpu_eval = _parse_last_dict(cpu_eval_out)

    parity_diff = abs(float(gpu_eval["pair_accuracy"]) - float(cpu_eval["pair_accuracy"]))
    checks = {
        "export_pairs_sufficient": int(export_stats.get("pairs_out", 0)) >= args.max_pairs,
        "split_train_exact": int(split_stats.get("train", -1)) == args.target_train_size,
        "split_valid_exact": int(split_stats.get("valid", -1)) == args.target_valid_size,
        "gate_missing_zero": int(lint_stats["missing_required_fields"]["rows_with_missing"]) == 0,
        "gate_leak_zero": int(lint_stats["leak_hit_count"]) == 0,
        "gate_truncation_ok": float(lint_stats["truncation_risk_last_k_steps"]) <= args.truncation_threshold,
        "train_pair_acc_ok": float(train_metrics["pair_accuracy"]) > args.min_pair_accuracy,
        "eval_gap_p50_ok": float(gpu_eval["gap_p50"]) > args.min_gap_p50,
        "parity_ok": parity_diff <= args.parity_threshold,
    }
    passed = all(checks.values())

    report = {
        "raw_path": str(raw_path),
        "work_dir": str(work_dir),
        "results_dir": str(results_dir),
        "seed": args.seed,
        "targets": {
            "max_pairs": args.max_pairs,
            "target_train_size": args.target_train_size,
            "target_valid_size": args.target_valid_size,
            "truncation_threshold": args.truncation_threshold,
            "min_pair_accuracy": args.min_pair_accuracy,
            "min_gap_p50": args.min_gap_p50,
            "parity_threshold": args.parity_threshold,
        },
        "export_stats": export_stats,
        "split_stats": split_stats,
        "lint_stats": lint_stats,
        "final_train_metrics": train_metrics,
        "gpu_eval": gpu_eval,
        "cpu_eval": cpu_eval,
        "pair_accuracy_parity_diff": parity_diff,
        "checks": checks,
        "passed": passed,
    }

    report_path = Path(args.report_path) if args.report_path else (work_dir / "reports" / "v021_acceptance.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report_path": str(report_path), "passed": passed}, ensure_ascii=False))

    if not passed:
        failed = [k for k, v in checks.items() if not v]
        raise SystemExit(f"Acceptance failed: {failed}. See report: {report_path}")


if __name__ == "__main__":
    main()
