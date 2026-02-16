#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import ast
import json
import os
import random
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


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _split_train_valid_with_size(
    input_path: Path,
    train_path: Path,
    valid_path: Path,
    valid_ratio: float,
    valid_size: int,
    seed: int,
) -> Dict:
    rows = _read_jsonl(input_path)
    if len(rows) < 2:
        raise RuntimeError(f"Need at least 2 rows to split train/valid, got {len(rows)}")
    rng = random.Random(seed)
    rng.shuffle(rows)

    if valid_size > 0:
        valid_n = valid_size
    else:
        valid_n = int(round(len(rows) * valid_ratio))
    valid_n = max(1, min(valid_n, len(rows) - 1))

    train_rows = rows[:-valid_n]
    valid_rows = rows[-valid_n:]
    _write_jsonl(train_path, train_rows)
    _write_jsonl(valid_path, valid_rows)
    return {
        "total": len(rows),
        "train": len(train_rows),
        "valid": len(valid_rows),
        "valid_ratio": valid_ratio,
        "valid_size_requested": valid_size,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run v0.4.0 real evidence acceptance.")
    ap.add_argument("--raw_path", type=str, default="data/raw/episodes_real_v040.jsonl")
    ap.add_argument("--adapter", type=str, default="real_logs_v1", choices=["real_logs_v1", "generic_jsonl", "armap_style"])
    ap.add_argument("--env_judge", type=str, default="real_logs", choices=["real_logs", "arithmetic"])
    ap.add_argument("--work_dir", type=str, default="data/real_v040")
    ap.add_argument("--results_dir", type=str, default="results/v040_real")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_pairs", type=int, default=1200)
    ap.add_argument("--target_train_size", type=int, default=1000)
    ap.add_argument("--target_valid_size", type=int, default=200)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--amp_dtype", type=str, default="bf16", choices=["none", "bf16", "fp16"])
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--pin_memory", type=str, default="auto", choices=["auto", "true", "false"])
    ap.add_argument("--persistent_workers", type=str, default="auto", choices=["auto", "true", "false"])
    ap.add_argument("--tail_drop_steps", type=int, default=2)
    ap.add_argument("--max_negative_attempts_per_positive", type=int, default=3)

    ap.add_argument("--last_k_steps", type=int, default=3)
    ap.add_argument("--truncation_threshold", type=float, default=0.05)
    ap.add_argument("--min_replay_ok_rate", type=float, default=0.0)
    ap.add_argument("--min_env_judge_consistency", type=float, default=0.95)
    ap.add_argument("--max_judge_unknown_rate", type=float, default=0.05)
    ap.add_argument("--min_pair_accuracy", type=float, default=0.55)
    ap.add_argument("--min_gap_p50", type=float, default=0.0)
    ap.add_argument("--parity_threshold", type=float, default=0.03)
    ap.add_argument("--input_format", type=str, default="flat", choices=["flat", "stepwise"])
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean", "last", "last_k_step"])
    ap.add_argument("--last_k_steps_pool", type=int, default=3)
    ap.add_argument("--report_path", type=str, default="", help="Default: <work_dir>/reports/v040_real_acceptance.json")
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

    pairs_all = work_dir / "pairs.all.jsonl"
    pairs_unsanitized = work_dir / "pairs.trainable.unsanitized.jsonl"
    pairs_sanitized = work_dir / "pairs.sanitized.jsonl"
    train_path = work_dir / "train.jsonl"
    valid_path = work_dir / "valid.jsonl"

    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "build_pairs_v030.py"),
            "--input_path",
            str(raw_path),
            "--output_path",
            str(pairs_all),
            "--trainable_output_path",
            str(pairs_unsanitized),
            "--adapter",
            args.adapter,
            "--env_judge",
            args.env_judge,
            "--seed",
            str(args.seed),
            "--max_pairs",
            str(args.max_pairs),
            "--tail_drop_steps",
            str(args.tail_drop_steps),
            "--max_negative_attempts_per_positive",
            str(args.max_negative_attempts_per_positive),
            "--stats_path",
            str(reports_dir / "export_stats_v040.json"),
        ]
    )

    _run(
        [
            sys.executable,
            str(repo_root / "ingest" / "sanitize_traj.py"),
            "--input_path",
            str(pairs_unsanitized),
            "--output_path",
            str(pairs_sanitized),
            "--leak_terms_file",
            str(repo_root / "configs" / "leak_terms.txt"),
            "--audit_path",
            str(reports_dir / "sanitize_audit.json"),
        ]
    )

    split_stats = _split_train_valid_with_size(
        input_path=pairs_sanitized,
        train_path=train_path,
        valid_path=valid_path,
        valid_ratio=0.1,
        valid_size=args.target_valid_size,
        seed=args.seed,
    )
    (reports_dir / "split_stats.json").write_text(json.dumps(split_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "lint_data.py"),
            "--path",
            str(train_path),
            "--max_len",
            str(args.max_len),
            "--last_k_steps",
            str(args.last_k_steps),
            "--step_markers",
            "step,action:,observation:,obs:",
            "--leak_terms_file",
            str(repo_root / "configs" / "leak_terms.txt"),
            "--fail_on_leak",
            "--fail_on_missing",
            "--fail_on_truncation_risk",
            "--max_truncation_risk_last_k_steps",
            str(args.truncation_threshold),
            "--fail_on_low_replay_ok",
            "--min_replay_ok_rate",
            str(args.min_replay_ok_rate),
            "--fail_on_low_env_consistency",
            "--min_env_judge_consistency",
            str(args.min_env_judge_consistency),
            "--fail_on_high_judge_unknown",
            "--max_judge_unknown_rate",
            str(args.max_judge_unknown_rate),
            "--report_path",
            str(reports_dir / "train_lint.json"),
        ]
    )

    _run(
        [
            sys.executable,
            "-m",
            "rm.train",
            "--train_path",
            str(train_path),
            "--valid_path",
            str(valid_path),
            "--save_dir",
            str(results_dir),
            "--max_len",
            str(args.max_len),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--epochs",
            str(args.epochs),
            "--seed",
            str(args.seed),
            "--num_workers",
            str(args.num_workers),
            "--device",
            args.device,
            "--amp_dtype",
            args.amp_dtype,
            "--grad_accum_steps",
            str(args.grad_accum_steps),
            "--max_steps",
            str(args.max_steps),
            "--pin_memory",
            args.pin_memory,
            "--prefetch_factor",
            str(args.prefetch_factor),
            "--persistent_workers",
            args.persistent_workers,
            "--input_format",
            args.input_format,
            "--pooling",
            args.pooling,
            "--last_k_steps_pool",
            str(args.last_k_steps_pool),
        ]
    )

    gpu_eval_out = _run(
        [
            sys.executable,
            "-m",
            "rm.eval",
            "--ckpt",
            str(results_dir / "rm.pt"),
            "--valid_path",
            str(valid_path),
            "--max_len",
            str(args.max_len),
            "--num_workers",
            "0",
            "--input_format",
            args.input_format,
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
            str(valid_path),
            "--max_len",
            str(args.max_len),
            "--num_workers",
            "0",
            "--input_format",
            args.input_format,
        ],
        env=cpu_env,
    )
    cpu_eval = _parse_last_dict(cpu_eval_out)

    export_stats = _load_json(reports_dir / "export_stats_v040.json")
    lint_stats = _load_json(reports_dir / "train_lint.json")
    train_metrics = _last_jsonl(results_dir / "metrics.jsonl")

    parity_diff = abs(float(gpu_eval["pair_accuracy"]) - float(cpu_eval["pair_accuracy"]))
    checks = {
        "export_pairs_sufficient": int(export_stats.get("pairs_out", 0)) >= args.max_pairs,
        "split_train_exact": int(split_stats.get("train", -1)) == args.target_train_size,
        "split_valid_exact": int(split_stats.get("valid", -1)) == args.target_valid_size,
        "gate_missing_zero": int(lint_stats["missing_required_fields"]["rows_with_missing"]) == 0,
        "gate_leak_zero": int(lint_stats["leak_hit_count"]) == 0,
        "gate_truncation_ok": float(lint_stats["truncation_risk_last_k_steps"]) <= args.truncation_threshold,
        "gate_replay_ok": float(lint_stats["neg_replay_success_rate"]) >= args.min_replay_ok_rate,
        "gate_env_consistency_ok": float(lint_stats["env_judge_consistency"]) >= args.min_env_judge_consistency,
        "gate_judge_unknown_ok": float(lint_stats["judge_unknown_rate"]) <= args.max_judge_unknown_rate,
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
            "min_replay_ok_rate": args.min_replay_ok_rate,
            "min_env_judge_consistency": args.min_env_judge_consistency,
            "max_judge_unknown_rate": args.max_judge_unknown_rate,
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

    report_path = Path(args.report_path) if args.report_path else (work_dir / "reports" / "v040_real_acceptance.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report_path": str(report_path), "passed": passed}, ensure_ascii=False))

    if not passed:
        failed = [k for k, v in checks.items() if not v]
        raise SystemExit(f"Acceptance failed: {failed}. See report: {report_path}")


if __name__ == "__main__":
    main()
