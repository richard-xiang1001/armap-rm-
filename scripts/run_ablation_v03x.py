#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def _run(cmd: List[str]) -> str:
    print("[run]", " ".join(cmd))
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.stderr.strip():
        print(proc.stderr.strip())
    return proc.stdout


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_last_jsonl(path: Path) -> Dict:
    last = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    if last is None:
        raise RuntimeError(f"No JSONL rows: {path}")
    return last


def _parse_last_dict(output: str) -> Dict:
    rows = [line.strip() for line in output.splitlines() if line.strip().startswith("{") and line.strip().endswith("}")]
    if not rows:
        raise RuntimeError(f"Cannot parse eval dict from output:\n{output}")
    return ast.literal_eval(rows[-1])


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _split(input_path: Path, train_path: Path, valid_path: Path, valid_size: int, seed: int) -> Dict:
    rows = _read_jsonl(input_path)
    if len(rows) < 2:
        raise RuntimeError(f"Need at least 2 rows to split, got {len(rows)}")
    rng = random.Random(seed)
    rng.shuffle(rows)

    valid_n = max(1, min(valid_size, len(rows) - 1))
    train = rows[:-valid_n]
    valid = rows[-valid_n:]
    _write_jsonl(train_path, train)
    _write_jsonl(valid_path, valid)
    return {"total": len(rows), "train": len(train), "valid": len(valid)}


def _variant_grid() -> List[Dict]:
    baseline = {
        "name": "baseline",
        "neg_type": "replay_corrupt",
        "instruction": "refined",
        "input_format": "flat",
        "pooling": "mean",
    }

    variants = [baseline]
    for neg in ("truncate_last_k", "swap_step"):
        variants.append({**baseline, "name": f"neg_{neg}", "neg_type": neg})
    for instruction in ("raw",):
        variants.append({**baseline, "name": f"instruction_{instruction}", "instruction": instruction})
    for input_format in ("stepwise",):
        variants.append({**baseline, "name": f"format_{input_format}", "input_format": input_format})
    for pooling in ("last", "last_k_step"):
        variants.append({**baseline, "name": f"pool_{pooling}", "pooling": pooling})
    return variants


def main() -> None:
    ap = argparse.ArgumentParser(description="Run v0.3.x ablations (single-variable toggles).")
    ap.add_argument("--raw_path", type=str, required=True)
    ap.add_argument("--adapter", type=str, default="generic_jsonl", choices=["generic_jsonl", "armap_style", "webshop_like"])
    ap.add_argument("--work_dir", type=str, default="data/ablation/v03x")
    ap.add_argument("--results_dir", type=str, default="results/ablation/v03x")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_pairs", type=int, default=1200)
    ap.add_argument("--target_valid_size", type=int, default=200)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--amp_dtype", type=str, default="bf16", choices=["none", "bf16", "fp16"])
    ap.add_argument("--tail_drop_steps", type=int, default=2)
    ap.add_argument("--last_k_steps", type=int, default=3)
    ap.add_argument("--last_k_steps_pool", type=int, default=3)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    raw_path = Path(args.raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw episodes not found: {raw_path}")

    work_root = Path(args.work_dir)
    results_root = Path(args.results_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for idx, cfg in enumerate(_variant_grid()):
        run_name = f"{idx:02d}_{cfg['name']}"
        run_work = work_root / run_name
        run_results = results_root / run_name
        reports = run_work / "reports"
        run_work.mkdir(parents=True, exist_ok=True)
        run_results.mkdir(parents=True, exist_ok=True)
        reports.mkdir(parents=True, exist_ok=True)

        pairs_unsanitized = run_work / "pairs.unsanitized.jsonl"
        pairs_sanitized = run_work / "pairs.sanitized.jsonl"
        train_path = run_work / "train.jsonl"
        valid_path = run_work / "valid.jsonl"

        use_refined = cfg["instruction"] == "refined"

        try:
            _run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "build_pairs_v030.py"),
                    "--input_path",
                    str(raw_path),
                    "--output_path",
                    str(pairs_unsanitized),
                    "--adapter",
                    args.adapter,
                    "--seed",
                    str(args.seed),
                    "--max_pairs",
                    str(args.max_pairs),
                    "--tail_drop_steps",
                    str(args.tail_drop_steps),
                    "--forced_neg_type",
                    cfg["neg_type"],
                    "--stats_path",
                    str(reports / "export_stats.json"),
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
                    str(reports / "sanitize_audit.json"),
                ]
            )

            split = _split(
                input_path=pairs_sanitized,
                train_path=train_path,
                valid_path=valid_path,
                valid_size=args.target_valid_size,
                seed=args.seed,
            )
            (reports / "split_stats.json").write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

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
                    "--report_path",
                    str(reports / "train_lint.json"),
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
                    str(run_results),
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
                    "--max_steps",
                    str(args.max_steps),
                    "--input_format",
                    cfg["input_format"],
                    "--pooling",
                    cfg["pooling"],
                    "--last_k_steps_pool",
                    str(args.last_k_steps_pool),
                    "--use_refined_instruction",
                    "true" if use_refined else "false",
                ]
            )

            eval_out = _run(
                [
                    sys.executable,
                    "-m",
                    "rm.eval",
                    "--ckpt",
                    str(run_results / "rm.pt"),
                    "--valid_path",
                    str(valid_path),
                    "--max_len",
                    str(args.max_len),
                    "--num_workers",
                    "0",
                    "--input_format",
                    cfg["input_format"],
                    "--use_refined_instruction",
                    "true" if use_refined else "false",
                ]
            )

            eval_metrics = _parse_last_dict(eval_out)
            train_metrics = _read_last_jsonl(run_results / "metrics.jsonl")
            lint = _read_json(reports / "train_lint.json")

            rows.append(
                {
                    "name": run_name,
                    "status": "ok",
                    "neg_type": cfg["neg_type"],
                    "instruction": cfg["instruction"],
                    "format": cfg["input_format"],
                    "pooling": cfg["pooling"],
                    "pair_accuracy": float(eval_metrics["pair_accuracy"]),
                    "gap_p50": float(eval_metrics["gap_p50"]),
                    "truncation_risk_last_k_steps": float(lint["truncation_risk_last_k_steps"]),
                    "neg_replay_success_rate": float(lint["neg_replay_success_rate"]),
                    "train_pair_accuracy": float(train_metrics["pair_accuracy"]),
                    "n_valid": int(eval_metrics["n"]),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "name": run_name,
                    "status": "failed",
                    "neg_type": cfg["neg_type"],
                    "instruction": cfg["instruction"],
                    "format": cfg["input_format"],
                    "pooling": cfg["pooling"],
                    "error": str(exc),
                }
            )

    out_csv = results_root / "v03x_ablation.csv"
    out_md = results_root / "v03x_ablation.md"

    fieldnames = [
        "name",
        "status",
        "neg_type",
        "instruction",
        "format",
        "pooling",
        "pair_accuracy",
        "gap_p50",
        "truncation_risk_last_k_steps",
        "neg_replay_success_rate",
        "train_pair_accuracy",
        "n_valid",
        "error",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "# v0.3.x Ablation Results",
        "",
        "| run | status | neg_type | instruction | format | pooling | pair_accuracy | gap_p50 | trunc_risk_last_k | replay_ok_rate |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {status} | {neg_type} | {instruction} | {format} | {pooling} | {pair_accuracy} | {gap_p50} | {truncation_risk_last_k_steps} | {neg_replay_success_rate} |".format(
                name=row.get("name", ""),
                status=row.get("status", ""),
                neg_type=row.get("neg_type", ""),
                instruction=row.get("instruction", ""),
                format=row.get("format", ""),
                pooling=row.get("pooling", ""),
                pair_accuracy=row.get("pair_accuracy", ""),
                gap_p50=row.get("gap_p50", ""),
                truncation_risk_last_k_steps=row.get("truncation_risk_last_k_steps", ""),
                neg_replay_success_rate=row.get("neg_replay_success_rate", ""),
            )
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"csv": str(out_csv), "md": str(out_md), "runs": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
