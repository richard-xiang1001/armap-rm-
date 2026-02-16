#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


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


def _run(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


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
    ap = argparse.ArgumentParser(description="v0.2.0 pipeline: export -> sanitize -> lint gate -> train -> eval")
    ap.add_argument("--raw_path", type=str, required=True)
    ap.add_argument("--adapter", type=str, default="generic_jsonl", choices=["generic_jsonl", "armap_style"])
    ap.add_argument("--work_dir", type=str, default="data/real")
    ap.add_argument("--results_dir", type=str, default="results/v020_real")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--valid_size", type=int, default=0, help="If >0, overrides ratio-based split.")
    ap.add_argument("--tail_drop_steps", type=int, default=2)
    ap.add_argument("--max_pairs", type=int, default=0)
    ap.add_argument("--leak_terms_file", type=str, default="configs/leak_terms.txt")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--last_k_steps", type=int, default=3)
    ap.add_argument("--max_truncation_risk_last_k_steps", type=float, default=0.05)

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--amp_dtype", type=str, choices=["none", "bf16", "fp16"], default="bf16")
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--pin_memory", type=str, choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--persistent_workers", type=str, choices=["auto", "true", "false"], default="auto")
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    reports_dir = work_dir / "reports"
    results_dir = Path(args.results_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    pairs_unsanitized = work_dir / "pairs.unsanitized.jsonl"
    pairs_sanitized = work_dir / "pairs.sanitized.jsonl"
    train_path = work_dir / "train.jsonl"
    valid_path = work_dir / "valid.jsonl"

    _run(
        [
            sys.executable,
            "ingest/export_pairs.py",
            "--input_path",
            args.raw_path,
            "--output_path",
            str(pairs_unsanitized),
            "--adapter",
            args.adapter,
            "--seed",
            str(args.seed),
            "--tail_drop_steps",
            str(args.tail_drop_steps),
            "--max_pairs",
            str(args.max_pairs),
            "--stats_path",
            str(reports_dir / "export_stats.json"),
        ]
    )

    _run(
        [
            sys.executable,
            "ingest/sanitize_traj.py",
            "--input_path",
            str(pairs_unsanitized),
            "--output_path",
            str(pairs_sanitized),
            "--leak_terms_file",
            args.leak_terms_file,
            "--audit_path",
            str(reports_dir / "sanitize_audit.json"),
        ]
    )

    split_stats = _split_train_valid_with_size(
        input_path=pairs_sanitized,
        train_path=train_path,
        valid_path=valid_path,
        valid_ratio=args.valid_ratio,
        valid_size=args.valid_size,
        seed=args.seed,
    )
    (reports_dir / "split_stats.json").write_text(json.dumps(split_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _run(
        [
            sys.executable,
            "scripts/lint_data.py",
            "--path",
            str(train_path),
            "--max_len",
            str(args.max_len),
            "--last_k_steps",
            str(args.last_k_steps),
            "--step_markers",
            "step,action:,observation:,obs:",
            "--leak_terms_file",
            args.leak_terms_file,
            "--fail_on_leak",
            "--fail_on_missing",
            "--fail_on_truncation_risk",
            "--max_truncation_risk_last_k_steps",
            str(args.max_truncation_risk_last_k_steps),
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
        ]
    )

    _run(
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
            str(args.num_workers),
        ]
    )

    summary = {
        "pairs_unsanitized": str(pairs_unsanitized),
        "pairs_sanitized": str(pairs_sanitized),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "results_dir": str(results_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
