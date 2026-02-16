#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def _run(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan tail_drop_steps impact on pair construction")
    ap.add_argument("--raw_path", type=str, required=True)
    ap.add_argument("--adapter", type=str, default="real_logs_v1")
    ap.add_argument("--env_judge", type=str, default="real_logs")
    ap.add_argument("--work_dir", type=str, default="data/diagnostics/tail_scan")
    ap.add_argument("--steps", type=str, default="2,3,5,8")
    ap.add_argument("--max_pairs", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    work_root = Path(args.work_dir)
    work_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for k_str in [x.strip() for x in args.steps.split(",") if x.strip()]:
        k = int(k_str)
        run_dir = work_root / f"tail_{k}"
        run_dir.mkdir(parents=True, exist_ok=True)
        stats_path = run_dir / "export_stats.json"

        _run(
            [
                sys.executable,
                str(repo_root / "scripts" / "build_pairs_v030.py"),
                "--input_path",
                args.raw_path,
                "--output_path",
                str(run_dir / "pairs.all.jsonl"),
                "--trainable_output_path",
                str(run_dir / "pairs.trainable.unsanitized.jsonl"),
                "--adapter",
                args.adapter,
                "--env_judge",
                args.env_judge,
                "--forced_neg_type",
                "truncate_last_k",
                "--tail_drop_steps",
                str(k),
                "--max_pairs",
                str(args.max_pairs),
                "--seed",
                str(args.seed),
                "--stats_path",
                str(stats_path),
            ]
        )

        st = _read_json(stats_path)
        rows.append(
            {
                "tail_drop_steps": k,
                "pairs_out": st.get("pairs_out", 0),
                "pairs_trainable": st.get("pairs_trainable", 0),
                "judge_unknown_rate": st.get("judge_unknown_rate", 0.0),
                "env_judge_consistency": st.get("env_judge_consistency", 0.0),
                "neg_replay_success_rate": st.get("neg_replay_success_rate", 0.0),
            }
        )

    out_csv = work_root / "tail_drop_scan.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tail_drop_steps",
                "pairs_out",
                "pairs_trainable",
                "judge_unknown_rate",
                "env_judge_consistency",
                "neg_replay_success_rate",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(json.dumps({"rows": rows, "out_csv": str(out_csv)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
