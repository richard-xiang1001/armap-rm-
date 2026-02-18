#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def _run(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, text=True)


def _parse_int_list(csv_text: str) -> List[int]:
    out: List[int] = []
    for token in str(csv_text).split(","):
        t = token.strip()
        if not t:
            continue
        out.append(int(t))
    if not out:
        raise ValueError("Expected at least one integer value.")
    return out


def _mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {"mean": float(statistics.mean(values)), "std": float(statistics.pstdev(values))}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run v0.6.0 Phase1+ stability sweep (multi-seed, multi-delta).")
    ap.add_argument("--raw_path", type=str, default="data/raw/episodes_swe_v060.jsonl")
    ap.add_argument("--max_pairs", type=int, default=10000)
    ap.add_argument("--target_valid_size", type=int, default=200)
    ap.add_argument("--target_train_size", type=int, default=0, help="0 => max_pairs - target_valid_size")
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--steps_delta_values", type=str, default="50,30")
    ap.add_argument("--min_pair_accuracy", type=float, default=0.55)
    ap.add_argument("--min_gap_p50", type=float, default=0.0)
    ap.add_argument("--parity_threshold", type=float, default=0.03)
    ap.add_argument("--work_dir", type=str, default="data/real_v060_swe_phase1p")
    ap.add_argument("--results_dir", type=str, default="results/v060_swe_phase1p")
    ap.add_argument("--report_path", type=str, default="data/real_v060_swe_phase1p/reports/v060_swe_phase1_plus.json")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    seeds = _parse_int_list(args.seeds)
    deltas = _parse_int_list(args.steps_delta_values)
    target_valid_size = int(args.target_valid_size)
    if target_valid_size <= 0:
        raise ValueError("--target_valid_size must be > 0")
    target_train_size = int(args.target_train_size)
    if target_train_size <= 0:
        target_train_size = int(args.max_pairs) - target_valid_size
    if target_train_size <= 0:
        raise ValueError("Resolved target_train_size must be > 0")

    rows: List[Dict] = []
    for delta in deltas:
        for seed in seeds:
            run_id = f"s{seed}_d{delta}"
            work_dir = Path(args.work_dir) / run_id
            results_dir = Path(args.results_dir) / run_id
            reports_dir = work_dir / "reports"
            precheck_report = reports_dir / "v060_swe_precheck.json"
            acceptance_report = reports_dir / "v060_swe_acceptance.json"

            cmd = [
                sys.executable,
                str(repo_root / "scripts" / "run_v060_swe_acceptance.py"),
                "--raw_path",
                args.raw_path,
                "--seed",
                str(seed),
                "--max_pairs",
                str(args.max_pairs),
                "--target_train_size",
                str(target_train_size),
                "--target_valid_size",
                str(target_valid_size),
                "--max_same_task_steps_delta",
                str(delta),
                "--work_dir",
                str(work_dir),
                "--results_dir",
                str(results_dir),
                "--precheck_report_path",
                str(precheck_report),
                "--report_path",
                str(acceptance_report),
                "--device",
                "auto",
                "--amp_dtype",
                "none",
                "--min_pair_accuracy",
                str(args.min_pair_accuracy),
                "--min_gap_p50",
                str(args.min_gap_p50),
                "--parity_threshold",
                str(args.parity_threshold),
            ]
            _run(cmd)
            report = json.loads(acceptance_report.read_text(encoding="utf-8"))
            lint = report.get("precheck_report", {}).get("lint_stats", {})
            rows.append(
                {
                    "run_id": run_id,
                    "seed": seed,
                    "max_same_task_steps_delta": delta,
                    "passed": bool(report.get("passed", False)),
                    "pair_accuracy": float(report.get("gpu_eval", {}).get("pair_accuracy", 0.0)),
                    "gap_p50": float(report.get("gpu_eval", {}).get("gap_p50", 0.0)),
                    "steps_delta_abs_p90": float(lint.get("steps_delta_abs_p90", 0.0)),
                    "truncation_risk_last_k_steps": float(lint.get("truncation_risk_last_k_steps", 0.0)),
                }
            )

    groups: Dict[int, List[Dict]] = {}
    for row in rows:
        groups.setdefault(int(row["max_same_task_steps_delta"]), []).append(row)

    summary_by_delta: Dict[str, Dict] = {}
    for delta, items in groups.items():
        pair_vals = [float(x["pair_accuracy"]) for x in items]
        gap_vals = [float(x["gap_p50"]) for x in items]
        step_vals = [float(x["steps_delta_abs_p90"]) for x in items]
        trunc_vals = [float(x["truncation_risk_last_k_steps"]) for x in items]
        summary_by_delta[str(delta)] = {
            "n_runs": len(items),
            "all_passed": all(bool(x["passed"]) for x in items),
            "pair_accuracy": _mean_std(pair_vals),
            "gap_p50": _mean_std(gap_vals),
            "steps_delta_abs_p90": _mean_std(step_vals),
            "truncation_risk_last_k_steps": _mean_std(trunc_vals),
        }

    report = {
        "raw_path": args.raw_path,
        "max_pairs": args.max_pairs,
        "target_train_size": target_train_size,
        "target_valid_size": target_valid_size,
        "seeds": seeds,
        "steps_delta_values": deltas,
        "runs": rows,
        "summary_by_delta": summary_by_delta,
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report_path": str(report_path), "runs": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
