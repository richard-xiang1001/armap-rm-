#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


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


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, text=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run v0.6.1 SWE Phase1+ matrix (pairs x seeds x steps_delta).")
    ap.add_argument("--raw_path", type=str, default="data/raw/episodes_swe_v060.jsonl")
    ap.add_argument("--max_pairs_values", type=str, default="10000,30000")
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--steps_delta_values", type=str, default="30,50,80")
    ap.add_argument("--min_pair_accuracy", type=float, default=0.0)
    ap.add_argument("--min_gap_p50", type=float, default=-1.0)
    ap.add_argument("--parity_threshold", type=float, default=1.0)
    ap.add_argument("--work_dir", type=str, default="data/real_v061_swe_phase1p")
    ap.add_argument("--results_dir", type=str, default="results/v061_swe_phase1p")
    ap.add_argument("--output_json", type=str, default="results/v061_swe_phase1p/matrix_raw.json")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    max_pairs_values = _parse_int_list(args.max_pairs_values)

    runs: List[Dict[str, Any]] = []
    phase1p_reports: List[Dict[str, Any]] = []

    for max_pairs in max_pairs_values:
        run_key = f"pairs_{max_pairs}"
        batch_work_dir = Path(args.work_dir) / run_key
        batch_results_dir = Path(args.results_dir) / run_key
        phase1p_report_path = batch_work_dir / "reports" / f"v061_phase1_plus_{run_key}.json"

        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "run_v060_swe_phase1_plus.py"),
            "--raw_path",
            str(args.raw_path),
            "--max_pairs",
            str(max_pairs),
            "--seeds",
            str(args.seeds),
            "--steps_delta_values",
            str(args.steps_delta_values),
            "--min_pair_accuracy",
            str(args.min_pair_accuracy),
            "--min_gap_p50",
            str(args.min_gap_p50),
            "--parity_threshold",
            str(args.parity_threshold),
            "--work_dir",
            str(batch_work_dir),
            "--results_dir",
            str(batch_results_dir),
            "--report_path",
            str(phase1p_report_path),
        ]
        _run(cmd)

        phase1p_report = _load_json(phase1p_report_path)
        phase1p_reports.append(
            {
                "max_pairs": max_pairs,
                "report_path": str(phase1p_report_path),
            }
        )
        for row in phase1p_report.get("runs", []):
            run_id = str(row.get("run_id", "") or "")
            acceptance_path = batch_work_dir / run_id / "reports" / "v060_swe_acceptance.json"
            acceptance = _load_json(acceptance_path)
            lint = acceptance.get("precheck_report", {}).get("lint_stats", {})

            runs.append(
                {
                    "max_pairs": max_pairs,
                    "seed": int(row.get("seed", 0)),
                    "max_same_task_steps_delta": int(row.get("max_same_task_steps_delta", 0)),
                    "run_id": run_id,
                    "work_dir": str(batch_work_dir / run_id),
                    "results_dir": str(batch_results_dir / run_id),
                    "acceptance_report_path": str(acceptance_path),
                    "passed": bool(row.get("passed", False)),
                    "pair_accuracy": float(row.get("pair_accuracy", 0.0)),
                    "gap_p50": float(row.get("gap_p50", 0.0)),
                    "steps_delta_abs_p90": float(lint.get("steps_delta_abs_p90", 0.0)),
                    "patch_delta_abs_p90": float(lint.get("patch_delta_abs_p90", 0.0)),
                    "tool_calls_delta_abs_p90": float(lint.get("tool_calls_delta_abs_p90", 0.0)),
                    "truncation_risk_last_k_steps": float(lint.get("truncation_risk_last_k_steps", 0.0)),
                    "judge_unknown_rate": float(lint.get("judge_unknown_rate", 1.0)),
                    "env_judge_consistency": float(lint.get("env_judge_consistency", 0.0)),
                }
            )

    out = {
        "raw_path": str(args.raw_path),
        "max_pairs_values": max_pairs_values,
        "seeds": _parse_int_list(args.seeds),
        "steps_delta_values": _parse_int_list(args.steps_delta_values),
        "runs": runs,
        "phase1p_reports": phase1p_reports,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(output_path), "n_runs": len(runs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
