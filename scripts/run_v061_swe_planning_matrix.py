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


def _select_ckpt_from_phase1(phase1_summary: Dict[str, Any], phase1_raw: Dict[str, Any]) -> Dict[str, Any]:
    groups = phase1_summary.get("groups", [])
    candidates = [
        g
        for g in groups
        if bool(g.get("all_acceptance_passed", False))
        and bool(g.get("hard_gate_all_ok", False))
        and bool(g.get("pair_acc_std_ok", False))
        and bool(g.get("all_gap_p50_positive", False))
    ]
    if not candidates:
        raise RuntimeError("No A-stage group passed all fixed gates for checkpoint selection.")

    def _group_key(g: Dict[str, Any]) -> tuple:
        max_pairs = int(g.get("max_pairs", 0))
        delta = int(g.get("max_same_task_steps_delta", 0))
        pair_mean = float(g.get("pair_accuracy", {}).get("mean", 0.0))
        pair_std = float(g.get("pair_accuracy", {}).get("std", 1.0))
        gap_mean = float(g.get("gap_p50", {}).get("mean", 0.0))
        prefer_30000 = 1 if max_pairs == 30000 else 0
        return (-pair_mean, pair_std, -gap_mean, -prefer_30000, -max_pairs, abs(delta - 30), delta)

    best_group = sorted(candidates, key=_group_key)[0]
    best_pairs = int(best_group["max_pairs"])
    best_delta = int(best_group["max_same_task_steps_delta"])

    matching_runs = [
        r
        for r in phase1_raw.get("runs", [])
        if int(r.get("max_pairs", -1)) == best_pairs
        and int(r.get("max_same_task_steps_delta", -1)) == best_delta
        and bool(r.get("passed", False))
    ]
    if not matching_runs:
        raise RuntimeError(f"No matching run found in matrix_raw for selected group: pairs={best_pairs}, delta={best_delta}")

    def _run_key(r: Dict[str, Any]) -> tuple:
        seed = int(r.get("seed", 0))
        return (0 if seed == 42 else 1, -float(r.get("pair_accuracy", 0.0)), -float(r.get("gap_p50", 0.0)), str(r.get("run_id", "")))

    for row in sorted(matching_runs, key=_run_key):
        ckpt = Path(str(row.get("results_dir", ""))) / "rm.pt"
        if ckpt.exists():
            return {
                "selection_mode": "phase1_auto",
                "selected_group": {
                    "max_pairs": best_pairs,
                    "max_same_task_steps_delta": best_delta,
                    "pair_accuracy_mean": float(best_group["pair_accuracy"]["mean"]),
                    "pair_accuracy_std": float(best_group["pair_accuracy"]["std"]),
                    "gap_p50_mean": float(best_group["gap_p50"]["mean"]),
                },
                "selected_run": {
                    "run_id": str(row.get("run_id", "")),
                    "seed": int(row.get("seed", 0)),
                    "results_dir": str(row.get("results_dir", "")),
                },
                "ckpt_path": str(ckpt),
            }
    raise RuntimeError("No existing rm.pt checkpoint found under selected group runs.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run v0.6.1 SWE planning matrix (multi-seed) with fixed ckpt selection policy.")
    ap.add_argument("--phase1_matrix_raw", type=str, default="results/v061_swe_phase1p/matrix_raw.json")
    ap.add_argument("--phase1_summary", type=str, default="results/v061_swe_phase1p/matrix_summary.json")
    ap.add_argument("--ckpt", type=str, default="", help="Optional override checkpoint path. If empty, auto-select from phase1 reports.")
    ap.add_argument("--raw_path", type=str, default="data/raw/episodes_swe_v060.jsonl")
    ap.add_argument("--results_dir", type=str, default="results/v061_swe_planning_matrix")
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--k_candidates", type=int, default=8)
    ap.add_argument("--n_tasks", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--max_visible_steps", type=int, default=0)
    ap.add_argument("--max_chars_per_step", type=int, default=64)
    ap.add_argument("--tail_block_min_keep", type=int, default=3)
    ap.add_argument("--preserve_tail_markers", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--task_slice", type=str, default="mixed", choices=["mixed", "all"])
    ap.add_argument("--emit_all_slice_report", type=str, default="true", choices=["true", "false"])
    ap.add_argument("--n_bootstrap", type=int, default=2000)
    ap.add_argument("--bootstrap_seed", type=int, default=-1)
    ap.add_argument("--success_delta_threshold", type=float, default=0.03)
    ap.add_argument("--audit_abs_corr_threshold", type=float, default=0.20)
    ap.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--forbidden_markers_file", type=str, default="configs/forbidden_markers_v060.json")
    ap.add_argument("--leak_terms_file", type=str, default="configs/leak_terms_v050.txt")
    ap.add_argument("--output_json", type=str, default="results/v061_swe_planning_matrix/matrix_raw.json")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    seeds = _parse_int_list(args.seeds)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.ckpt:
        ckpt_selection = {
            "selection_mode": "manual",
            "ckpt_path": str(args.ckpt),
        }
    else:
        phase1_summary = _load_json(Path(args.phase1_summary))
        phase1_raw = _load_json(Path(args.phase1_matrix_raw))
        ckpt_selection = _select_ckpt_from_phase1(phase1_summary=phase1_summary, phase1_raw=phase1_raw)

    ckpt_path = Path(str(ckpt_selection["ckpt_path"]))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    per_seed_reports: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_dir = results_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        report_path = seed_dir / "planning_eval.json"

        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "run_v061_swe_planning_eval.py"),
            "--raw_path",
            str(args.raw_path),
            "--ckpt",
            str(ckpt_path),
            "--results_dir",
            str(seed_dir),
            "--report_path",
            str(report_path),
            "--seed",
            str(seed),
            "--k_candidates",
            str(args.k_candidates),
            "--n_tasks",
            str(args.n_tasks),
            "--batch_size",
            str(args.batch_size),
            "--max_len",
            str(args.max_len),
            "--max_visible_steps",
            str(args.max_visible_steps),
            "--max_chars_per_step",
            str(args.max_chars_per_step),
            "--tail_block_min_keep",
            str(args.tail_block_min_keep),
            "--task_slice",
            str(args.task_slice),
            "--emit_all_slice_report",
            str(args.emit_all_slice_report),
            "--n_bootstrap",
            str(args.n_bootstrap),
            "--bootstrap_seed",
            str(args.bootstrap_seed if args.bootstrap_seed >= 0 else seed),
            "--success_delta_threshold",
            str(args.success_delta_threshold),
            "--audit_abs_corr_threshold",
            str(args.audit_abs_corr_threshold),
            "--device",
            str(args.device),
            "--forbidden_markers_file",
            str(args.forbidden_markers_file),
            "--leak_terms_file",
            str(args.leak_terms_file),
        ]
        if args.preserve_tail_markers:
            cmd.append("--preserve_tail_markers")
        else:
            cmd.append("--no-preserve_tail_markers")
        _run(cmd)

        report = _load_json(report_path)
        per_seed_reports.append(
            {
                "seed": seed,
                "report_path": str(report_path),
                "passed": bool(report.get("passed", False)),
                "selected_slice": str(report.get("selected_slice", args.task_slice)),
                "delta_rm_vs_best": float(report.get("delta_vs_baselines", {}).get("rm_minus_best_baseline", 0.0)),
                "delta_ci_low": float(report.get("delta_ci", {}).get("ci_low", 0.0)),
                "delta_ci_high": float(report.get("delta_ci", {}).get("ci_high", 0.0)),
                "rm_score_steps_len_corr": float(report.get("audit_metrics", {}).get("rm_score_steps_len_corr", 0.0)),
                "rm_score_token_proxy_corr": float(report.get("audit_metrics", {}).get("rm_score_token_proxy_corr", 0.0)),
            }
        )

    out = {
        "ckpt_selection": ckpt_selection,
        "raw_path": str(args.raw_path),
        "seeds": seeds,
        "task_slice": args.task_slice,
        "k_candidates": args.k_candidates,
        "n_tasks": args.n_tasks,
        "thresholds": {
            "success_delta_threshold": args.success_delta_threshold,
            "audit_abs_corr_threshold": args.audit_abs_corr_threshold,
        },
        "per_seed_reports": per_seed_reports,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(output_path), "n_seeds": len(per_seed_reports)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
