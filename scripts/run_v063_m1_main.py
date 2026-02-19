#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _run(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, text=True)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _parse_seed_list(seeds_csv: str, fallback_seed: int) -> List[int]:
    items = [x.strip() for x in str(seeds_csv or "").split(",") if x.strip()]
    if not items:
        return [int(fallback_seed)]
    return [int(x) for x in items]


def _aggregate_bool_checks(per_seed_reports: List[Dict[str, Any]], key: str) -> Dict[str, bool]:
    all_keys = sorted({k for row in per_seed_reports for k in (row.get(key, {}) or {}).keys()})
    out: Dict[str, bool] = {}
    for ck in all_keys:
        out[ck] = all(bool((row.get(key, {}) or {}).get(ck, False)) for row in per_seed_reports)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run v0.6.3 M1 main pipeline (manifest -> online -> eval -> summary).")
    ap.add_argument("--raw_path", type=str, default="data/raw/episodes_swe_v060.jsonl")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--swe_lite_manifest_path", type=str, required=True)
    ap.add_argument("--docker_image", type=str, default="swebench/swebench-lite:latest")
    ap.add_argument("--workspace_root", type=str, default="/tmp/v063_swe_real_m1")
    ap.add_argument("--results_dir", type=str, default="results/v063_swe_real_main")
    ap.add_argument("--n_tasks", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--bon_n", type=int, default=4)
    ap.add_argument("--rollout_horizon", type=int, default=1)
    ap.add_argument("--max_env_steps", type=int, default=1)
    ap.add_argument("--max_task_runtime_sec", type=int, default=600)
    ap.add_argument("--max_patch_chars", type=int, default=50000)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--emit_step_candidates", type=str, default="true", choices=["true", "false"])
    ap.add_argument("--n_bootstrap", type=int, default=2000)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = results_dir / "task_manifest_intersection.jsonl"
    drop_report_path = results_dir / "task_manifest_drop_report.json"

    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "build_v063_task_manifest.py"),
            "--raw_path",
            str(args.raw_path),
            "--swe_lite_manifest_path",
            str(args.swe_lite_manifest_path),
            "--output_path",
            str(manifest_path),
            "--drop_report_path",
            str(drop_report_path),
            "--max_tasks",
            str(args.n_tasks),
        ]
    )

    manifest_rows = _read_jsonl(manifest_path)
    if not manifest_rows:
        raise RuntimeError(f"Manifest intersection empty: {manifest_path}")
    task_ids = [str(x.get("instance_id") or "").strip() for x in manifest_rows if str(x.get("instance_id") or "").strip()]
    if not task_ids:
        raise RuntimeError(f"No valid instance_id in {manifest_path}")
    task_ids = task_ids[: max(int(args.n_tasks), 1)]
    task_ids_csv = ",".join(task_ids)

    seed_list = _parse_seed_list(args.seeds, fallback_seed=args.seed)
    per_seed_reports: List[Dict[str, Any]] = []

    for seed in seed_list:
        seed_dir = results_dir / f"seed_{seed}"
        traces_path = seed_dir / "online_traces.jsonl"
        eval_path = seed_dir / "online_eval.json"
        summary_json = seed_dir / "online_summary.json"
        summary_md = seed_dir / "online_summary.md"

        _run(
            [
                sys.executable,
                str(repo_root / "scripts" / "run_v063_swe_bon_online.py"),
                "--raw_path",
                str(args.raw_path),
                "--ckpt",
                str(args.ckpt),
                "--results_dir",
                str(seed_dir),
                "--env_backend",
                "swe_real",
                "--swe_lite_manifest_path",
                str(manifest_path),
                "--docker_image",
                str(args.docker_image),
                "--workspace_root",
                str(Path(args.workspace_root) / f"seed_{seed}"),
                "--max_task_runtime_sec",
                str(args.max_task_runtime_sec),
                "--max_patch_chars",
                str(args.max_patch_chars),
                "--task_ids",
                task_ids_csv,
                "--seed",
                str(seed),
                "--bon_n",
                str(args.bon_n),
                "--rollout_horizon",
                str(args.rollout_horizon),
                "--max_env_steps",
                str(args.max_env_steps),
                "--max_len",
                str(args.max_len),
                "--batch_size",
                str(args.batch_size),
                "--device",
                str(args.device),
                "--emit_step_candidates",
                str(args.emit_step_candidates),
            ]
        )

        _run(
            [
                sys.executable,
                str(repo_root / "scripts" / "eval_v062_online.py"),
                "--traces_path",
                str(traces_path),
                "--output_json",
                str(eval_path),
                "--n_bootstrap",
                str(args.n_bootstrap),
                "--bootstrap_seed",
                str(seed),
            ]
        )

        _run(
            [
                sys.executable,
                str(repo_root / "scripts" / "summarize_v062_online.py"),
                "--eval_json",
                str(eval_path),
                "--output_json",
                str(summary_json),
                "--output_md",
                str(summary_md),
            ]
        )

        rep = _read_json(eval_path)
        per_seed_reports.append(
            {
                "seed": int(seed),
                "eval_path": str(eval_path),
                "summary_md": str(summary_md),
                "n_tasks": int(rep.get("n_tasks", 0)),
                "passed": bool(rep.get("passed", False)),
                "planning_checks": dict(rep.get("planning_checks", {})),
                "executor_health_checks": dict(rep.get("executor_health_checks", {})),
                "delta": float(rep.get("delta_vs_baselines", {}).get("rm_minus_best_baseline", 0.0)),
                "ci_low": float(rep.get("delta_ci", {}).get("ci_low", 0.0)),
                "executor_health": dict(rep.get("executor_health", {})),
            }
        )

    planning_checks = _aggregate_bool_checks(per_seed_reports, key="planning_checks")
    executor_health_checks = _aggregate_bool_checks(per_seed_reports, key="executor_health_checks")
    passed = all(planning_checks.values()) and all(executor_health_checks.values()) and bool(per_seed_reports)

    exec_error_breakdown: Dict[str, int] = {}
    for row in per_seed_reports:
        br = dict(row.get("executor_health", {}).get("exec_error_breakdown", {}))
        for k, v in br.items():
            exec_error_breakdown[str(k)] = int(exec_error_breakdown.get(str(k), 0)) + int(v)

    drop_report = _read_json(drop_report_path) if drop_report_path.exists() else {}
    manifest_coverage = {
        "selected_instances": len(task_ids),
        "intersection_instances": int(drop_report.get("stats", {}).get("intersection_instances", 0)),
        "raw_instances": int(drop_report.get("stats", {}).get("raw_instances", 0)),
        "manifest_instances": int(drop_report.get("stats", {}).get("manifest_instances", 0)),
        "drop_reason_counts": dict(drop_report.get("drop_reason_counts", {})),
    }

    report = {
        "version": "v0.6.3-m1-main",
        "raw_path": str(args.raw_path),
        "ckpt": str(args.ckpt),
        "swe_lite_manifest_path": str(args.swe_lite_manifest_path),
        "task_manifest_intersection_path": str(manifest_path),
        "drop_report_path": str(drop_report_path),
        "manifest_coverage": manifest_coverage,
        "seed_list": seed_list,
        "n_tasks": len(task_ids),
        "task_ids": task_ids,
        "planning_checks": planning_checks,
        "executor_health_checks": executor_health_checks,
        "exec_error_breakdown": exec_error_breakdown,
        "per_seed": per_seed_reports,
        "passed": bool(passed),
    }

    out_path = results_dir / "m1_main_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"m1_main_report": str(out_path), "passed": passed, "seeds": seed_list}, ensure_ascii=False))


if __name__ == "__main__":
    main()
