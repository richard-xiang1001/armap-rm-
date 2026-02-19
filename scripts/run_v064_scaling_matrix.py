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


def _parse_int_list(csv_text: str) -> List[int]:
    out: List[int] = []
    for token in str(csv_text or "").split(","):
        t = token.strip()
        if not t:
            continue
        out.append(int(t))
    if not out:
        raise ValueError("Expected at least one integer in CSV.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run v0.6.4 scaling matrix over bon_n x seeds on fixed task set.")
    ap.add_argument("--raw_path", type=str, default="data/raw/episodes_swe_v060.jsonl")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--swe_lite_manifest_path", type=str, required=True)
    ap.add_argument("--docker_image", type=str, default="swebench/swebench-lite:latest")
    ap.add_argument("--workspace_root", type=str, default="/tmp/v064_scaling")
    ap.add_argument("--results_dir", type=str, default="results/v064_scaling")
    ap.add_argument("--output_json", type=str, default="results/v064_scaling/matrix_raw.json")
    ap.add_argument("--bon_n_grid", type=str, default="1,2,4,8")
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--n_tasks", type=int, default=100)
    ap.add_argument("--fixed_horizon", type=int, default=1)
    ap.add_argument("--fixed_max_env_steps", type=int, default=1)
    ap.add_argument("--max_task_runtime_sec", type=int, default=600)
    ap.add_argument("--max_patch_chars", type=int, default=50000)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--n_bootstrap", type=int, default=2000)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--emit_step_candidates", type=str, default="false", choices=["true", "false"])
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = _read_jsonl(Path(args.swe_lite_manifest_path))
    if not manifest_rows:
        raise RuntimeError(f"Empty manifest: {args.swe_lite_manifest_path}")
    task_ids = [str(x.get("instance_id") or "").strip() for x in manifest_rows if str(x.get("instance_id") or "").strip()]
    if not task_ids:
        raise RuntimeError("Manifest has no valid instance_id rows.")
    task_ids = sorted(task_ids)[: max(int(args.n_tasks), 1)]
    task_ids_csv = ",".join(task_ids)

    bon_grid = _parse_int_list(args.bon_n_grid)
    seed_list = _parse_int_list(args.seeds)

    cells: List[Dict[str, Any]] = []
    for bon_n in bon_grid:
        for seed in seed_list:
            cell_dir = results_dir / f"bon_n_{bon_n}" / f"seed_{seed}"
            traces_path = cell_dir / "online_traces.jsonl"
            eval_path = cell_dir / "online_eval.json"

            _run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "run_v063_swe_bon_online.py"),
                    "--raw_path",
                    str(args.raw_path),
                    "--ckpt",
                    str(args.ckpt),
                    "--results_dir",
                    str(cell_dir),
                    "--env_backend",
                    "swe_real",
                    "--swe_lite_manifest_path",
                    str(args.swe_lite_manifest_path),
                    "--docker_image",
                    str(args.docker_image),
                    "--workspace_root",
                    str(Path(args.workspace_root) / f"bon_n_{bon_n}" / f"seed_{seed}"),
                    "--max_task_runtime_sec",
                    str(args.max_task_runtime_sec),
                    "--max_patch_chars",
                    str(args.max_patch_chars),
                    "--task_ids",
                    task_ids_csv,
                    "--seed",
                    str(seed),
                    "--bon_n",
                    str(bon_n),
                    "--rollout_horizon",
                    str(args.fixed_horizon),
                    "--max_env_steps",
                    str(args.fixed_max_env_steps),
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

            rep = _read_json(eval_path)
            eff = dict(rep.get("efficiency_metrics", {}).get("rm_bon", {}))
            ex = dict(rep.get("executor_health", {}))
            cells.append(
                {
                    "bon_n": int(bon_n),
                    "seed": int(seed),
                    "n_tasks": int(rep.get("n_tasks", 0)),
                    "success_rate_by_policy": dict(rep.get("success_rate_by_policy", {})),
                    "delta": float(rep.get("delta_vs_baselines", {}).get("rm_minus_best_baseline", 0.0)),
                    "ci_low": float(rep.get("delta_ci", {}).get("ci_low", 0.0)),
                    "ci_high": float(rep.get("delta_ci", {}).get("ci_high", 0.0)),
                    "avg_wall_time_sec": float(eff.get("avg_wall_time_sec", 0.0)),
                    "avg_rm_forward_calls": float(eff.get("avg_rm_forward_calls", 0.0)),
                    "timeout_rate": float(ex.get("timeout_rate", 0.0) or 0.0),
                    "passed": bool(rep.get("passed", False)),
                    "eval_path": str(eval_path),
                }
            )

    by_bon: Dict[int, List[Dict[str, Any]]] = {}
    for cell in cells:
        by_bon.setdefault(int(cell["bon_n"]), []).append(cell)

    aggregate_by_bon: List[Dict[str, Any]] = []
    for bon_n in sorted(by_bon.keys()):
        group = by_bon[bon_n]
        n = len(group)
        aggregate_by_bon.append(
            {
                "bon_n": bon_n,
                "n_cells": n,
                "delta_mean": float(sum(float(x["delta"]) for x in group) / n) if n else 0.0,
                "ci_low_mean": float(sum(float(x["ci_low"]) for x in group) / n) if n else 0.0,
                "avg_wall_time_sec_mean": float(sum(float(x["avg_wall_time_sec"]) for x in group) / n) if n else 0.0,
                "avg_rm_forward_calls_mean": float(sum(float(x["avg_rm_forward_calls"]) for x in group) / n) if n else 0.0,
                "timeout_rate_mean": float(sum(float(x["timeout_rate"]) for x in group) / n) if n else 0.0,
            }
        )

    payload = {
        "version": "v0.6.4-scaling",
        "raw_path": str(args.raw_path),
        "ckpt": str(args.ckpt),
        "swe_lite_manifest_path": str(args.swe_lite_manifest_path),
        "task_ids": task_ids,
        "n_tasks": len(task_ids),
        "bon_n_grid": bon_grid,
        "seeds": seed_list,
        "fixed_horizon": int(args.fixed_horizon),
        "fixed_max_env_steps": int(args.fixed_max_env_steps),
        "cells": cells,
        "aggregate_by_bon_n": aggregate_by_bon,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(output_path), "n_cells": len(cells)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
