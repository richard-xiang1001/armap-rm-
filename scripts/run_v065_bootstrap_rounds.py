#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import statistics
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


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_int_list(csv_text: str) -> List[int]:
    out: List[int] = []
    for token in str(csv_text or "").split(","):
        t = token.strip()
        if not t:
            continue
        out.append(int(t))
    if not out:
        raise ValueError("Expected non-empty integer CSV list.")
    return out


def _last_jsonl_row(path: Path) -> Dict[str, Any]:
    last: Dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            last = json.loads(line)
    if last is None:
        raise RuntimeError(f"No JSON rows found in {path}")
    return last


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def _merge_traces(paths: List[Path], output_path: Path) -> int:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_jsonl(path))
    _write_jsonl(output_path, rows)
    return len(rows)


def _load_manifest_task_ids(path: Path, n_tasks: int) -> List[str]:
    rows = _read_jsonl(path)
    task_ids = [str(x.get("instance_id") or "").strip() for x in rows if str(x.get("instance_id") or "").strip()]
    task_ids = sorted(task_ids)
    task_ids = task_ids[: max(int(n_tasks), 1)]
    if not task_ids:
        raise RuntimeError(f"No task ids resolved from manifest: {path}")
    return task_ids


def _distribution_shift(prev_round: Dict[str, Any] | None, cur_round: Dict[str, Any]) -> Dict[str, Any]:
    if prev_round is None:
        return {
            "has_previous_round": False,
            "delta_shift": None,
            "ci_low_shift": None,
            "executor_health_shift": {},
        }

    prev_ex = dict(prev_round.get("executor_health_mean", {}))
    cur_ex = dict(cur_round.get("executor_health_mean", {}))
    keys = sorted(set(prev_ex.keys()) | set(cur_ex.keys()))
    ex_shift = {k: float(cur_ex.get(k, 0.0)) - float(prev_ex.get(k, 0.0)) for k in keys}

    return {
        "has_previous_round": True,
        "delta_shift": float(cur_round.get("delta_mean", 0.0)) - float(prev_round.get("delta_mean", 0.0)),
        "ci_low_shift": float(cur_round.get("ci_low_mean", 0.0)) - float(prev_round.get("ci_low_mean", 0.0)),
        "executor_health_shift": ex_shift,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run v0.6.5 bootstrap rounds: online -> pairs -> train -> online.")
    ap.add_argument("--base_ckpt", type=str, required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--raw_path", type=str, default="data/raw/episodes_swe_v060.jsonl")
    ap.add_argument("--swe_lite_manifest_path", type=str, required=True)
    ap.add_argument("--docker_image", type=str, default="swebench/swebench-lite:latest")
    ap.add_argument("--workspace_root", type=str, default="/tmp/v065_bootstrap")
    ap.add_argument("--results_dir", type=str, default="results/v065_bootstrap")
    ap.add_argument("--n_tasks", type=int, default=100)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--bon_n", type=int, default=4)
    ap.add_argument("--rollout_horizon", type=int, default=1)
    ap.add_argument("--max_env_steps", type=int, default=1)
    ap.add_argument("--max_task_runtime_sec", type=int, default=600)
    ap.add_argument("--max_patch_chars", type=int, default=50000)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--n_bootstrap", type=int, default=2000)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--emit_step_candidates", type=str, default="true", choices=["true", "false"])
    ap.add_argument("--pair_policy", type=str, default="task_success_then_rm_gap")
    ap.add_argument("--train_valid_ratio", type=float, default=0.1)
    ap.add_argument("--train_epochs", type=int, default=2)
    ap.add_argument("--train_lr", type=float, default=3e-4)
    ap.add_argument("--train_batch_size", type=int, default=64)
    ap.add_argument("--train_num_workers", type=int, default=2)
    ap.add_argument("--train_max_steps", type=int, default=0)
    ap.add_argument("--output_json", type=str, default="results/v065_bootstrap/bootstrap_summary.json")
    args = ap.parse_args()

    if int(args.rounds) < 1:
        raise ValueError("--rounds must be >= 1")

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    seed_list = _parse_int_list(args.seeds)
    task_ids = _load_manifest_task_ids(Path(args.swe_lite_manifest_path), n_tasks=int(args.n_tasks))
    task_ids_csv = ",".join(task_ids)

    current_ckpt = str(args.base_ckpt)
    rounds_out: List[Dict[str, Any]] = []
    prev_round_summary: Dict[str, Any] | None = None

    for round_idx in range(int(args.rounds) + 1):
        round_dir = results_dir / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)
        per_seed: List[Dict[str, Any]] = []
        seed_trace_paths: List[Path] = []

        for seed in seed_list:
            seed_dir = round_dir / f"seed_{seed}"
            traces_path = seed_dir / "online_traces.jsonl"
            eval_path = seed_dir / "online_eval.json"
            summary_md = seed_dir / "online_summary.md"

            _run(
                [
                    sys.executable,
                    str(repo_root / "scripts" / "run_v063_swe_bon_online.py"),
                    "--raw_path",
                    str(args.raw_path),
                    "--ckpt",
                    str(current_ckpt),
                    "--results_dir",
                    str(seed_dir),
                    "--env_backend",
                    "swe_real",
                    "--swe_lite_manifest_path",
                    str(args.swe_lite_manifest_path),
                    "--docker_image",
                    str(args.docker_image),
                    "--workspace_root",
                    str(Path(args.workspace_root) / f"round_{round_idx}" / f"seed_{seed}"),
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
                    str(seed_dir / "online_summary.json"),
                    "--output_md",
                    str(summary_md),
                ]
            )

            rep = _read_json(eval_path)
            per_seed.append(
                {
                    "seed": int(seed),
                    "eval_path": str(eval_path),
                    "summary_md": str(summary_md),
                    "delta": float(rep.get("delta_vs_baselines", {}).get("rm_minus_best_baseline", 0.0)),
                    "ci_low": float(rep.get("delta_ci", {}).get("ci_low", 0.0)),
                    "ci_high": float(rep.get("delta_ci", {}).get("ci_high", 0.0)),
                    "passed": bool(rep.get("passed", False)),
                    "planning_checks": dict(rep.get("planning_checks", {})),
                    "executor_health_checks": dict(rep.get("executor_health_checks", {})),
                    "executor_health": dict(rep.get("executor_health", {})),
                }
            )
            seed_trace_paths.append(traces_path)

        merged_traces_path = round_dir / "online_traces_merged.jsonl"
        merged_rows = _merge_traces(seed_trace_paths, merged_traces_path)

        round_summary: Dict[str, Any] = {
            "round": round_idx,
            "ckpt_in": str(current_ckpt),
            "n_tasks": len(task_ids),
            "seed_list": seed_list,
            "delta_mean": _mean([float(x["delta"]) for x in per_seed]),
            "delta_std": _std([float(x["delta"]) for x in per_seed]),
            "ci_low_mean": _mean([float(x["ci_low"]) for x in per_seed]),
            "ci_high_mean": _mean([float(x["ci_high"]) for x in per_seed]),
            "per_seed": per_seed,
            "merged_traces_path": str(merged_traces_path),
            "merged_trace_rows": int(merged_rows),
        }

        ex_health_keys = ["runnable_task_rate", "patch_apply_ok_rate", "test_exec_success_rate", "timeout_rate"]
        ex_health_mean = {}
        for key in ex_health_keys:
            vals = [float((x.get("executor_health", {}) or {}).get(key, 0.0) or 0.0) for x in per_seed]
            ex_health_mean[key] = _mean(vals)
        round_summary["executor_health_mean"] = ex_health_mean
        round_summary["executor_health_all_green"] = all(
            all(bool(v) for v in (x.get("executor_health_checks", {}) or {}).values()) for x in per_seed
        )

        pairs_train = round_dir / "pairs_train.jsonl"
        pairs_valid = round_dir / "pairs_valid.jsonl"
        pairs_stats = round_dir / "pairs_stats.json"

        _run(
            [
                sys.executable,
                str(repo_root / "scripts" / "build_v065_bootstrap_pairs.py"),
                "--traces_path",
                str(merged_traces_path),
                "--output_train_jsonl",
                str(pairs_train),
                "--output_valid_jsonl",
                str(pairs_valid),
                "--pair_policy",
                str(args.pair_policy),
                "--seed",
                str(seed_list[0]),
                "--valid_ratio",
                str(args.train_valid_ratio),
                "--stats_json",
                str(pairs_stats),
            ]
        )

        lint_report = round_dir / "pairs_lint_report.json"
        _run(
            [
                sys.executable,
                str(repo_root / "scripts" / "lint_data.py"),
                "--path",
                str(pairs_train),
                "--max_len",
                str(args.max_len),
                "--leak_terms_file",
                str(repo_root / "configs" / "leak_terms_v050.txt"),
                "--forbidden_markers_file",
                str(repo_root / "configs" / "forbidden_markers_v060.json"),
                "--fail_on_missing",
                "--report_path",
                str(lint_report),
            ]
        )

        train_metrics = None
        ckpt_out = str(current_ckpt)
        if round_idx < int(args.rounds):
            train_dir = round_dir / "train"
            _run(
                [
                    sys.executable,
                    "-m",
                    "rm.train",
                    "--train_path",
                    str(pairs_train),
                    "--valid_path",
                    str(pairs_valid),
                    "--save_dir",
                    str(train_dir),
                    "--max_len",
                    str(args.max_len),
                    "--batch_size",
                    str(args.train_batch_size),
                    "--epochs",
                    str(args.train_epochs),
                    "--lr",
                    str(args.train_lr),
                    "--num_workers",
                    str(args.train_num_workers),
                    "--seed",
                    str(seed_list[0]),
                    "--max_steps",
                    str(args.train_max_steps),
                    "--device",
                    str(args.device),
                    "--amp_dtype",
                    "none",
                    "--init_ckpt",
                    str(current_ckpt),
                ]
            )
            train_metrics = _last_jsonl_row(train_dir / "metrics.jsonl")
            ckpt_out = str(train_dir / "rm.pt")
            if not Path(ckpt_out).exists():
                raise FileNotFoundError(f"Round {round_idx} train output ckpt missing: {ckpt_out}")

        round_summary["pairs_train"] = str(pairs_train)
        round_summary["pairs_valid"] = str(pairs_valid)
        round_summary["pairs_stats"] = str(pairs_stats)
        round_summary["pairs_lint_report"] = str(lint_report)
        round_summary["train_metrics"] = train_metrics
        round_summary["ckpt_out"] = ckpt_out

        shift = _distribution_shift(prev_round_summary, round_summary)
        round_summary["distribution_shift_report"] = shift
        _write_json(round_dir / "distribution_shift_report.json", shift)
        _write_json(round_dir / "round_report.json", round_summary)

        rounds_out.append(round_summary)
        prev_round_summary = round_summary
        current_ckpt = ckpt_out

    delta_by_round = {str(r["round"]): float(r.get("delta_mean", 0.0)) for r in rounds_out}
    ci_by_round = {
        str(r["round"]): {
            "ci_low_mean": float(r.get("ci_low_mean", 0.0)),
            "ci_high_mean": float(r.get("ci_high_mean", 0.0)),
        }
        for r in rounds_out
    }
    executor_health_by_round = {str(r["round"]): dict(r.get("executor_health_mean", {})) for r in rounds_out}
    variance_by_seed = {str(r["round"]): float(r.get("delta_std", 0.0)) for r in rounds_out}

    round0 = rounds_out[0]
    round_last = rounds_out[-1]
    variance_growth_limit = float(round0.get("delta_std", 0.0)) * 1.25
    if float(round0.get("delta_std", 0.0)) == 0.0:
        variance_growth_limit = 0.0

    checks = {
        "final_delta_positive_vs_round0": float(round_last.get("delta_mean", 0.0)) > float(round0.get("delta_mean", 0.0)),
        "variance_not_expand_over_25pct": float(round_last.get("delta_std", 0.0)) <= float(variance_growth_limit),
        "all_rounds_executor_health_green": all(bool(r.get("executor_health_all_green", False)) for r in rounds_out),
    }
    passed = all(checks.values())

    summary = {
        "version": "v0.6.5-bootstrap",
        "raw_path": str(args.raw_path),
        "base_ckpt": str(args.base_ckpt),
        "rounds": int(args.rounds),
        "seed_list": seed_list,
        "task_ids": task_ids,
        "delta_by_round": delta_by_round,
        "ci_by_round": ci_by_round,
        "executor_health_by_round": executor_health_by_round,
        "variance_by_seed": variance_by_seed,
        "round_reports": rounds_out,
        "checks": checks,
        "passed": bool(passed),
    }

    out_path = Path(args.output_json)
    _write_json(out_path, summary)
    print(json.dumps({"output_json": str(out_path), "passed": passed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
