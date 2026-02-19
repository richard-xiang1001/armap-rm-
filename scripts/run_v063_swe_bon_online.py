#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

# Allow `python3 scripts/run_v063_swe_bon_online.py` from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from online.envs.swe_real_env import SWERealOnlineEnv
from online.envs.swe_sim_env import SWESimOnlineEnv
from online.planners.bon import BoNPlanner
from online.policy.propose import propose_actions


def _parse_bool_text(value: str) -> bool:
    v = str(value or "").strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean text: {value}")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _task_seed(base_seed: int, policy: str, task_id: str) -> int:
    raw = f"{base_seed}:{policy}:{task_id}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def _build_env(args):
    if args.env_backend == "sim":
        return SWESimOnlineEnv(raw_path=args.raw_path)
    if not str(args.swe_lite_manifest_path or "").strip():
        raise ValueError("--swe_lite_manifest_path is required when --env_backend swe_real")
    return SWERealOnlineEnv(
        raw_path=args.raw_path,
        manifest_path=args.swe_lite_manifest_path,
        docker_image=args.docker_image,
        workspace_root=args.workspace_root,
        max_task_runtime_sec=args.max_task_runtime_sec,
        max_patch_chars=args.max_patch_chars,
        keep_failed_workdirs=_parse_bool_text(args.keep_failed_workdirs),
    )


def _baseline_step(
    env,
    policy: str,
    task_ctx: Dict[str, Any],
    history: List[Dict[str, Any]],
    bon_n: int,
    seed: int,
    emit_step_candidates: bool,
) -> Dict[str, Any]:
    start = time.perf_counter()
    obs_text = env.get_observation_text()
    obs_hash = hashlib.sha256(obs_text.encode("utf-8")).hexdigest()

    n = 1 if policy == "baseline_no_planning" else max(int(bon_n), 1)
    actions = propose_actions(obs_text=obs_text, history=history, n=n, seed=seed, task_ctx=task_ctx)
    if not actions:
        raise RuntimeError(f"No actions proposed for policy={policy}")

    if policy == "baseline_no_planning":
        selected = dict(actions[0])
    else:
        rng = random.Random(int(seed) + 19)
        selected = dict(actions[rng.randrange(len(actions))])

    selected["delta_steps"] = int(selected.get("delta_steps", 1))
    _, _, env_done, env_info = env.step(selected)
    planning_overhead_ms = (time.perf_counter() - start) * 1000.0

    candidates: List[Dict[str, Any]] = []
    if emit_step_candidates:
        for act in actions:
            patch_text = str(act.get("patch_text") or "")
            candidates.append(
                {
                    "candidate_id": str(act.get("candidate_id") or ""),
                    "instance_id": str(act.get("instance_id") or ""),
                    "action_text": str(act.get("action_text") or ""),
                    "rm_score": None,
                    "token_proxy": int(len(str(act.get("action_text") or "")) + len(patch_text)),
                    "sim_done": None,
                    "sim_info": {},
                }
            )

    return {
        "obs_hash": obs_hash,
        "candidates": candidates,
        "selected_candidate_id": str(selected.get("candidate_id") or ""),
        "selected_rm_score": None,
        "env_done": bool(env_done),
        "env_info": dict(env_info or {}),
        "planning_overhead_ms": float(planning_overhead_ms),
        "total_rm_forward_calls": 0,
    }


def _run_policy_for_task(
    env,
    planner: BoNPlanner,
    policy: str,
    task_id: str,
    seed: int,
    bon_n: int,
    rollout_horizon: int,
    max_env_steps: int,
    emit_step_candidates: bool,
) -> Dict[str, Any]:
    env.reset(task_id=task_id, seed=seed)
    start = time.perf_counter()

    steps: List[Dict[str, Any]] = []
    total_tokens = 0
    total_rm_calls = 0

    for t in range(max(int(max_env_steps), 1)):
        if env.is_done():
            break

        task_ctx = env.get_task_context()
        history = env.render_history()
        step_seed = int(seed) + t * 101

        if policy == "rm_bon":
            rec = planner.plan_step(
                env=env,
                task_ctx=task_ctx,
                history=history,
                bon_n=max(int(bon_n), 1),
                rollout_horizon=max(int(rollout_horizon), 1),
                seed=step_seed,
            )
            if not emit_step_candidates:
                rec["candidates"] = []
        else:
            rec = _baseline_step(
                env=env,
                policy=policy,
                task_ctx=task_ctx,
                history=history,
                bon_n=bon_n,
                seed=step_seed,
                emit_step_candidates=emit_step_candidates,
            )

        info = dict(rec.get("env_info") or {})
        total_tokens += int(len(str(info.get("committed_text") or "")) + int(info.get("patch_chars", 0)))
        total_rm_calls += int(rec.get("total_rm_forward_calls", 0))

        steps.append(
            {
                "t": t,
                "obs_hash": str(rec.get("obs_hash") or ""),
                "candidates": list(rec.get("candidates") or []),
                "selected_candidate_id": str(rec.get("selected_candidate_id") or ""),
                "selected_rm_score": rec.get("selected_rm_score"),
                "env_done": bool(rec.get("env_done", False)),
                "env_info": info,
                "planning_overhead_ms": float(rec.get("planning_overhead_ms", 0.0)),
            }
        )

    success = bool(env.get_terminal_success())
    wall_time = time.perf_counter() - start
    return {
        "task_id": task_id,
        "policy": policy,
        "success": success,
        "n_steps": len(steps),
        "total_token_proxy": int(total_tokens),
        "total_rm_forward_calls": int(total_rm_calls),
        "wall_time_sec": float(wall_time),
        "steps": steps,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run v0.6.3 SWE BoN online loop with env backend switch.")
    ap.add_argument("--raw_path", type=str, default="data/raw/episodes_swe_v060.jsonl")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--results_dir", type=str, default="results/v063_swe_real")
    ap.add_argument("--env_backend", type=str, default="swe_real", choices=["sim", "swe_real"])
    ap.add_argument("--swe_lite_manifest_path", type=str, default="results/v063_swe_real/task_manifest_intersection.jsonl")
    ap.add_argument("--docker_image", type=str, default="swebench/swebench-lite:latest")
    ap.add_argument("--workspace_root", type=str, default="/tmp/v063_swe_real")
    ap.add_argument("--max_task_runtime_sec", type=int, default=600)
    ap.add_argument("--max_patch_chars", type=int, default=50000)
    ap.add_argument("--keep_failed_workdirs", type=str, default="false", choices=["true", "false"])
    ap.add_argument("--n_tasks", type=int, default=100)
    ap.add_argument("--task_ids", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--planner", type=str, default="bon", choices=["bon"])
    ap.add_argument("--bon_n", type=int, default=4)
    ap.add_argument("--rollout_horizon", type=int, default=1)
    ap.add_argument("--max_env_steps", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--emit_step_candidates", type=str, default="true", choices=["true", "false"])
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    traces_path = results_dir / "online_traces.jsonl"
    meta_path = results_dir / "online_run_meta.json"

    env = _build_env(args)
    planner = BoNPlanner(ckpt=args.ckpt, device=args.device, max_len=args.max_len, batch_size=args.batch_size)

    emit_candidates = _parse_bool_text(args.emit_step_candidates)

    if str(args.task_ids).strip():
        selected_tasks = [x.strip() for x in str(args.task_ids).split(",") if x.strip()]
    else:
        all_task_ids = env.list_task_ids()
        rng = random.Random(int(args.seed))
        rng.shuffle(all_task_ids)
        selected_tasks = all_task_ids[: max(int(args.n_tasks), 1)]

    rows: List[Dict[str, Any]] = []
    policies = ["baseline_no_planning", "baseline_random_bon", "rm_bon"]
    t0 = time.perf_counter()
    for policy in policies:
        for task_id in selected_tasks:
            seed = _task_seed(base_seed=args.seed, policy=policy, task_id=task_id)
            rows.append(
                _run_policy_for_task(
                    env=env,
                    planner=planner,
                    policy=policy,
                    task_id=task_id,
                    seed=seed,
                    bon_n=args.bon_n,
                    rollout_horizon=args.rollout_horizon,
                    max_env_steps=args.max_env_steps,
                    emit_step_candidates=emit_candidates,
                )
            )

    _write_jsonl(traces_path, rows)

    payload = {
        "raw_path": str(args.raw_path),
        "env_backend": str(args.env_backend),
        "swe_lite_manifest_path": str(args.swe_lite_manifest_path),
        "docker_image": str(args.docker_image),
        "workspace_root": str(args.workspace_root),
        "max_task_runtime_sec": int(args.max_task_runtime_sec),
        "max_patch_chars": int(args.max_patch_chars),
        "ckpt": str(args.ckpt),
        "planner": str(args.planner),
        "seed": int(args.seed),
        "bon_n": int(args.bon_n),
        "rollout_horizon": int(args.rollout_horizon),
        "max_env_steps": int(args.max_env_steps),
        "max_len": int(args.max_len),
        "batch_size": int(args.batch_size),
        "device": str(args.device),
        "emit_step_candidates": emit_candidates,
        "n_tasks": len(selected_tasks),
        "policies": policies,
        "traces_path": str(traces_path),
        "wall_time_sec": float(time.perf_counter() - t0),
    }
    _write_json(meta_path, payload)
    print(json.dumps({"traces_path": str(traces_path), "meta_path": str(meta_path), "n_rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
