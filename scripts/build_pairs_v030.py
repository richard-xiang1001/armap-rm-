#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

# Allow `python3 scripts/build_pairs_v030.py` from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingest.envs.arithmetic_env import judge_arithmetic_episode
from ingest.negatives.registry import NEG_ORDER, generate_negative_with_fallback
from ingest.refine_instruction import refine_instruction


def _normalize_step_lines(steps: List[str]) -> str:
    lines: List[str] = []
    step_idx = 1
    for raw in steps:
        text = str(raw or "").strip()
        if not text:
            continue
        lower = text.lower()
        if lower.startswith("task:"):
            lines.append(text)
            continue
        if lower.startswith("final:"):
            lines.append(text)
            continue
        match = re.match(r"(?i)^step\s+\d+\s*:\s*(.*)$", text)
        body = match.group(1).strip() if match else text
        lines.append(f"Step {step_idx}: {body}")
        step_idx += 1
    return "\n".join(lines).strip()


def _iter_adapter_episodes(adapter: str, input_path: str) -> Iterable[Dict]:
    module_name = f"ingest.adapters.{adapter}"
    module = importlib.import_module(module_name)
    if not hasattr(module, "iter_episodes"):
        raise RuntimeError(f"Adapter {adapter} has no iter_episodes(path) function")
    return module.iter_episodes(input_path)


def _build_pair(pos: Dict, neg: Dict, meta: Dict) -> Dict | None:
    instruction_raw = str(pos.get("instruction_raw", "") or "").strip()
    instruction_refined = str(pos.get("instruction_refined", instruction_raw) or instruction_raw).strip()
    traj_pos = _normalize_step_lines(pos.get("steps", []))
    traj_neg = _normalize_step_lines(neg.get("steps", []))
    if not instruction_refined or not traj_pos or not traj_neg:
        return None

    pair_meta = {
        "env": pos.get("env", "unknown_env"),
        "task_id": pos.get("task_id"),
        "seed": pos.get("seed"),
        "pos_run_id": pos.get("run_id"),
        "neg_run_id": neg.get("run_id"),
        "pos_episode_id": pos.get("episode_id"),
        "neg_episode_id": neg.get("episode_id"),
        "pos_episode_len": len(pos.get("steps", [])),
        "neg_episode_len": len(neg.get("steps", [])),
        "pos_success": bool(pos.get("success") is True),
        "neg_success": bool(neg.get("success") is True),
        # Keep old key for backward compatibility.
        "neg_strategy": meta.get("neg_type", "unknown"),
        # v0.3.0 additions.
        "construction_method": meta.get("construction_method", "v030_negative_pipeline"),
        "neg_type": meta.get("neg_type", "unknown"),
        "replay_ok": bool(meta.get("replay_ok", False)),
        "replay_attempted": bool(meta.get("replay_attempted", False)),
        "env_judge_pos": meta.get("env_judge_pos"),
        "env_judge_neg": meta.get("env_judge_neg"),
        "env_judge_consistent": bool(meta.get("env_judge_consistent", False)),
        "instruction_refine_mode": meta.get("instruction_refine_mode", "rule"),
        "instruction_refine_changed": bool(meta.get("instruction_refine_changed", False)),
        "neg_attempted_types": meta.get("neg_attempted_types", []),
    }

    # Optional judge debug payloads to support audit/ablation.
    if "pos_judge_reason" in meta:
        pair_meta["pos_judge_reason"] = meta["pos_judge_reason"]
    if "neg_judge_reason" in meta:
        pair_meta["neg_judge_reason"] = meta["neg_judge_reason"]

    return {
        "instruction_raw": instruction_raw or instruction_refined,
        "instruction_refined": instruction_refined,
        "traj_pos": traj_pos,
        "traj_neg": traj_neg,
        "meta": pair_meta,
    }


def build_pairs_v030(
    input_path: str,
    output_path: str,
    adapter: str,
    seed: int,
    max_pairs: int,
    tail_drop_steps: int,
    max_negative_attempts_per_positive: int,
    refine_mode: str,
    forced_neg_type: str,
    stats_path: str = "",
) -> Dict:
    rng = random.Random(seed)
    episodes = list(_iter_adapter_episodes(adapter=adapter, input_path=input_path))

    pairs: List[Dict] = []
    skipped_no_negative = 0
    skipped_empty = 0
    success_candidates = 0

    neg_counter: Counter = Counter()
    attempted_counter: Counter = Counter()
    replay_attempted = 0
    replay_ok = 0
    env_consistent = 0

    for episode in episodes:
        if episode.get("success") is not True:
            continue

        success_candidates += 1

        instruction_raw = str(episode.get("instruction_raw") or "").strip()
        refined, refine_meta = refine_instruction(
            instruction_raw=instruction_raw,
            steps=episode.get("steps", []),
            mode=refine_mode,
        )
        pos = dict(episode)
        pos["instruction_refined"] = refined or instruction_raw

        pos_judge = judge_arithmetic_episode(
            instruction=pos.get("instruction_refined", ""),
            steps=[str(x) for x in pos.get("steps", [])],
            meta=pos.get("raw_meta"),
        )

        generated = generate_negative_with_fallback(
            episode=pos,
            rng=rng,
            tail_drop_steps=tail_drop_steps,
            max_attempts=max_negative_attempts_per_positive,
            forced_neg_type=forced_neg_type,
        )
        if generated is None:
            skipped_no_negative += 1
            continue

        neg_steps = generated.get("steps")
        if not isinstance(neg_steps, list) or not neg_steps:
            skipped_no_negative += 1
            continue

        neg = dict(pos)
        neg["steps"] = [str(x) for x in neg_steps]
        neg["success"] = False
        neg["run_id"] = f"{pos.get('run_id', 'run')}::{generated.get('neg_type', 'neg')}"

        env_judge_pos = pos_judge.success
        env_judge_neg = generated.get("env_judge_neg")
        env_judge_consistent = env_judge_pos is True and env_judge_neg is False

        attempted_types = list(generated.get("attempted_neg_types", []))
        for name in attempted_types:
            attempted_counter[name] += 1

        attempted_flag = bool(generated.get("replay_attempted"))
        replay_attempted += int(attempted_flag)
        if attempted_flag and bool(generated.get("replay_ok")):
            replay_ok += 1
        env_consistent += int(bool(env_judge_consistent))
        neg_counter[str(generated.get("neg_type", "unknown"))] += 1

        pair = _build_pair(
            pos=pos,
            neg=neg,
            meta={
                "construction_method": "v030_negative_pipeline",
                "neg_type": generated.get("neg_type", "unknown"),
                "replay_ok": bool(generated.get("replay_ok", False)),
                "replay_attempted": bool(generated.get("replay_attempted", False)),
                "env_judge_pos": env_judge_pos,
                "env_judge_neg": env_judge_neg,
                "env_judge_consistent": env_judge_consistent,
                "instruction_refine_mode": refine_meta.get("mode", refine_mode),
                "instruction_refine_changed": bool(refine_meta.get("changed", False)),
                "neg_attempted_types": attempted_types,
                "pos_judge_reason": pos_judge.reason,
                "neg_judge_reason": generated.get("judge_reason", ""),
            },
        )
        if pair is None:
            skipped_empty += 1
            continue
        pairs.append(pair)

        if max_pairs > 0 and len(pairs) >= max_pairs:
            break

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    replay_rate = float(replay_ok / replay_attempted) if replay_attempted > 0 else 0.0
    env_consistency_rate = float(env_consistent / max(len(pairs), 1))
    stats = {
        "input_path": input_path,
        "output_path": output_path,
        "adapter": adapter,
        "seed": seed,
        "episodes_in": len(episodes),
        "success_candidates": success_candidates,
        "pairs_out": len(pairs),
        "forced_neg_type": forced_neg_type or "",
        "neg_order": list(NEG_ORDER),
        "neg_type_counts": dict(neg_counter),
        "neg_attempted_counts": dict(attempted_counter),
        "neg_replay_attempted": replay_attempted,
        "neg_replay_ok_count": replay_ok,
        "neg_replay_success_rate": replay_rate,
        "env_judge_consistent_count": env_consistent,
        "env_judge_consistency": env_consistency_rate,
        "skipped_no_negative": skipped_no_negative,
        "skipped_empty_pairs": skipped_empty,
    }
    if stats_path:
        path = Path(stats_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Build v0.3.0 RM pairs with ARMAP-style generated negatives.")
    ap.add_argument("--input_path", type=str, required=True)
    ap.add_argument("--output_path", type=str, required=True)
    ap.add_argument("--adapter", type=str, default="generic_jsonl", choices=["generic_jsonl", "armap_style", "webshop_like"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_pairs", type=int, default=0, help="0 means unlimited")
    ap.add_argument("--tail_drop_steps", type=int, default=2)
    ap.add_argument("--max_negative_attempts_per_positive", type=int, default=2)
    ap.add_argument("--refine_mode", type=str, default="rule", choices=["rule"])
    ap.add_argument(
        "--forced_neg_type",
        type=str,
        default="",
        choices=["", "replay_corrupt", "swap_step", "truncate_last_k"],
        help="Optional: force one neg type for ablation.",
    )
    ap.add_argument("--stats_path", type=str, default="")
    args = ap.parse_args()

    stats = build_pairs_v030(
        input_path=args.input_path,
        output_path=args.output_path,
        adapter=args.adapter,
        seed=args.seed,
        max_pairs=args.max_pairs,
        tail_drop_steps=args.tail_drop_steps,
        max_negative_attempts_per_positive=args.max_negative_attempts_per_positive,
        refine_mode=args.refine_mode,
        forced_neg_type=args.forced_neg_type,
        stats_path=args.stats_path,
    )
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
