#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Allow `python3 ingest/export_pairs.py` from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def _tail_truncate(steps: List[str], tail_drop_steps: int) -> List[str] | None:
    if len(steps) <= 1:
        return None
    drop = max(1, min(tail_drop_steps, len(steps) - 1))
    new_steps = steps[:-drop]
    if not new_steps:
        return None
    return new_steps


def _group_key(ep: Dict) -> str:
    task_id = ep.get("task_id")
    if task_id:
        return str(task_id)
    refined = str(ep.get("instruction_refined", "") or "").strip()
    if refined:
        return refined
    return str(ep.get("instruction_raw", "") or "").strip()


def _build_pair(pos: Dict, neg: Dict, strategy: str) -> Dict | None:
    instruction_raw = str(pos.get("instruction_raw", "") or "").strip()
    instruction_refined = str(pos.get("instruction_refined", instruction_raw) or instruction_raw).strip()
    traj_pos = _normalize_step_lines(pos.get("steps", []))
    traj_neg = _normalize_step_lines(neg.get("steps", []))
    if not instruction_refined or not traj_pos or not traj_neg:
        return None
    meta = {
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
        "neg_strategy": strategy,
    }
    return {
        "instruction_raw": instruction_raw or instruction_refined,
        "instruction_refined": instruction_refined,
        "traj_pos": traj_pos,
        "traj_neg": traj_neg,
        "meta": meta,
    }


def _iter_adapter_episodes(adapter: str, input_path: str) -> Iterable[Dict]:
    module_name = f"ingest.adapters.{adapter}"
    module = importlib.import_module(module_name)
    if not hasattr(module, "iter_episodes"):
        raise RuntimeError(f"Adapter {adapter} has no iter_episodes(path) function")
    return module.iter_episodes(input_path)


def export_pairs(
    input_path: str,
    output_path: str,
    adapter: str,
    seed: int,
    tail_drop_steps: int,
    max_pairs: int,
    stats_path: str = "",
) -> Dict:
    rng = random.Random(seed)
    episodes = list(_iter_adapter_episodes(adapter=adapter, input_path=input_path))
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for ep in episodes:
        grouped[_group_key(ep)].append(ep)

    strategy_counter: Counter = Counter()
    pairs: List[Dict] = []
    skipped_empty = 0

    for _, eps in grouped.items():
        success_eps = [e for e in eps if e.get("success") is True]
        fail_eps = [e for e in eps if e.get("success") is False]

        if success_eps and fail_eps:
            rng.shuffle(fail_eps)
            for i, pos in enumerate(success_eps):
                neg = fail_eps[i % len(fail_eps)]
                pair = _build_pair(pos=pos, neg=neg, strategy="same_task_failure")
                if pair is None:
                    skipped_empty += 1
                    continue
                pairs.append(pair)
                strategy_counter["same_task_failure"] += 1
                if max_pairs > 0 and len(pairs) >= max_pairs:
                    break
        elif success_eps:
            for pos in success_eps:
                neg_steps = _tail_truncate(pos.get("steps", []), tail_drop_steps=tail_drop_steps)
                if not neg_steps:
                    continue
                neg = dict(pos)
                neg["steps"] = neg_steps
                neg["success"] = False
                neg["run_id"] = f"{pos.get('run_id', 'run')}::tail_drop"
                pair = _build_pair(pos=pos, neg=neg, strategy="tail_truncate")
                if pair is None:
                    skipped_empty += 1
                    continue
                pairs.append(pair)
                strategy_counter["tail_truncate"] += 1
                if max_pairs > 0 and len(pairs) >= max_pairs:
                    break

        if max_pairs > 0 and len(pairs) >= max_pairs:
            break

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "input_path": input_path,
        "output_path": output_path,
        "adapter": adapter,
        "seed": seed,
        "episodes_in": len(episodes),
        "task_groups": len(grouped),
        "pairs_out": len(pairs),
        "strategy_counts": dict(strategy_counter),
        "skipped_empty_pairs": skipped_empty,
    }
    if stats_path:
        path = Path(stats_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Export real episode logs into RM pairwise JSONL.")
    ap.add_argument("--input_path", type=str, required=True)
    ap.add_argument("--output_path", type=str, required=True)
    ap.add_argument("--adapter", type=str, default="generic_jsonl", choices=["generic_jsonl", "armap_style"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tail_drop_steps", type=int, default=2)
    ap.add_argument("--max_pairs", type=int, default=0, help="0 means unlimited.")
    ap.add_argument("--stats_path", type=str, default="")
    args = ap.parse_args()

    stats = export_pairs(
        input_path=args.input_path,
        output_path=args.output_path,
        adapter=args.adapter,
        seed=args.seed,
        tail_drop_steps=args.tail_drop_steps,
        max_pairs=args.max_pairs,
        stats_path=args.stats_path,
    )
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
