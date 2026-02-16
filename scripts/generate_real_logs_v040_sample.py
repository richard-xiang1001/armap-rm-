#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def _episode(task_id: str, run_id: str, success: bool, rng: random.Random) -> Dict:
    query_id = rng.randint(1000, 9999)
    price = rng.randint(50, 600)
    steps: List[Dict] = [
        {"obs": f"Loaded catalog page {query_id}", "action": "open_search", "result": "ok"},
        {"obs": f"Filtered price<={price}", "action": "set_filter", "result": "ok"},
        {"obs": "Added item to cart", "action": "add_to_cart", "result": "ok"},
        {"obs": "Checkout submitted", "action": "checkout", "result": "ok"},
        {"text": f"success: {str(success).lower()}"},
    ]
    return {
        "episode_id": run_id,
        "task_id": task_id,
        "instruction": f"Find a product under {price} and place an order.",
        "steps": steps,
        "terminal": True,
        "success": success,
        "env": "real_logs_v1_sample",
        "meta": {
            "task_success": success,
            "session_id": f"sess_{query_id}",
            "score": 1.0 if success else 0.0,
            "user_email": "masked@example.com",
        },
    }


def generate(out_path: str, n_tasks: int, rollouts_per_task: int, seed: int) -> int:
    rng = random.Random(seed)
    rows: List[Dict] = []
    for task_idx in range(n_tasks):
        task_id = f"real_task_{task_idx:04d}"
        outcomes = [True, False]
        while len(outcomes) < rollouts_per_task:
            outcomes.append(rng.choice([True, False]))
        rng.shuffle(outcomes)

        for rollout_idx, success in enumerate(outcomes[:rollouts_per_task]):
            run_id = f"{task_id}_run_{rollout_idx:02d}"
            rows.append(_episode(task_id=task_id, run_id=run_id, success=success, rng=rng))

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a small real_logs_v1 JSONL sample for v0.4.0 acceptance.")
    ap.add_argument("--out_path", type=str, default="data/raw/episodes_real_v040.jsonl")
    ap.add_argument("--n_tasks", type=int, default=500)
    ap.add_argument("--rollouts_per_task", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    total = generate(
        out_path=args.out_path,
        n_tasks=args.n_tasks,
        rollouts_per_task=args.rollouts_per_task,
        seed=args.seed,
    )
    print(json.dumps({"out_path": args.out_path, "episodes": total}, ensure_ascii=False))


if __name__ == "__main__":
    main()
