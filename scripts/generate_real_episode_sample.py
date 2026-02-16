#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


OPS = ["+", "-", "*"]


def _compute(a: int, b: int, op: str) -> int:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    return a * b


def _episode(task_id: str, run_id: str, instruction: str, correct_answer: int, success: bool, rng: random.Random) -> Dict:
    if success:
        final_answer = correct_answer
    else:
        wrong = correct_answer
        while wrong == correct_answer:
            wrong = correct_answer + rng.choice([-3, -2, -1, 1, 2, 3, 11])
        final_answer = wrong

    steps: List[str] = [
        f"action: parse expression for {instruction}",
        "observation: computed intermediate values",
        f"action: verify candidate answer {final_answer}",
        f"success: {str(success).lower()}",
        f"score={1.0 if success else 0.0}",
        f"Final: {final_answer}",
    ]
    return {
        "episode_id": run_id,
        "task_id": task_id,
        "run_id": run_id,
        "instruction": instruction,
        "steps": steps,
        "success": success,
        "env": "sample_arithmetic_real_like",
        "meta": {
            "task_success": success,
            "episode_return": 1.0 if success else 0.0,
            "note": "contains deliberate leakage fields for sanitize testing",
        },
    }


def generate(out_path: str, n_tasks: int, rollouts_per_task: int, seed: int) -> int:
    rng = random.Random(seed)
    rows: List[Dict] = []
    for task_idx in range(n_tasks):
        a = rng.randint(1, 20)
        b = rng.randint(1, 20)
        c = rng.randint(1, 20)
        op1 = rng.choice(OPS)
        op2 = rng.choice(OPS)
        mid = _compute(a, b, op1)
        target = _compute(mid, c, op2)
        expr = f"({a} {op1} {b}) {op2} {c}"
        instruction = f"Solve task {task_idx}: compute {expr}."
        task_id = f"task_{task_idx:04d}"

        # Ensure each task has at least one success and one failure for hard negatives.
        outcomes = [True, False]
        while len(outcomes) < rollouts_per_task:
            outcomes.append(rng.choice([True, False]))
        rng.shuffle(outcomes)

        for rollout_idx, success in enumerate(outcomes[:rollouts_per_task]):
            run_id = f"{task_id}_run_{rollout_idx:02d}"
            rows.append(_episode(task_id, run_id, instruction, target, success, rng))

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a minimal real-like episode JSONL sample.")
    ap.add_argument("--out_path", type=str, default="data/raw/example_episodes.jsonl")
    ap.add_argument("--n_tasks", type=int, default=50)
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

