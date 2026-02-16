#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare a synthetic preference-pair dataset for text-only RM training.

We generate arithmetic instructions with step-by-step trajectories:
- instruction: describes a target expression
- traj_pos: correct multi-step reasoning trace ending with the correct final answer
- traj_neg: identical-looking trace but with a subtly wrong intermediate step/final answer

This is NOT meant to be a benchmark; it is a minimal, runnable scaffold.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


OPS = [
    ("+", lambda a, b: a + b),
    ("-", lambda a, b: a - b),
    ("*", lambda a, b: a * b),
]


@dataclass
class Sample:
    instruction_raw: str
    instruction_refined: str
    traj_pos: str
    traj_neg: str
    meta: Dict


def _rand_int(rng: random.Random) -> int:
    # keep numbers small so traces are short and stable
    return rng.randint(1, 20)


def _make_expression(rng: random.Random) -> Tuple[str, int]:
    """Create a simple two-step expression: (a op1 b) op2 c"""
    a, b, c = _rand_int(rng), _rand_int(rng), _rand_int(rng)
    op1_sym, op1_fn = rng.choice(OPS)
    op2_sym, op2_fn = rng.choice(OPS)
    mid = op1_fn(a, b)
    out = op2_fn(mid, c)
    expr = f"({a} {op1_sym} {b}) {op2_sym} {c}"
    return expr, out


def _trajectory(expr: str, out: int) -> str:
    # A deliberately plain trace format: easy to parse, easy to extend.
    # In real environments you'd include obs/action; here we keep it text-only.
    return "\n".join(
        [
            f"Task: Compute {expr}",
            "Step 1: Evaluate the parentheses.",
            f"Step 2: Finish the remaining operation.",
            f"Final: {out}",
        ]
    )


def _make_negative(rng: random.Random, expr: str, out: int) -> Tuple[str, Dict]:
    """Create a hard-ish negative by perturbing a step or the final answer."""
    mode = rng.choice(["final_off_by", "sign_flip", "random_final"])
    if mode == "final_off_by":
        delta = rng.choice([-3, -2, -1, 1, 2, 3])
        neg_out = out + delta
        reason = {"neg_mode": mode, "delta": delta}
    elif mode == "sign_flip":
        neg_out = -out
        reason = {"neg_mode": mode}
    else:
        # ensure it's different
        neg_out = out
        while neg_out == out:
            neg_out = rng.randint(-100, 200)
        reason = {"neg_mode": mode}
    return _trajectory(expr, neg_out), reason


def generate_one(rng: random.Random, idx: int) -> Sample:
    expr, out = _make_expression(rng)
    instruction = f"Compute the value of: {expr}. Provide the final numeric answer."

    pos = _trajectory(expr, out)
    neg, neg_meta = _make_negative(rng, expr, out)

    # "instruction_refined" mirrors ARMAP's idea of a refined goal derived from a trajectory.
    # Here we keep it the same as instruction for simplicity.
    return Sample(
        instruction_raw=instruction,
        instruction_refined=instruction,
        traj_pos=pos,
        traj_neg=neg,
        meta={
            "id": idx,
            "env": "synthetic_arithmetic",
            "expr": expr,
            "label_out": out,
            **neg_meta,
        },
    )


def write_jsonl(path: str, samples: List[Sample]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(
                json.dumps(
                    {
                        "instruction_raw": s.instruction_raw,
                        "instruction_refined": s.instruction_refined,
                        "traj_pos": s.traj_pos,
                        "traj_neg": s.traj_neg,
                        "meta": s.meta,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data")
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_valid", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rng = random.Random(args.seed)

    train = [generate_one(rng, i) for i in range(args.n_train)]
    valid = [generate_one(rng, 10_000_000 + i) for i in range(args.n_valid)]

    train_path = os.path.join(args.out_dir, "train.jsonl")
    valid_path = os.path.join(args.out_dir, "valid.jsonl")

    write_jsonl(train_path, train)
    write_jsonl(valid_path, valid)

    print(f"Wrote train: {train_path} ({len(train)} samples)")
    print(f"Wrote valid: {valid_path} ({len(valid)} samples)")


if __name__ == "__main__":
    main()
