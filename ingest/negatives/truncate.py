from __future__ import annotations

import random
from typing import Dict, List

from ingest.envs.arithmetic_env import judge_arithmetic_episode


def _truncate_tail(steps: List[str], drop_steps: int) -> List[str] | None:
    if len(steps) <= 1:
        return None
    drop = max(1, min(drop_steps, len(steps) - 1))
    out = steps[:-drop]
    if not out:
        return None
    return out


def generate(episode: Dict, rng: random.Random, tail_drop_steps: int = 2) -> Dict | None:
    """Tail truncation fallback negative with env judge metadata."""

    instruction = str(episode.get("instruction_refined") or episode.get("instruction_raw") or "")
    steps = [str(x) for x in episode.get("steps", [])]
    drop = tail_drop_steps
    if tail_drop_steps <= 0:
        drop = rng.choice([1, 2])

    neg_steps = _truncate_tail(steps, drop_steps=drop)
    if not neg_steps:
        return None

    judge = judge_arithmetic_episode(instruction=instruction, steps=neg_steps, meta=episode.get("raw_meta"))
    replay_ok = judge.replay_ok and judge.success is False
    return {
        "steps": neg_steps,
        "neg_type": "truncate_last_k",
        "replay_ok": bool(replay_ok),
        "env_judge_neg": judge.success,
        "judge_reason": judge.reason,
        "judge_extras": judge.extras,
    }
