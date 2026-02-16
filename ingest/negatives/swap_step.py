from __future__ import annotations

import random
import re
from typing import Dict, List

from ingest.envs.arithmetic_env import judge_arithmetic_episode


_FINAL_PATTERN = re.compile(r"(?i)^\s*final\s*:")


def generate(episode: Dict, rng: random.Random) -> Dict | None:
    """Swap one step with Final line to create executable but failing structure."""

    instruction = str(episode.get("instruction_refined") or episode.get("instruction_raw") or "")
    steps = [str(x) for x in episode.get("steps", [])]
    if len(steps) < 3:
        return None

    final_idx = -1
    for i in range(len(steps) - 1, -1, -1):
        if _FINAL_PATTERN.match(steps[i].strip()):
            final_idx = i
            break
    if final_idx <= 0:
        return None

    candidates = [i for i in range(final_idx) if steps[i].strip()]
    if not candidates:
        return None

    swap_idx = rng.choice(candidates)
    neg_steps = list(steps)
    neg_steps[swap_idx], neg_steps[final_idx] = neg_steps[final_idx], neg_steps[swap_idx]

    judge = judge_arithmetic_episode(instruction=instruction, steps=neg_steps, meta=episode.get("raw_meta"))
    replay_ok = judge.replay_ok and judge.success is False
    return {
        "steps": neg_steps,
        "neg_type": "swap_step",
        "replay_ok": bool(replay_ok),
        "env_judge_neg": judge.success,
        "judge_reason": judge.reason,
        "judge_extras": judge.extras,
    }
