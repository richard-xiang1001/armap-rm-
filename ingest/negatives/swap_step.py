from __future__ import annotations

import random
import re
from typing import Dict, List

from ingest.envs.protocol import EnvJudge


_FINAL_PATTERN = re.compile(r"(?i)^\s*final\s*:")
_ACTION_HINT_PATTERN = re.compile(r"(?i)\b(action|click|type|select|submit|tool|api)\b")


def generate(episode: Dict, rng: random.Random, env_judge: EnvJudge) -> Dict | None:
    """Swap adjacent high-impact steps to create plausible but failing structure."""

    instruction = str(episode.get("instruction_refined") or episode.get("instruction_raw") or "")
    steps = [str(x) for x in episode.get("steps", [])]
    if len(steps) < 3:
        return None

    final_idx = -1
    for i in range(len(steps) - 1, -1, -1):
        if _FINAL_PATTERN.match(steps[i].strip()):
            final_idx = i
            break
    # real logs may not have explicit Final line; then allow swapping across full tail.
    bound = final_idx if final_idx > 1 else len(steps)
    if bound <= 2:
        return None

    candidates = [i for i in range(0, bound - 1) if steps[i].strip() and steps[i + 1].strip()]
    action_candidates = [i for i in candidates if _ACTION_HINT_PATTERN.search(steps[i]) or _ACTION_HINT_PATTERN.search(steps[i + 1])]
    if action_candidates:
        candidates = action_candidates
    if not candidates:
        return None

    swap_idx = rng.choice(candidates)
    neg_steps = list(steps)
    neg_steps[swap_idx], neg_steps[swap_idx + 1] = neg_steps[swap_idx + 1], neg_steps[swap_idx]

    raw_meta = dict(episode.get("raw_meta") or {})
    raw_meta["generated_negative"] = True
    raw_meta["success"] = False
    judge = env_judge.judge(instruction=instruction, steps=neg_steps, meta=raw_meta)
    replay_ok = bool(judge.replay_ok) and judge.success is False
    return {
        "steps": neg_steps,
        "neg_type": "swap_step",
        "replay_ok": bool(replay_ok),
        "env_judge_neg": judge.success,
        "env_judge_terminal": judge.terminal,
        "env_judge_score": judge.score,
        "judge_reason": judge.reason,
        "judge_extras": judge.extras,
        "corruption_target": "adjacent_swap",
        "corruption_step_idx": swap_idx,
    }
