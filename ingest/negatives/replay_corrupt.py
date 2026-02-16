from __future__ import annotations

import random
import re
from typing import Dict, List

from ingest.envs.arithmetic_env import judge_arithmetic_episode, parse_final_from_steps, parse_target_from_instruction


_FINAL_PATTERN = re.compile(r"(?i)^\s*final\s*:\s*([-+]?\d+)\s*$")
_VERIFY_PATTERN = re.compile(r"(?i)(candidate\s+answer\s+)([-+]?\d+)")


def _perturb_number(rng: random.Random, val: int) -> int:
    deltas = [-3, -2, -1, 1, 2, 3, 5, -5]
    out = val + rng.choice(deltas)
    if out == val:
        out += 1
    return out


def _replace_final(steps: List[str], new_final: int) -> List[str]:
    out = list(steps)
    for i in range(len(out) - 1, -1, -1):
        line = str(out[i])
        if _FINAL_PATTERN.match(line):
            out[i] = f"Final: {new_final}"
            break
    return out


def _replace_verify(steps: List[str], new_final: int) -> List[str]:
    out = []
    replaced = False
    for line in steps:
        if not replaced and _VERIFY_PATTERN.search(str(line)):
            out.append(_VERIFY_PATTERN.sub(rf"\g<1>{new_final}", str(line), count=1))
            replaced = True
        else:
            out.append(str(line))
    return out


def generate(episode: Dict, rng: random.Random) -> Dict | None:
    """Replay-corruption negative: local value perturbation + env replay check."""

    instruction = str(episode.get("instruction_refined") or episode.get("instruction_raw") or "")
    steps = [str(x) for x in episode.get("steps", [])]
    if len(steps) < 2:
        return None

    target, _ = parse_target_from_instruction(instruction)
    final, _ = parse_final_from_steps(steps)
    if final is None:
        return None

    base = target if target is not None else final
    corrupted_final = _perturb_number(rng, base)
    if corrupted_final == final:
        corrupted_final = _perturb_number(rng, final)

    neg_steps = _replace_final(steps, corrupted_final)
    neg_steps = _replace_verify(neg_steps, corrupted_final)

    judge = judge_arithmetic_episode(instruction=instruction, steps=neg_steps, meta=episode.get("raw_meta"))
    replay_ok = judge.replay_ok and judge.success is False
    return {
        "steps": neg_steps,
        "neg_type": "replay_corrupt",
        "replay_ok": bool(replay_ok),
        "env_judge_neg": judge.success,
        "judge_reason": judge.reason,
        "judge_extras": judge.extras,
    }
