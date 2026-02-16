from __future__ import annotations

import random
import re
from typing import Dict, List

from ingest.envs.arithmetic_env import parse_final_from_steps, parse_target_from_instruction
from ingest.envs.protocol import EnvJudge


_FINAL_PATTERN = re.compile(r"(?i)^\s*final\s*:\s*([-+]?\d+)\s*$")
_VERIFY_PATTERN = re.compile(r"(?i)(candidate\s+answer\s+)([-+]?\d+)")
_KV_NUMBER_PATTERN = re.compile(r"(?i)\b([a-z_][a-z0-9_]{1,32})\s*[:=]\s*(-?\d+)\b")
_ACTION_HINT_PATTERN = re.compile(r"(?i)\b(action|click|type|select|submit|tool|api)\b")
_RESULT_OK_PATTERN = re.compile(r"(?i)\b(result|status)\s*[:=]\s*(ok|success|passed)\b")


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


def _select_high_impact_indices(steps: List[str]) -> List[int]:
    n = len(steps)
    if n <= 0:
        return []
    start = max(0, int(n * 0.8))
    candidates = [i for i in range(start, n)]
    if not candidates:
        candidates = list(range(n))
    keyed = [i for i in candidates if _ACTION_HINT_PATTERN.search(str(steps[i]))]
    return keyed or candidates


def _perturb_action_param(lines: List[str], idx: int, rng: random.Random) -> List[str] | None:
    line = str(lines[idx])
    m = _KV_NUMBER_PATTERN.search(line)
    if not m:
        return None
    key = m.group(1)
    value = int(m.group(2))
    changed = _perturb_number(rng, value)
    out = list(lines)
    out[idx] = _KV_NUMBER_PATTERN.sub(f"{key}={changed}", line, count=1)
    return out


def generate(episode: Dict, rng: random.Random, env_judge: EnvJudge) -> Dict | None:
    """Replay-corruption negative: local value perturbation + env replay check."""

    instruction = str(episode.get("instruction_refined") or episode.get("instruction_raw") or "")
    steps = [str(x) for x in episode.get("steps", [])]
    if len(steps) < 2:
        return None

    target, _ = parse_target_from_instruction(instruction)
    final, _ = parse_final_from_steps(steps)

    # First try key-parameter corruption on high-impact tail steps.
    neg_steps = None
    param_corrupt_idx = None
    for idx in _select_high_impact_indices(steps):
        mutated = _perturb_action_param(steps, idx=idx, rng=rng)
        if mutated is not None:
            neg_steps = mutated
            param_corrupt_idx = idx
            break

    # Fallback to final-answer corruption.
    if neg_steps is None:
        if final is not None:
            base = target if target is not None else final
            corrupted_final = _perturb_number(rng, base)
            if corrupted_final == final:
                corrupted_final = _perturb_number(rng, final)
            neg_steps = _replace_final(steps, corrupted_final)
            neg_steps = _replace_verify(neg_steps, corrupted_final)
        else:
            # Real-logs fallback: corrupt the last action/result marker.
            neg_steps = list(steps)
            changed_idx = None
            for idx in reversed(_select_high_impact_indices(steps)):
                line = str(neg_steps[idx])
                if _RESULT_OK_PATTERN.search(line):
                    neg_steps[idx] = _RESULT_OK_PATTERN.sub(r"\1=failed", line, count=1)
                    changed_idx = idx
                    break
            if changed_idx is None:
                # Guaranteed mutation fallback.
                neg_steps[-1] = str(neg_steps[-1]) + " | status=failed"
                changed_idx = len(neg_steps) - 1
            param_corrupt_idx = changed_idx

    raw_meta = dict(episode.get("raw_meta") or {})
    raw_meta["generated_negative"] = True
    raw_meta["success"] = False
    judge = env_judge.judge(instruction=instruction, steps=neg_steps, meta=raw_meta)
    replay_ok = bool(judge.replay_ok) and judge.success is False
    return {
        "steps": neg_steps,
        "neg_type": "replay_corrupt",
        "replay_ok": bool(replay_ok),
        "env_judge_neg": judge.success,
        "env_judge_terminal": judge.terminal,
        "env_judge_score": judge.score,
        "judge_reason": judge.reason,
        "judge_extras": judge.extras,
        "corruption_target": "action_param" if param_corrupt_idx is not None else "final_answer",
        "corruption_step_idx": param_corrupt_idx,
    }
