from __future__ import annotations

import re
from typing import List


OBS_PREFIX = ("observation:", "obs:")
ACT_PREFIX = ("action:",)
RES_PREFIX = ("final:", "result:", "success:", "score=")


def _strip_step_prefix(line: str) -> str:
    m = re.match(r"(?i)^step\s+\d+\s*:\s*(.*)$", line.strip())
    if m:
        return m.group(1).strip()
    return line.strip()


def _tag_for_line(body: str) -> str:
    lower = body.lower()
    if lower.startswith(OBS_PREFIX):
        return "OBS"
    if lower.startswith(ACT_PREFIX):
        return "ACT"
    if lower.startswith(RES_PREFIX):
        return "RES"
    return "OBS"


def format_instruction(instruction: str) -> str:
    text = str(instruction or "").strip()
    if not text:
        return ""
    return f"<GOAL>{text}</GOAL>"


def format_trajectory_stepwise(traj: str) -> str:
    lines = [line.strip() for line in str(traj or "").splitlines() if line.strip()]
    out: List[str] = []
    step_idx = 1
    for line in lines:
        lower = line.lower()
        if lower.startswith("task:"):
            out.append(f"Task: <OBS>{line[5:].strip()}</OBS>")
            continue
        if lower.startswith("final:"):
            body = line.split(":", 1)[1].strip() if ":" in line else line
            out.append(f"Final: <RES>{body}</RES>")
            continue

        body = _strip_step_prefix(line)
        tag = _tag_for_line(body)
        out.append(f"Step {step_idx}: <{tag}>{body}</{tag}>")
        step_idx += 1

    return "\n".join(out)


def get_last_k_step_char_start(traj: str, last_k_steps: int = 3) -> int:
    """Return char start for the last K step/final lines inside trajectory text.

    Works for both flat and stepwise trajectory strings.
    """

    text = str(traj or "")
    if not text:
        return 0

    step_markers = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip().lower()
        if stripped.startswith("step ") or stripped.startswith("final:"):
            step_markers.append(cursor + (len(line) - len(line.lstrip())))
        cursor += len(line)

    if not step_markers:
        return 0
    k = max(int(last_k_steps), 1)
    return step_markers[-k]
