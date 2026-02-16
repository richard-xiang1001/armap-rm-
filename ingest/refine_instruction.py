from __future__ import annotations

import re
from typing import Dict, List, Tuple


EXPR_PATTERNS = (
    re.compile(r"(?i)\bcompute\s+([^\.\?]+)"),
    re.compile(r"(?i)\bvalue\s+of\s*:\s*([^\.\?]+)"),
)


def _extract_expr(text: str) -> str:
    for pattern in EXPR_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(1).strip()
    return ""


def refine_instruction_rule(instruction_raw: str, steps: List[str] | None = None) -> Tuple[str, Dict]:
    """Rule-based instruction normalization for stable, reproducible v0.3.0 ingest.

    We keep this deterministic and dependency-free:
    - collapse whitespace
    - detect arithmetic expression and normalize to one canonical sentence
    - otherwise keep original instruction text
    """

    raw = " ".join(str(instruction_raw or "").strip().split())
    if not raw:
        return "", {"mode": "rule", "changed": False, "reason": "empty"}

    expr = _extract_expr(raw)
    if not expr and steps:
        for line in steps:
            candidate = _extract_expr(str(line))
            if candidate:
                expr = candidate
                break

    if expr:
        refined = f"Compute the value of: {expr}. Provide the final numeric answer."
        changed = refined != raw
        return refined, {"mode": "rule", "changed": changed, "reason": "expr_normalized"}

    # Generic cleanup only.
    refined = raw
    return refined, {"mode": "rule", "changed": refined != raw, "reason": "whitespace_normalized"}


def refine_instruction(instruction_raw: str, steps: List[str] | None = None, mode: str = "rule") -> Tuple[str, Dict]:
    if mode != "rule":
        raise ValueError(f"Unsupported refine mode: {mode}")
    return refine_instruction_rule(instruction_raw=instruction_raw, steps=steps)
