from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class JudgeResult:
    success: Optional[bool]
    terminal: Optional[bool]
    score: Optional[float]
    replay_ok: bool
    reason: str
    extras: Dict[str, Any]


@dataclass
class CompareResult:
    consistent: bool
    unknown: bool
    reason: str


class EnvJudge(Protocol):
    name: str
    protocol_version: str

    def judge(self, instruction: str, steps: List[str], meta: Dict | None = None) -> JudgeResult:
        ...


def compare_judgments(pos: JudgeResult, neg: JudgeResult) -> CompareResult:
    if pos.success is None or neg.success is None:
        return CompareResult(
            consistent=False,
            unknown=True,
            reason=f"judge_unknown(pos={pos.success},neg={neg.success})",
        )
    consistent = bool(pos.success is True and neg.success is False)
    return CompareResult(
        consistent=consistent,
        unknown=False,
        reason="ok" if consistent else f"success_order_invalid(pos={pos.success},neg={neg.success})",
    )
