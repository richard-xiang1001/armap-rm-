from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol


@dataclass
class JudgeResult:
    success: Optional[bool]
    replay_ok: bool
    reason: str
    extras: Dict


class EnvJudge(Protocol):
    def judge(self, instruction: str, steps: List[str], meta: Dict | None = None) -> JudgeResult:
        ...
