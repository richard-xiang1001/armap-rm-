from __future__ import annotations

from typing import Dict, List, Optional

from ingest.adapters.common import coerce_bool
from ingest.envs.protocol import EnvJudge, JudgeResult


def _extract_final_success(steps: List[str]) -> Optional[bool]:
    for line in reversed(steps):
        parsed = coerce_bool(str(line).split(":", 1)[-1].strip()) if ":" in str(line) else coerce_bool(line)
        lower = str(line).strip().lower()
        if lower.startswith("success:") or lower.startswith("done:") or lower.startswith("completed:"):
            return parsed
    return None


def _keyword_success(steps: List[str]) -> Optional[bool]:
    success_hits = ("completed", "success", "succeeded", "correct", "done")
    fail_hits = ("failed", "error", "incorrect", "denied", "timeout")
    for line in reversed(steps):
        lower = str(line).strip().lower()
        if any(k in lower for k in fail_hits):
            return False
        if any(k in lower for k in success_hits):
            return True
    return None


class RealLogsEnvJudge(EnvJudge):
    name = "real_logs"
    protocol_version = "v0.4.1"

    def judge(self, instruction: str, steps: List[str], meta: Dict | None = None) -> JudgeResult:
        raw_meta = meta if isinstance(meta, dict) else {}
        generated_negative = bool(raw_meta.get("generated_negative", False))

        step_success = _extract_final_success(steps)
        keyword_success = _keyword_success(steps)
        meta_success = coerce_bool(raw_meta.get("success"))
        if meta_success is None:
            meta_success = coerce_bool(raw_meta.get("task_success"))
        terminal = coerce_bool(raw_meta.get("terminal"))
        if terminal is None:
            terminal = coerce_bool(raw_meta.get("done"))
        if terminal is None:
            terminal = bool(steps)

        # For generated negatives we must avoid inheriting positive episode labels from meta.
        if generated_negative:
            success = step_success if step_success is not None else keyword_success
        else:
            success = step_success if step_success is not None else meta_success
            if success is None:
                success = keyword_success

        if success is None and generated_negative:
            return JudgeResult(
                success=False,
                terminal=terminal,
                score=0.0,
                replay_ok=bool(terminal),
                reason="inferred_failure_generated_negative",
                extras={
                    "step_success": step_success,
                    "keyword_success": keyword_success,
                    "meta_success": meta_success,
                    "terminal": terminal,
                    "generated_negative": generated_negative,
                },
            )

        if success is None:
            return JudgeResult(
                success=None,
                terminal=terminal,
                score=None,
                replay_ok=bool(terminal),
                reason="unknown_success",
                extras={
                    "step_success": step_success,
                    "keyword_success": keyword_success,
                    "meta_success": meta_success,
                    "terminal": terminal,
                    "generated_negative": generated_negative,
                },
            )

        return JudgeResult(
            success=bool(success),
            terminal=terminal,
            score=1.0 if bool(success) else 0.0,
            replay_ok=bool(terminal),
            reason="ok",
            extras={
                "step_success": step_success,
                "keyword_success": keyword_success,
                "meta_success": meta_success,
                "terminal": terminal,
                "generated_negative": generated_negative,
            },
        )
