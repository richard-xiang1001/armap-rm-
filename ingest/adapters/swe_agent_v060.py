from __future__ import annotations

from typing import Any, Dict, Iterator, List

from ingest.adapters.common import coerce_bool, load_jsonl, pick_first, stringify_step


_SWE_META_KEYS = (
    "resolved",
    "tests_passed",
    "patch_applied",
    "error",
    "timeout",
    "timed_out",
    "status",
    "exit_code",
    "repo",
    "instance_id",
    "problem_id",
)


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _stringify_swe_step(step: Any) -> str:
    if isinstance(step, str):
        return step.strip()
    if isinstance(step, dict):
        parts: List[str] = []
        for key in (
            "response",
            "action",
            "observation",
            "thought",
            "tool",
            "command",
            "result",
            "content",
            "text",
        ):
            value = step.get(key)
            if value is not None and str(value).strip():
                parts.append(f"{key}: {str(value).strip()}")
        if parts:
            return " | ".join(parts)
        return stringify_step(step)
    if isinstance(step, list):
        joined = " | ".join(str(x).strip() for x in step if str(x).strip())
        return joined.strip()
    return str(step).strip()


def _normalize_steps(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            text = _stringify_swe_step(item)
            if text:
                out.append(text)
        return out
    if isinstance(value, dict):
        text = _stringify_swe_step(value)
        return [text] if text else []
    text = str(value).strip()
    return [text] if text else []


def _extract_success(row: Dict, raw_meta: Dict) -> bool | None:
    success = coerce_bool(
        pick_first(
            row,
            ("success", "task_success", "is_success", "resolved", "tests_passed", "patch_applied"),
            raw_meta.get("success"),
        )
    )
    if success is not None:
        return bool(success)

    for key in ("resolved", "tests_passed", "patch_applied"):
        value = coerce_bool(raw_meta.get(key))
        if value is not None:
            return bool(value)

    timeout = coerce_bool(pick_first(row, ("timeout", "timed_out"), raw_meta.get("timeout")))
    if timeout is True:
        return False

    for key in ("error", "exception", "last_error"):
        if str(raw_meta.get(key, "") or "").strip():
            return False
    return None


def normalize_episode(row: Dict, index: int) -> Dict | None:
    raw_meta = row.get("meta", {}) if isinstance(row.get("meta"), dict) else {}
    raw_meta = dict(raw_meta)

    instruction_raw = _first_non_empty(
        pick_first(
            row,
            (
                "instruction",
                "instruction_raw",
                "instruction_refined",
                "prompt",
                "goal",
                "task",
                "problem_statement",
                "query",
            ),
            "",
        ),
        raw_meta.get("instruction"),
        raw_meta.get("instruction_raw"),
        raw_meta.get("instruction_refined"),
        raw_meta.get("problem_statement"),
        raw_meta.get("prompt"),
        raw_meta.get("query"),
    )
    if not instruction_raw:
        return None

    steps = _normalize_steps(pick_first(row, ("steps", "trajectory", "history", "events", "messages"), []))
    if not steps:
        return None

    for key in _SWE_META_KEYS:
        value = pick_first(row, (key,), raw_meta.get(key))
        if value is not None:
            raw_meta[key] = value

    terminal = coerce_bool(pick_first(row, ("terminal", "done"), raw_meta.get("terminal")))
    if terminal is not None:
        raw_meta["terminal"] = bool(terminal)

    success = _extract_success(row=row, raw_meta=raw_meta)
    if success is not None:
        raw_meta["success"] = bool(success)

    episode_id = pick_first(row, ("episode_id", "id", "trace_id", "run_id"), index)
    task_id = pick_first(
        row,
        ("task_id", "instance_id", "problem_id", "issue_id"),
        raw_meta.get("task_id") or raw_meta.get("instance_id") or raw_meta.get("problem_id"),
    )
    run_id = pick_first(row, ("run_id", "trace_id", "rollout_id", "session_id"), raw_meta.get("run_id"))
    seed = pick_first(row, ("seed", "random_seed"), raw_meta.get("seed"))
    env = pick_first(row, ("env", "env_name", "environment"), "swe_agent_v060")

    instruction_refined = _first_non_empty(
        pick_first(row, ("instruction_refined",), ""),
        raw_meta.get("instruction_refined"),
        instruction_raw,
    )

    return {
        "episode_id": episode_id,
        "task_id": str(task_id) if task_id is not None else None,
        "run_id": str(run_id) if run_id is not None else None,
        "seed": seed,
        "env": str(env),
        "instruction_raw": instruction_raw,
        "instruction_refined": instruction_refined,
        "steps": steps,
        "success": success,
        "raw_meta": raw_meta,
    }


def iter_episodes(path: str) -> Iterator[Dict]:
    for index, row in enumerate(load_jsonl(path)):
        normalized = normalize_episode(row=row, index=index)
        if normalized is None:
            continue
        yield normalized
