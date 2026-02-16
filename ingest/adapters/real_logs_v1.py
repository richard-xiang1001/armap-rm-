from __future__ import annotations

from typing import Dict, Iterator, List

from ingest.adapters.common import coerce_bool, load_jsonl, pick_first, stringify_step


def _normalize_steps(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            text = stringify_step(item)
            if text:
                out.append(text)
        return out
    if isinstance(value, dict):
        text = stringify_step(value)
        return [text] if text else []
    return [str(value).strip()] if str(value).strip() else []


def normalize_episode(row: Dict, index: int) -> Dict | None:
    instruction_raw = str(
        pick_first(
            row,
            ("instruction", "instruction_raw", "instruction_refined", "prompt", "goal", "task"),
            "",
        )
        or ""
    ).strip()
    if not instruction_raw:
        return None

    steps = _normalize_steps(pick_first(row, ("steps", "trajectory", "events", "history"), []))
    if not steps:
        return None

    raw_meta = row.get("meta", {}) if isinstance(row.get("meta"), dict) else {}
    raw_meta = dict(raw_meta)

    terminal = coerce_bool(pick_first(row, ("terminal", "done"), raw_meta.get("terminal")))
    success = coerce_bool(pick_first(row, ("success", "task_success", "is_success"), raw_meta.get("success")))
    if terminal is not None:
        raw_meta["terminal"] = bool(terminal)
    if success is not None:
        raw_meta["success"] = bool(success)

    episode_id = pick_first(row, ("episode_id", "id", "trace_id"), index)
    task_id = pick_first(row, ("task_id", "task", "goal_id"), raw_meta.get("task_id"))
    run_id = pick_first(row, ("run_id", "trace_id", "rollout_id"), raw_meta.get("run_id"))
    seed = pick_first(row, ("seed", "random_seed"), raw_meta.get("seed"))
    env = pick_first(row, ("env", "env_name", "environment"), "real_logs_v1")

    return {
        "episode_id": episode_id,
        "task_id": str(task_id) if task_id is not None else None,
        "run_id": str(run_id) if run_id is not None else None,
        "seed": seed,
        "env": str(env),
        "instruction_raw": instruction_raw,
        "instruction_refined": str(pick_first(row, ("instruction_refined",), instruction_raw) or instruction_raw).strip(),
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
