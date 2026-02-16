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
            (
                "instruction_raw",
                "instruction_refined",
                "instruction",
                "prompt",
                "goal",
                "task",
            ),
            "",
        )
        or ""
    ).strip()
    if not instruction_raw:
        return None

    instruction_refined = str(pick_first(row, ("instruction_refined",), instruction_raw) or instruction_raw).strip()
    steps_value = pick_first(row, ("steps", "trajectory", "traj", "events", "history", "messages"), [])
    steps = _normalize_steps(steps_value)
    if not steps:
        # Fallback for logs that only keep one flattened trajectory field.
        steps = _normalize_steps(pick_first(row, ("traj_pos", "trace", "log"), ""))

    success = coerce_bool(pick_first(row, ("success", "done", "task_success", "is_success", "outcome"), None))

    episode_id = pick_first(row, ("episode_id", "id"), index)
    task_id = pick_first(row, ("task_id", "task", "goal_id"), None)
    run_id = pick_first(row, ("run_id", "rollout_id", "trace_id"), None)
    seed = pick_first(row, ("seed", "random_seed"), None)
    env = pick_first(row, ("env", "env_name", "environment"), "unknown_env")

    return {
        "episode_id": episode_id,
        "task_id": str(task_id) if task_id is not None else None,
        "run_id": str(run_id) if run_id is not None else None,
        "seed": seed,
        "env": str(env),
        "instruction_raw": instruction_raw,
        "instruction_refined": instruction_refined or instruction_raw,
        "steps": steps,
        "success": success,
        "raw_meta": row.get("meta", {}),
    }


def iter_episodes(path: str) -> Iterator[Dict]:
    for index, row in enumerate(load_jsonl(path)):
        normalized = normalize_episode(row, index=index)
        if normalized is None:
            continue
        yield normalized

