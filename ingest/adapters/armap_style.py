from __future__ import annotations

from typing import Dict, Iterator, List

from ingest.adapters.common import coerce_bool, load_jsonl, pick_first, stringify_step


def _armap_steps(row: Dict) -> List[str]:
    candidates = (
        row.get("steps"),
        row.get("trajectory"),
        row.get("traj"),
        row.get("rollout"),
        row.get("events"),
    )
    for value in candidates:
        if value is None:
            continue
        if isinstance(value, str):
            return [line.strip() for line in value.splitlines() if line.strip()]
        if isinstance(value, list):
            out: List[str] = []
            for item in value:
                text = stringify_step(item)
                if text:
                    out.append(text)
            return out
    return []


def normalize_episode(row: Dict, index: int) -> Dict | None:
    instruction = str(
        pick_first(
            row,
            (
                "instruction_refined",
                "instruction_raw",
                "instruction",
                "goal",
                "task",
            ),
            "",
        )
        or ""
    ).strip()
    if not instruction:
        return None

    meta = row.get("meta", {}) if isinstance(row.get("meta"), dict) else {}
    success = coerce_bool(
        pick_first(
            row,
            ("success", "task_success", "is_success", "done", "outcome"),
            pick_first(meta, ("success", "task_success", "is_success", "done", "outcome"), None),
        )
    )

    steps = _armap_steps(row)
    episode_id = pick_first(row, ("episode_id", "id"), index)
    task_id = pick_first(row, ("task_id", "task", "goal_id"), pick_first(meta, ("task_id",), None))
    run_id = pick_first(row, ("run_id", "trace_id", "rollout_id"), pick_first(meta, ("run_id",), None))
    env = pick_first(row, ("env", "env_name"), pick_first(meta, ("env", "env_name"), "armap_env"))
    seed = pick_first(row, ("seed",), pick_first(meta, ("seed",), None))

    return {
        "episode_id": episode_id,
        "task_id": str(task_id) if task_id is not None else None,
        "run_id": str(run_id) if run_id is not None else None,
        "seed": seed,
        "env": str(env),
        "instruction_raw": instruction,
        "instruction_refined": instruction,
        "steps": steps,
        "success": success,
        "raw_meta": meta,
    }


def iter_episodes(path: str) -> Iterator[Dict]:
    for index, row in enumerate(load_jsonl(path)):
        normalized = normalize_episode(row, index=index)
        if normalized is None:
            continue
        yield normalized

