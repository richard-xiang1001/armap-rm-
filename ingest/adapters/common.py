from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, Sequence


def load_jsonl(path: str) -> Iterator[Dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def pick_first(obj: Dict, keys: Sequence[str], default=None):
    for key in keys:
        if key in obj and obj[key] is not None:
            return obj[key]
    return default


def coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "y", "1", "success", "passed"}:
            return True
        if v in {"false", "no", "n", "0", "fail", "failed"}:
            return False
    return None


def stringify_step(step) -> str:
    if isinstance(step, str):
        return step.strip()
    if isinstance(step, dict):
        if "text" in step and step["text"] is not None:
            return str(step["text"]).strip()
        parts = []
        for key in ("thought", "action", "observation", "obs", "tool", "result", "content"):
            value = step.get(key)
            if value is not None and str(value).strip():
                parts.append(f"{key}: {value}")
        if parts:
            return " | ".join(parts)
        return json.dumps(step, ensure_ascii=False, sort_keys=True)
    return str(step).strip()

