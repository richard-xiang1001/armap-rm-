from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def _dedup_keep_order(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        token = str(value or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def load_forbidden_markers(path: str | Path) -> Dict[str, List[str] | str]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Forbidden markers file not found: {cfg_path}")
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Forbidden markers payload must be an object: {cfg_path}")

    version = str(payload.get("version", "") or "").strip() or "unknown"
    hard = payload.get("hard_leak_markers", [])
    soft = payload.get("soft_marker_audit", [])
    if not isinstance(hard, list) or not isinstance(soft, list):
        raise ValueError("forbidden marker lists must be arrays")

    hard_markers = _dedup_keep_order([str(x) for x in hard])
    soft_markers = _dedup_keep_order([str(x) for x in soft])
    union_markers = _dedup_keep_order(hard_markers + soft_markers)
    return {
        "version": version,
        "hard_leak_markers": hard_markers,
        "soft_marker_audit": soft_markers,
        "all_markers": union_markers,
    }
