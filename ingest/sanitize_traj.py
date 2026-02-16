#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_TERMS = [
    "reward",
    "success",
    "done",
    "score",
    "completed",
    "task_success",
    "is_success",
    "final_reward",
    "episode_return",
    "correct",
    "incorrect",
]

STRUCTURED_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    (
        re.compile(r"(?i)\b(reward|score|final_reward|episode_return)\s*[:=]\s*[-+]?\d+(?:\.\d+)?"),
        r"\1=[MASK]",
    ),
    (
        re.compile(
            r"(?i)\b(success|done|completed|task_success|is_success|correct|incorrect)\s*[:=]\s*(true|false|0|1|yes|no)"
        ),
        r"\1=[MASK]",
    ),
)


def _read_jsonl(path: str) -> Iterable[Dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_terms_file(path: str) -> List[str]:
    terms: List[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            terms.append(line)
    return terms


def _dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _normalize_step_markers(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    out: List[str] = []
    step_idx = 1
    for line in lines:
        lower = line.lower()
        if lower.startswith("task:"):
            out.append(line)
            continue
        if lower.startswith("final:"):
            out.append(line)
            continue
        match = re.match(r"(?i)^step\s+\d+\s*:\s*(.*)$", line)
        body = match.group(1).strip() if match else line
        out.append(f"Step {step_idx}: {body}")
        step_idx += 1
    return "\n".join(out)


def _sanitize_text(text: str, terms: List[str]) -> Tuple[str, int, int]:
    sanitized = text
    pattern_masks = 0
    term_masks = 0
    for pattern, replacement in STRUCTURED_PATTERNS:
        sanitized, n = pattern.subn(replacement, sanitized)
        pattern_masks += n
    for term in terms:
        escaped = re.escape(term)
        sanitized, n = re.subn(escaped, "[MASK]", sanitized, flags=re.IGNORECASE)
        term_masks += n
    return sanitized, pattern_masks, term_masks


def sanitize_file(
    input_path: str,
    output_path: str,
    leak_terms: List[str],
    drop_empty_rows: bool = True,
    audit_path: str = "",
) -> Dict:
    rows_in = 0
    rows_out = 0
    rows_dropped = 0
    rows_masked = 0
    total_pattern_masks = 0
    total_term_masks = 0
    field_mask_counts = {"instruction_refined": 0, "traj_pos": 0, "traj_neg": 0}

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out_f:
        for row in _read_jsonl(input_path):
            rows_in += 1
            row = dict(row)
            row_masks = 0

            for field in ("instruction_refined", "traj_pos", "traj_neg"):
                value = str(row.get(field, "") or "")
                sanitized, n_pattern, n_term = _sanitize_text(value, leak_terms)
                if field.startswith("traj_"):
                    sanitized = _normalize_step_markers(sanitized)
                row[field] = sanitized
                field_mask_counts[field] += n_pattern + n_term
                total_pattern_masks += n_pattern
                total_term_masks += n_term
                row_masks += n_pattern + n_term

            if row_masks > 0:
                rows_masked += 1

            if drop_empty_rows and (not str(row.get("traj_pos", "")).strip() or not str(row.get("traj_neg", "")).strip()):
                rows_dropped += 1
                continue

            meta = row.get("meta")
            if not isinstance(meta, dict):
                meta = {}
            meta["sanitize"] = {
                "masked_fragments": row_masks,
                "drop_empty_rows_policy": bool(drop_empty_rows),
            }
            row["meta"] = meta

            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows_out += 1

    audit = {
        "input_path": input_path,
        "output_path": output_path,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "rows_dropped": rows_dropped,
        "rows_masked": rows_masked,
        "total_pattern_masks": total_pattern_masks,
        "total_term_masks": total_term_masks,
        "field_mask_counts": field_mask_counts,
        "sample_keep_ratio": float(rows_out / max(rows_in, 1)),
        "leak_terms": leak_terms,
    }
    if audit_path:
        path = Path(audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    ap = argparse.ArgumentParser(description="Sanitize trajectory pairs by masking leakage and normalizing steps.")
    ap.add_argument("--input_path", type=str, required=True)
    ap.add_argument("--output_path", type=str, required=True)
    ap.add_argument(
        "--leak_terms",
        type=str,
        default=",".join(DEFAULT_TERMS),
        help="Comma-separated leak terms.",
    )
    ap.add_argument("--leak_terms_file", type=str, default="")
    ap.add_argument("--audit_path", type=str, default="")
    ap.add_argument("--drop_empty_rows", dest="drop_empty_rows", action="store_true")
    ap.add_argument("--keep_empty_rows", dest="drop_empty_rows", action="store_false")
    ap.set_defaults(drop_empty_rows=True)
    args = ap.parse_args()

    terms = [x.strip() for x in args.leak_terms.split(",") if x.strip()]
    if args.leak_terms_file:
        terms.extend(_load_terms_file(args.leak_terms_file))
    terms = _dedup_keep_order(terms)

    audit = sanitize_file(
        input_path=args.input_path,
        output_path=args.output_path,
        leak_terms=terms,
        drop_empty_rows=args.drop_empty_rows,
        audit_path=args.audit_path,
    )
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()

