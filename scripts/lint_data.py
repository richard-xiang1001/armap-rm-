#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Allow `python3 scripts/lint_data.py` from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rm.model import CharTokenizer


REQUIRED_FIELDS = ("instruction_refined", "traj_pos", "traj_neg")
PAIR_SEP = "\n\n---\n\n"


def load_jsonl(path: Path, sample_n: int) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if sample_n > 0 and len(rows) >= sample_n:
                break
    return rows


def char_token_len(text: str) -> int:
    # CharTokenizer maps each input character to one token (+BOS/+EOS).
    return len(text) + 2


def normalize_terms(csv_terms: str) -> List[str]:
    return [x.strip().lower() for x in csv_terms.split(",") if x.strip()]


def load_terms_file(path: Path) -> List[str]:
    terms: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip().lower()
            if not t or t.startswith("#"):
                continue
            terms.append(t)
    return terms


def dedup_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def truncation_visible(text: str, marker: str, max_len: int) -> bool:
    if marker not in text:
        return False
    # rm.model.CharTokenizer keeps at most (max_len - 2) characters.
    keep_chars = max(max_len - 2, 0)
    return marker in text[:keep_chars]


def find_step_marker_positions(text: str, markers: List[str]) -> List[int]:
    markers_l = [m.lower() for m in markers]
    positions: List[int] = []
    cursor = 0
    # splitlines(keepends=True) lets us keep accurate character offsets.
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip().lower()
        if any(stripped.startswith(m) for m in markers_l):
            positions.append(cursor + (len(line) - len(line.lstrip())))
        cursor += len(line)
    return positions


def tail_steps_truncated(text: str, markers: List[str], max_len: int, last_k_steps: int) -> Tuple[bool, bool]:
    positions = find_step_marker_positions(text, markers)
    if not positions:
        return False, False
    k = max(last_k_steps, 1)
    tail_positions = positions[-k:]
    keep_chars = max(max_len - 2, 0)
    truncated = any(p >= keep_chars for p in tail_positions)
    return True, truncated


def summarize(values: List[int]) -> Dict[str, float | int]:
    if not values:
        return {"n": 0, "min": 0, "max": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Lint preference-pair RM data (JSONL).")
    ap.add_argument("--path", type=str, required=True, help="Path to JSONL file.")
    ap.add_argument("--max_len", type=int, default=512, help="Target model max_len for truncation-risk analysis.")
    ap.add_argument(
        "--sample_n",
        type=int,
        default=0,
        help="Number of rows to analyze (0 means all rows).",
    )
    ap.add_argument(
        "--keywords",
        type=str,
        default="reward,success,done,score",
        help="Comma-separated leakage keywords (case-insensitive).",
    )
    ap.add_argument(
        "--leak_terms_file",
        type=str,
        default="",
        help="Optional newline-separated leak terms file (comments with # are ignored).",
    )
    ap.add_argument(
        "--step_markers",
        type=str,
        default="step,action:,observation:,obs:",
        help="Comma-separated step boundary markers for truncation-risk checks.",
    )
    ap.add_argument(
        "--last_k_steps",
        type=int,
        default=3,
        help="How many tail steps to protect in truncation-risk analysis.",
    )
    ap.add_argument("--fail_on_leak", action="store_true", help="Exit non-zero when leakage is detected.")
    ap.add_argument("--fail_on_missing", action="store_true", help="Exit non-zero when required fields are missing.")
    ap.add_argument(
        "--fail_on_truncation_risk",
        action="store_true",
        help="Exit non-zero when truncation_risk_last_k_steps exceeds threshold.",
    )
    ap.add_argument(
        "--max_truncation_risk_last_k_steps",
        type=float,
        default=0.05,
        help="Threshold used with --fail_on_truncation_risk.",
    )
    ap.add_argument(
        "--report_path",
        type=str,
        default="",
        help="Optional path to save full lint report JSON.",
    )
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    # Reuse tokenizer semantics from the current RM stack for truncation checks.
    _ = CharTokenizer()

    keywords = normalize_terms(args.keywords)
    if args.leak_terms_file:
        terms_path = Path(args.leak_terms_file)
        if not terms_path.exists():
            raise FileNotFoundError(f"Leak terms file not found: {terms_path}")
        keywords.extend(load_terms_file(terms_path))
    keywords = dedup_keep_order(keywords)
    step_markers = dedup_keep_order(normalize_terms(args.step_markers))
    rows = load_jsonl(path, sample_n=args.sample_n)

    missing_by_field = {k: 0 for k in REQUIRED_FIELDS}
    missing_rows = 0
    missing_cells = 0
    empty_traj_rows = 0
    leak_rows = 0
    leak_examples: List[Dict] = []
    truncation_risk_rows = 0
    truncation_risk_last_k_rows = 0
    rows_with_step_markers = 0

    instr_lens: List[int] = []
    pos_lens: List[int] = []
    neg_lens: List[int] = []

    for idx, obj in enumerate(rows):
        row_missing = False
        for key in REQUIRED_FIELDS:
            value = obj.get(key)
            if value is None:
                missing_by_field[key] += 1
                missing_cells += 1
                row_missing = True
        if row_missing:
            missing_rows += 1

        instruction = str(obj.get("instruction_refined", "") or "")
        traj_pos = str(obj.get("traj_pos", "") or "")
        traj_neg = str(obj.get("traj_neg", "") or "")

        instr_lens.append(char_token_len(instruction))
        pos_lens.append(char_token_len(traj_pos))
        neg_lens.append(char_token_len(traj_neg))

        if not traj_pos.strip() or not traj_neg.strip():
            empty_traj_rows += 1

        row_has_leak = False
        fields = {
            "instruction_refined": instruction,
            "traj_pos": traj_pos,
            "traj_neg": traj_neg,
        }
        lowered = {k: v.lower() for k, v in fields.items()}
        for field_name, text_l in lowered.items():
            for kw in keywords:
                if kw in text_l:
                    row_has_leak = True
                    if len(leak_examples) < 20:
                        leak_examples.append({"row": idx, "field": field_name, "keyword": kw})
        if row_has_leak:
            leak_rows += 1

        pos_text = instruction + PAIR_SEP + traj_pos
        neg_text = instruction + PAIR_SEP + traj_neg
        pos_has_final = "Final:" in pos_text
        neg_has_final = "Final:" in neg_text
        pos_truncated = pos_has_final and not truncation_visible(pos_text, "Final:", args.max_len)
        neg_truncated = neg_has_final and not truncation_visible(neg_text, "Final:", args.max_len)
        if pos_truncated or neg_truncated:
            truncation_risk_rows += 1

        pos_has_steps, pos_tail_truncated = tail_steps_truncated(
            pos_text, markers=step_markers, max_len=args.max_len, last_k_steps=args.last_k_steps
        )
        neg_has_steps, neg_tail_truncated = tail_steps_truncated(
            neg_text, markers=step_markers, max_len=args.max_len, last_k_steps=args.last_k_steps
        )
        if pos_has_steps or neg_has_steps:
            rows_with_step_markers += 1
        if pos_tail_truncated or neg_tail_truncated:
            truncation_risk_last_k_rows += 1

    n_rows = len(rows)
    report = {
        "path": str(path),
        "n_rows": n_rows,
        "required_fields": list(REQUIRED_FIELDS),
        "missing_required_fields": {
            "rows_with_missing": missing_rows,
            "total_missing_cells": missing_cells,
            "by_field": missing_by_field,
        },
        "empty_traj_rows": empty_traj_rows,
        "empty_traj_ratio": float(empty_traj_rows / max(n_rows, 1)),
        "instruction_len_stats": summarize(instr_lens),
        "traj_pos_len_stats": summarize(pos_lens),
        "traj_neg_len_stats": summarize(neg_lens),
        "leak_keywords": keywords,
        "leak_hit_count": leak_rows,
        "leak_hit_examples": leak_examples,
        "max_len": args.max_len,
        "truncation_risk_ratio_final_line": float(truncation_risk_rows / max(n_rows, 1)),
        "step_markers": step_markers,
        "last_k_steps": args.last_k_steps,
        "rows_with_step_markers": rows_with_step_markers,
        "truncation_risk_last_k_steps": float(truncation_risk_last_k_rows / max(n_rows, 1)),
        "max_truncation_risk_last_k_steps": args.max_truncation_risk_last_k_steps,
    }

    if args.report_path:
        out_path = Path(args.report_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False))

    should_fail = False
    if args.fail_on_leak and leak_rows > 0:
        should_fail = True
    if args.fail_on_missing and missing_rows > 0:
        should_fail = True
    if (
        args.fail_on_truncation_risk
        and report["truncation_risk_last_k_steps"] > args.max_truncation_risk_last_k_steps
    ):
        should_fail = True
    if should_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
