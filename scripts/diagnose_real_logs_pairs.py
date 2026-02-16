#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator


def _read_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose judge outcomes from pairs.all.jsonl")
    ap.add_argument("--path", type=str, required=True)
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--out_path", type=str, default="")
    args = ap.parse_args()

    path = Path(args.path)
    rows = list(_read_jsonl(path))

    unknown = 0
    reason_counter: Counter[str] = Counter()
    posneg_counter: Counter[str] = Counter()
    neg_type_counter: Counter[str] = Counter()

    for row in rows:
        meta = row.get("meta", {}) if isinstance(row.get("meta"), dict) else {}
        judge_unknown = bool(meta.get("judge_unknown", False))
        if judge_unknown:
            unknown += 1
            reason_counter[str(meta.get("judge_unknown_reason", "unknown"))] += 1
        pos = meta.get("env_judge_pos")
        neg = meta.get("env_judge_neg")
        posneg_counter[f"pos={pos}|neg={neg}"] += 1
        neg_type_counter[str(meta.get("neg_type", "unknown"))] += 1

    report = {
        "path": str(path),
        "n_rows": len(rows),
        "judge_unknown_count": unknown,
        "judge_unknown_rate": float(unknown / max(len(rows), 1)),
        "judge_unknown_reason_top": reason_counter.most_common(args.top_n),
        "pos_neg_matrix_top": posneg_counter.most_common(args.top_n),
        "neg_type_counts": dict(neg_type_counter),
    }

    if args.out_path:
        out_path = Path(args.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
