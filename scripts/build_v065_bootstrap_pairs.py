#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_trace_text(row: Dict[str, Any]) -> str:
    task_id = str(row.get("task_id") or "").strip()
    policy = str(row.get("policy") or "").strip()
    lines: List[str] = [f"Task: {task_id}", f"Policy: {policy}"]
    steps = list(row.get("steps") or [])
    for idx, step in enumerate(steps, start=1):
        env_info = dict(step.get("env_info") or {})
        text = str(env_info.get("committed_text") or "").strip()
        if not text:
            text = str(step.get("selected_candidate_id") or "").strip()
        if not text:
            text = "<empty>"
        lines.append(f"Step {idx}: {text}")
        lines.append(f"PatchChars {idx}: {int(env_info.get('patch_chars', 0))}")
        lines.append(f"ExecErr {idx}: {str(env_info.get('exec_error_type') or 'none')}")
        lines.append(f"TestsPassed {idx}: {bool(env_info.get('tests_passed', False))}")
    if len(lines) <= 2:
        lines.append("Step 1: <no_steps_recorded>")
    return "\n".join(lines).strip()


def _instruction_for_task(task_id: str) -> Tuple[str, str]:
    raw = f"Solve SWE-bench Lite instance {task_id}."
    refined = f"Given instance `{task_id}`, propose a patch that makes tests pass."
    return raw, refined


def _trace_mean_rm_score(row: Dict[str, Any]) -> float | None:
    vals: List[float] = []
    for step in list(row.get("steps") or []):
        score = step.get("selected_rm_score")
        if score is None:
            continue
        vals.append(float(score))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _trace_uid(row: Dict[str, Any]) -> str:
    task_id = str(row.get("task_id") or "")
    policy = str(row.get("policy") or "")
    step_sig = ",".join([str((x.get("selected_candidate_id") or "")) for x in list(row.get("steps") or [])])
    h = hashlib.sha256(f"{task_id}|{policy}|{step_sig}".encode("utf-8")).hexdigest()[:16]
    return f"{task_id}:{policy}:{h}"


def _pair_row(
    task_id: str,
    pos: Dict[str, Any],
    neg: Dict[str, Any],
    pair_source: str,
    rm_gap: float | None,
    pair_policy: str,
) -> Dict[str, Any] | None:
    pos_traj = _build_trace_text(pos)
    neg_traj = _build_trace_text(neg)
    if not pos_traj.strip() or not neg_traj.strip():
        return None
    if pos_traj.strip() == neg_traj.strip():
        return None

    instruction_raw, instruction_refined = _instruction_for_task(task_id)
    pos_uid = _trace_uid(pos)
    neg_uid = _trace_uid(neg)

    return {
        "instruction_raw": instruction_raw,
        "instruction_refined": instruction_refined,
        "traj_pos": pos_traj,
        "traj_neg": neg_traj,
        "meta": {
            "task_id": task_id,
            "pair_source": pair_source,
            "pair_policy": pair_policy,
            "neg_type": "bootstrap_online",
            "env": "swe_real_online",
            "format_variant": "flat",
            "pos_uid": pos_uid,
            "neg_uid": neg_uid,
            "pos_policy": str(pos.get("policy") or ""),
            "neg_policy": str(neg.get("policy") or ""),
            "pos_success": bool(pos.get("success", False)),
            "neg_success": bool(neg.get("success", False)),
            "rm_gap": (float(rm_gap) if rm_gap is not None else None),
        },
    }


def _dedup_pairs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        key = (
            str(row.get("instruction_refined") or ""),
            str(row.get("traj_pos") or ""),
            str(row.get("traj_neg") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build v0.6.5 bootstrap RM train/valid pairs from online traces.")
    ap.add_argument("--traces_path", type=str, required=True)
    ap.add_argument("--output_train_jsonl", type=str, required=True)
    ap.add_argument("--output_valid_jsonl", type=str, required=True)
    ap.add_argument("--pair_policy", type=str, default="task_success_then_rm_gap")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--min_rm_gap", type=float, default=0.05)
    ap.add_argument("--max_pairs_per_task", type=int, default=8)
    ap.add_argument("--stats_json", type=str, default="")
    args = ap.parse_args()

    rows = _read_jsonl(Path(args.traces_path))
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_id = str(row.get("task_id") or "").strip()
        if not task_id:
            continue
        by_task[task_id].append(row)

    rng = random.Random(int(args.seed))
    out_pairs: List[Dict[str, Any]] = []
    count_success_failure = 0
    count_rm_gap = 0
    count_skipped = 0

    for task_id in sorted(by_task.keys()):
        traces = list(by_task[task_id])
        success_rows = [x for x in traces if bool(x.get("success", False))]
        failure_rows = [x for x in traces if not bool(x.get("success", False))]

        task_pairs: List[Dict[str, Any]] = []
        if success_rows and failure_rows:
            rng.shuffle(success_rows)
            rng.shuffle(failure_rows)
            for pos in success_rows:
                for neg in failure_rows:
                    pair = _pair_row(
                        task_id=task_id,
                        pos=pos,
                        neg=neg,
                        pair_source="same_task_success_over_failure",
                        rm_gap=None,
                        pair_policy=str(args.pair_policy),
                    )
                    if pair is not None:
                        task_pairs.append(pair)
                    if len(task_pairs) >= int(args.max_pairs_per_task):
                        break
                if len(task_pairs) >= int(args.max_pairs_per_task):
                    break
            count_success_failure += len(task_pairs)
        else:
            scored = []
            used_imputed = False
            for row in traces:
                score = _trace_mean_rm_score(row)
                if score is None:
                    score = 0.0
                    used_imputed = True
                scored.append((score, row))
            scored.sort(key=lambda x: x[0])
            if len(scored) >= 2:
                lo_score, lo_row = scored[0]
                hi_score, hi_row = scored[-1]
                gap = float(hi_score - lo_score)
                if gap >= float(args.min_rm_gap):
                    pair = _pair_row(
                        task_id=task_id,
                        pos=hi_row,
                        neg=lo_row,
                        pair_source=("same_task_rm_gap_pseudo_imputed" if used_imputed else "same_task_rm_gap_pseudo"),
                        rm_gap=gap,
                        pair_policy=str(args.pair_policy),
                    )
                    if pair is not None:
                        task_pairs.append(pair)
                        count_rm_gap += 1
            if not task_pairs:
                count_skipped += 1

        out_pairs.extend(task_pairs)

    out_pairs = _dedup_pairs(out_pairs)
    if len(out_pairs) < 2:
        raise RuntimeError(
            f"Insufficient bootstrap pairs ({len(out_pairs)}). "
            "Need at least 2 for train/valid split; check traces quality or relax min_rm_gap."
        )

    rng.shuffle(out_pairs)
    valid_n = int(round(len(out_pairs) * float(args.valid_ratio)))
    valid_n = max(1, min(valid_n, len(out_pairs) - 1))
    train_rows = out_pairs[:-valid_n]
    valid_rows = out_pairs[-valid_n:]

    train_path = Path(args.output_train_jsonl)
    valid_path = Path(args.output_valid_jsonl)
    _write_jsonl(train_path, train_rows)
    _write_jsonl(valid_path, valid_rows)

    stats = {
        "traces_path": str(args.traces_path),
        "n_trace_rows": len(rows),
        "n_tasks": len(by_task),
        "pair_policy": str(args.pair_policy),
        "total_pairs": len(out_pairs),
        "train_pairs": len(train_rows),
        "valid_pairs": len(valid_rows),
        "pair_source_counts": {
            "same_task_success_over_failure": int(count_success_failure),
            "same_task_rm_gap_pseudo": int(count_rm_gap),
        },
        "skipped_tasks_without_pair": int(count_skipped),
        "min_rm_gap": float(args.min_rm_gap),
    }

    if str(args.stats_json).strip():
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output_train_jsonl": str(train_path),
                "output_valid_jsonl": str(valid_path),
                "total_pairs": len(out_pairs),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
