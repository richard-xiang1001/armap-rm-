#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Allow `python3 scripts/run_v061_swe_planning_eval.py` from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingest.adapters.swe_agent_v060 import iter_episodes
from ingest.rules import load_forbidden_markers

PAIR_SEP = "\n\n---\n\n"
_TAIL_CRITICAL_PATTERN = re.compile(r"(?i)\b(observation|obs:|test|pytest|patch|diff --git|error|traceback)\b")
_PATCH_HINT_PATTERN = re.compile(r"(?i)(diff --git|@@|^\+\+\+|^---|\bpatch\b|\bapply_patch\b|\bgit diff\b)")
_TOOL_CALL_PATTERN = re.compile(r"(?i)\b(action|tool|command|cmd|bash|python|pytest|pip|git|apply_patch|run)\b")


def _run(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, text=True)


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


def _clip_step_text(text: str, max_chars_per_step: int) -> str:
    body = " ".join(str(text or "").split())
    if max_chars_per_step <= 0 or len(body) <= max_chars_per_step:
        return body
    keep = max(8, max_chars_per_step - 3)
    return body[:keep].rstrip() + "..."


def _format_step(text: str, idx: int, max_chars_per_step: int, allow_clip: bool) -> str:
    lower = text.lower()
    clipped = _clip_step_text(text, max_chars_per_step=max_chars_per_step) if allow_clip else " ".join(text.split())
    if lower.startswith("task:") or lower.startswith("final:"):
        return clipped
    match = re.match(r"(?i)^step\s+\d+\s*:\s*(.*)$", text)
    body = match.group(1).strip() if match else clipped
    if allow_clip:
        body = _clip_step_text(body, max_chars_per_step=max_chars_per_step)
    else:
        body = " ".join(body.split())
    return f"Step {idx}: {body}"


def _serialize_steps(
    steps: List[str],
    max_visible_steps: int,
    max_chars_per_step: int,
    text_budget_chars: int,
    tail_block_min_keep: int,
    preserve_tail_markers: bool,
) -> str:
    source = [str(raw or "").strip() for raw in steps if str(raw or "").strip()]
    if not source:
        return ""
    if max_visible_steps > 0 and len(source) > max_visible_steps:
        source = source[-max_visible_steps:]

    n = len(source)
    tail_keep = max(1, min(tail_block_min_keep, n))
    tail_start = n - tail_keep
    preserved_tail = set(range(tail_start, n))
    if preserve_tail_markers:
        for i in range(n - 1, -1, -1):
            if i in preserved_tail:
                continue
            if _TAIL_CRITICAL_PATTERN.search(source[i]):
                preserved_tail.add(i)
            elif i < tail_start:
                break

    selected: List[int] = sorted(preserved_tail)
    for i in range(min(preserved_tail) - 1 if preserved_tail else n - 1, -1, -1):
        selected.insert(0, i)

    def render(indices: List[int], allow_tail_clip: bool = False) -> str:
        out: List[str] = []
        for j, idx in enumerate(indices, start=1):
            allow_clip = idx not in preserved_tail or allow_tail_clip
            out.append(_format_step(source[idx], idx=j, max_chars_per_step=max_chars_per_step, allow_clip=allow_clip))
        return "\n".join(out).strip()

    serialized = render(selected)
    if text_budget_chars <= 0 or len(serialized) <= text_budget_chars:
        return serialized

    trim_indices = list(selected)
    while trim_indices and len(render(trim_indices)) > text_budget_chars:
        removable_pos = next((pos for pos, idx in enumerate(trim_indices) if idx not in preserved_tail), None)
        if removable_pos is None:
            break
        trim_indices.pop(removable_pos)
    serialized = render(trim_indices)
    if len(serialized) <= text_budget_chars:
        return serialized
    return serialized[: max(8, text_budget_chars)].rstrip()


def _estimate_patch_len(steps: List[str]) -> int:
    total = 0
    for step in steps:
        text = str(step or "")
        for line in text.splitlines() or [text]:
            stripped = line.strip()
            if not stripped:
                continue
            if _PATCH_HINT_PATTERN.search(stripped) or stripped.startswith(("+", "-")):
                total += len(stripped)
    return total


def _estimate_tool_calls(steps: List[str]) -> int:
    count = 0
    for step in steps:
        if _TOOL_CALL_PATTERN.search(str(step or "")):
            count += 1
    return count


def _failure_bucket(episode: Dict[str, Any]) -> str:
    if episode.get("success") is True:
        return "success"
    meta = episode.get("raw_meta") if isinstance(episode.get("raw_meta"), dict) else {}
    if bool(meta.get("timeout")) or bool(meta.get("timed_out")):
        return "timeout"
    text = " ".join(
        [
            str(meta.get("error", "") or ""),
            str(meta.get("exception", "") or ""),
            str(meta.get("last_error", "") or ""),
            str(meta.get("eval_logs", "") or ""),
        ]
    ).lower()
    if ("traceback" in text) or ("error" in text) or ("exception" in text):
        return "error"
    return "other_failure"


def _load_terms(path: Path) -> List[str]:
    terms: List[str] = []
    if not path.exists():
        return terms
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip().lower()
        if not token or token.startswith("#"):
            continue
        terms.append(token)
    return terms


def _counter_to_sorted_dict(counter: Counter) -> Dict[str, int]:
    return {str(k): int(counter[k]) for k in sorted(counter.keys(), key=lambda x: int(x))}


def _build_candidates(
    raw_path: Path,
    k_candidates: int,
    max_tasks: int,
    seed: int,
    max_len: int,
    max_visible_steps: int,
    max_chars_per_step: int,
    tail_block_min_keep: int,
    preserve_tail_markers: bool,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ep in iter_episodes(str(raw_path)):
        task_id = str(ep.get("task_id", "") or "")
        if not task_id:
            continue
        if ep.get("success") not in {True, False}:
            continue
        grouped[task_id].append(ep)

    task_stats: Dict[str, Dict[str, int]] = {}
    for task_id, rows in grouped.items():
        success_count = sum(1 for x in rows if x.get("success") is True)
        failure_count = sum(1 for x in rows if x.get("success") is False)
        task_stats[task_id] = {
            "success_count": success_count,
            "failure_count": failure_count,
            "total_count": len(rows),
        }

    eligible_all = [task_id for task_id, st in task_stats.items() if int(st["total_count"]) >= k_candidates]
    eligible_mixed = [
        task_id
        for task_id, st in task_stats.items()
        if int(st["total_count"]) >= k_candidates and int(st["success_count"]) >= 1 and int(st["failure_count"]) >= 1
    ]

    eligible_all_sorted = sorted(eligible_all)
    eligible_mixed_sorted = sorted(eligible_mixed)
    rng.shuffle(eligible_all_sorted)
    rng.shuffle(eligible_mixed_sorted)

    selected_all = eligible_all_sorted[: max_tasks] if max_tasks > 0 else eligible_all_sorted
    selected_mixed = eligible_mixed_sorted[: max_tasks] if max_tasks > 0 else eligible_mixed_sorted

    selected_union = sorted(set(selected_all) | set(selected_mixed))

    candidates: List[Dict[str, Any]] = []
    skipped_due_empty_text = 0
    effective_union: List[str] = []

    for task_id in selected_union:
        rows = grouped[task_id]
        st = task_stats[task_id]
        if len(rows) < k_candidates:
            continue

        succ = [x for x in rows if x.get("success") is True]
        fail = [x for x in rows if x.get("success") is False]

        if succ and fail:
            s_pick = rng.choice(succ)
            f_pick = rng.choice(fail)
            chosen = [s_pick, f_pick]
            rest = [x for x in rows if x is not s_pick and x is not f_pick]
            chosen.extend(rng.sample(rest, k_candidates - 2))
        else:
            chosen = rng.sample(rows, k_candidates)
        rng.shuffle(chosen)

        instruction = str(chosen[0].get("instruction_refined") or chosen[0].get("instruction_raw") or task_id).strip()
        traj_budget = max(0, max_len - 2 - len(instruction) - len(PAIR_SEP))

        task_rows: List[Dict[str, Any]] = []
        for idx, ep in enumerate(chosen):
            steps = [str(x) for x in ep.get("steps", [])]
            trajectory_text = _serialize_steps(
                steps=steps,
                max_visible_steps=max_visible_steps,
                max_chars_per_step=max_chars_per_step,
                text_budget_chars=traj_budget,
                tail_block_min_keep=tail_block_min_keep,
                preserve_tail_markers=preserve_tail_markers,
            )
            if not trajectory_text:
                task_rows = []
                break

            task_rows.append(
                {
                    "task_id": task_id,
                    "candidate_id": f"{task_id}__{idx}",
                    "instruction_refined": instruction,
                    "trajectory_text": trajectory_text,
                    "success_label": bool(ep.get("success") is True),
                    "steps_len": len(steps),
                    "token_proxy": len(instruction) + len(trajectory_text),
                    "meta": {
                        "episode_id": ep.get("episode_id"),
                        "run_id": ep.get("run_id"),
                        "env": ep.get("env"),
                        "patch_len": _estimate_patch_len(steps),
                        "tool_calls": _estimate_tool_calls(steps),
                        "failure_bucket": _failure_bucket(ep),
                        "task_success_count": int(st["success_count"]),
                        "task_failure_count": int(st["failure_count"]),
                        "is_mixed_task": bool(int(st["success_count"]) >= 1 and int(st["failure_count"]) >= 1),
                    },
                }
            )

        if len(task_rows) != k_candidates:
            skipped_due_empty_text += 1
            continue

        candidates.extend(task_rows)
        effective_union.append(task_id)

    effective_union_set = set(effective_union)
    selected_all_effective = sorted([t for t in selected_all if t in effective_union_set])
    selected_mixed_effective = sorted([t for t in selected_mixed if t in effective_union_set])

    selected_stats = [task_stats[t] for t in selected_all_effective]
    succ_hist = Counter(int(x["success_count"]) for x in selected_stats)
    fail_hist = Counter(int(x["failure_count"]) for x in selected_stats)
    type_counter = Counter()
    for x in selected_stats:
        s = int(x["success_count"])
        f = int(x["failure_count"])
        if s >= 1 and f >= 1:
            type_counter["mixed"] += 1
        elif s >= 1 and f == 0:
            type_counter["all_success"] += 1
        elif s == 0 and f >= 1:
            type_counter["all_failure"] += 1
        else:
            type_counter["unknown"] += 1

    task_mix_stats = {
        "all_tasks_with_labels": len(grouped),
        "eligible_all_at_k": len(eligible_all),
        "eligible_mixed_at_k": len(eligible_mixed),
        "selected_all_count": len(selected_all_effective),
        "selected_mixed_count": len(selected_mixed_effective),
        "selected_union_count": len(effective_union_set),
        "per_task_success_count": _counter_to_sorted_dict(succ_hist),
        "per_task_failure_count": _counter_to_sorted_dict(fail_hist),
        "task_type_breakdown": dict(type_counter),
    }

    return {
        "candidates": candidates,
        "task_id_sets": {
            "selected_all": selected_all_effective,
            "selected_mixed": selected_mixed_effective,
            "selected_union": sorted(effective_union_set),
        },
        "coverage": {
            "eligible_task_count_all": len(eligible_all),
            "eligible_task_count_mixed": len(eligible_mixed),
            "used_task_count_all": len(selected_all_effective),
            "used_task_count_mixed": len(selected_mixed_effective),
            "used_task_count_union": len(effective_union_set),
            "skipped_due_empty_text": skipped_due_empty_text,
        },
        "task_mix_stats": task_mix_stats,
    }


def _pick_random(items: List[Dict[str, Any]], rng: random.Random) -> Dict[str, Any]:
    return items[rng.randrange(len(items))]


def _pick_shortest(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return min(items, key=lambda x: (int(x.get("steps_len", 0)), str(x.get("candidate_id", ""))))


def _pick_rm(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return max(items, key=lambda x: (float(x.get("rm_score", 0.0)), -int(x.get("steps_len", 0)), str(x.get("candidate_id", ""))))


def _calc_policy_metrics(selected: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not selected:
        return {
            "success_rate": 0.0,
            "avg_steps_len": 0.0,
            "avg_token_proxy": 0.0,
            "failure_type_distribution": {},
            "n_tasks": 0,
        }
    success = [1.0 if bool(x.get("success_label", False)) else 0.0 for x in selected]
    steps = [float(x.get("steps_len", 0)) for x in selected]
    tokens = [float(x.get("token_proxy", 0)) for x in selected]
    fail_counter: Counter = Counter()
    for row in selected:
        if bool(row.get("success_label", False)):
            continue
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        fail_counter[str(meta.get("failure_bucket", "unknown"))] += 1
    return {
        "success_rate": float(np.mean(success)),
        "avg_steps_len": float(np.mean(steps)),
        "avg_token_proxy": float(np.mean(tokens)),
        "failure_type_distribution": dict(fail_counter),
        "n_tasks": len(selected),
    }


def _bootstrap_delta_ci(per_task_outcomes: List[Dict[str, Any]], n_bootstrap: int, seed: int) -> Dict[str, Any]:
    n = len(per_task_outcomes)
    if n <= 0:
        return {
            "bootstrap_mean": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "n_bootstrap": int(n_bootstrap),
            "seed": int(seed),
        }

    rand_arr = np.array([float(x["random_success"]) for x in per_task_outcomes], dtype=np.float64)
    short_arr = np.array([float(x["shortest_success"]) for x in per_task_outcomes], dtype=np.float64)
    rm_arr = np.array([float(x["rm_success"]) for x in per_task_outcomes], dtype=np.float64)

    rng = np.random.default_rng(int(seed))
    deltas: List[float] = []
    for _ in range(max(int(n_bootstrap), 1)):
        idx = rng.integers(0, n, size=n)
        r = float(np.mean(rand_arr[idx]))
        s = float(np.mean(short_arr[idx]))
        m = float(np.mean(rm_arr[idx]))
        deltas.append(m - max(r, s))

    delta_np = np.array(deltas, dtype=np.float64)
    return {
        "bootstrap_mean": float(np.mean(delta_np)),
        "ci_low": float(np.percentile(delta_np, 2.5)),
        "ci_high": float(np.percentile(delta_np, 97.5)),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }


def _evaluate_slice(
    task_ids: List[str],
    by_task: Dict[str, List[Dict[str, Any]]],
    rng_seed: int,
    marker_terms: List[str],
    n_bootstrap: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    rng_random = random.Random(rng_seed)
    selected_random: List[Dict[str, Any]] = []
    selected_shortest: List[Dict[str, Any]] = []
    selected_rm: List[Dict[str, Any]] = []
    rm_minus_shortest_steps: List[float] = []

    for task_id in sorted(task_ids):
        items = by_task[task_id]
        rand_pick = _pick_random(items, rng_random)
        short_pick = _pick_shortest(items)
        rm_pick = _pick_rm(items)
        selected_random.append(rand_pick)
        selected_shortest.append(short_pick)
        selected_rm.append(rm_pick)
        rm_minus_shortest_steps.append(float(rm_pick.get("steps_len", 0)) - float(short_pick.get("steps_len", 0)))

    policy_metrics = {
        "baseline_random": _calc_policy_metrics(selected_random),
        "baseline_shortest": _calc_policy_metrics(selected_shortest),
        "rm_rerank": _calc_policy_metrics(selected_rm),
    }

    best_baseline = max(
        float(policy_metrics["baseline_random"]["success_rate"]),
        float(policy_metrics["baseline_shortest"]["success_rate"]),
    )
    rm_success = float(policy_metrics["rm_rerank"]["success_rate"])
    delta_rm_vs_best = rm_success - best_baseline

    def _hit_any(text: str) -> bool:
        lower = str(text or "").lower()
        return any(term in lower for term in marker_terms)

    selected_rm_marker_hit_rate = float(np.mean([1.0 if _hit_any(x.get("trajectory_text", "")) else 0.0 for x in selected_rm])) if selected_rm else 0.0

    per_task_outcomes: List[Dict[str, Any]] = []
    for a, b, c in zip(selected_random, selected_shortest, selected_rm):
        per_task_outcomes.append(
            {
                "task_id": str(c.get("task_id", "")),
                "random_success": int(bool(a.get("success_label", False))),
                "shortest_success": int(bool(b.get("success_label", False))),
                "rm_success": int(bool(c.get("success_label", False))),
            }
        )

    delta_ci = _bootstrap_delta_ci(per_task_outcomes=per_task_outcomes, n_bootstrap=n_bootstrap, seed=bootstrap_seed)

    return {
        "n_tasks": len(task_ids),
        "success_rate_by_policy": {
            "baseline_random": policy_metrics["baseline_random"]["success_rate"],
            "baseline_shortest": policy_metrics["baseline_shortest"]["success_rate"],
            "rm_rerank": policy_metrics["rm_rerank"]["success_rate"],
        },
        "delta_vs_baselines": {
            "best_baseline_success_rate": best_baseline,
            "rm_minus_best_baseline": delta_rm_vs_best,
        },
        "efficiency_metrics": {
            "baseline_random": {
                "avg_steps_len": policy_metrics["baseline_random"]["avg_steps_len"],
                "avg_token_proxy": policy_metrics["baseline_random"]["avg_token_proxy"],
            },
            "baseline_shortest": {
                "avg_steps_len": policy_metrics["baseline_shortest"]["avg_steps_len"],
                "avg_token_proxy": policy_metrics["baseline_shortest"]["avg_token_proxy"],
            },
            "rm_rerank": {
                "avg_steps_len": policy_metrics["rm_rerank"]["avg_steps_len"],
                "avg_token_proxy": policy_metrics["rm_rerank"]["avg_token_proxy"],
            },
        },
        "failure_type_distribution": {
            "baseline_random": policy_metrics["baseline_random"]["failure_type_distribution"],
            "baseline_shortest": policy_metrics["baseline_shortest"]["failure_type_distribution"],
            "rm_rerank": policy_metrics["rm_rerank"]["failure_type_distribution"],
        },
        "audit_metrics": {
            "rm_minus_shortest_steps_mean": float(np.mean(rm_minus_shortest_steps)) if rm_minus_shortest_steps else 0.0,
            "selected_rm_marker_hit_rate": selected_rm_marker_hit_rate,
        },
        "delta_ci": delta_ci,
        "per_task_outcomes": per_task_outcomes,
    }


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _parse_bool_text(value: str) -> bool:
    v = str(value or "").strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean text: {value}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run v0.6.1 SWE offline planning eval (random/shortest/rm_rerank).")
    ap.add_argument("--raw_path", type=str, default="data/raw/episodes_swe_v060.jsonl")
    ap.add_argument("--ckpt", type=str, default="results/v060_swe/rm.pt")
    ap.add_argument("--results_dir", type=str, default="results/v061_swe_planning")
    ap.add_argument("--candidates_path", type=str, default="")
    ap.add_argument("--scores_path", type=str, default="")
    ap.add_argument("--report_path", type=str, default="")
    ap.add_argument("--k_candidates", type=int, default=8)
    ap.add_argument("--n_tasks", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--max_visible_steps", type=int, default=0)
    ap.add_argument("--max_chars_per_step", type=int, default=64)
    ap.add_argument("--tail_block_min_keep", type=int, default=3)
    ap.add_argument("--preserve_tail_markers", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--task_slice", type=str, default="mixed", choices=["mixed", "all"])
    ap.add_argument("--emit_all_slice_report", type=str, default="true", choices=["true", "false"])
    ap.add_argument("--n_bootstrap", type=int, default=2000)
    ap.add_argument("--bootstrap_seed", type=int, default=-1)
    ap.add_argument("--success_delta_threshold", type=float, default=0.03)
    ap.add_argument("--audit_abs_corr_threshold", type=float, default=0.20)
    ap.add_argument("--forbidden_markers_file", type=str, default="configs/forbidden_markers_v060.json")
    ap.add_argument("--leak_terms_file", type=str, default="configs/leak_terms_v050.txt")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = Path(args.candidates_path) if args.candidates_path else (results_dir / "planning_candidates.jsonl")
    scores_path = Path(args.scores_path) if args.scores_path else (results_dir / "planning_scored_candidates.jsonl")
    report_path = Path(args.report_path) if args.report_path else (results_dir / "planning_eval.json")

    emit_all_slice_report = _parse_bool_text(args.emit_all_slice_report)
    bootstrap_seed = int(args.bootstrap_seed) if int(args.bootstrap_seed) >= 0 else int(args.seed)

    built = _build_candidates(
        raw_path=Path(args.raw_path),
        k_candidates=args.k_candidates,
        max_tasks=args.n_tasks,
        seed=args.seed,
        max_len=args.max_len,
        max_visible_steps=args.max_visible_steps,
        max_chars_per_step=args.max_chars_per_step,
        tail_block_min_keep=args.tail_block_min_keep,
        preserve_tail_markers=bool(args.preserve_tail_markers),
    )
    _write_jsonl(candidates_path, built["candidates"])

    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "score_v061_swe_candidates.py"),
            "--ckpt",
            str(args.ckpt),
            "--candidates_path",
            str(candidates_path),
            "--output_path",
            str(scores_path),
            "--batch_size",
            str(args.batch_size),
            "--max_len",
            str(args.max_len),
            "--device",
            str(args.device),
        ]
    )

    scored = _read_jsonl(scores_path)
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in scored:
        task_id = str(row.get("task_id", "") or "")
        if task_id:
            by_task[task_id].append(row)

    marker_cfg = load_forbidden_markers(repo_root / args.forbidden_markers_file)
    marker_terms = [str(x).lower() for x in marker_cfg.get("hard_leak_markers", []) + marker_cfg.get("soft_marker_audit", [])]
    marker_terms.extend(_load_terms(repo_root / args.leak_terms_file))
    marker_terms = sorted(set([x for x in marker_terms if x]))

    scored_task_ids = {k for k, v in by_task.items() if len(v) >= args.k_candidates}
    selected_all_set = set(built["task_id_sets"]["selected_all"])
    selected_mixed_set = set(built["task_id_sets"]["selected_mixed"])

    slice_all_ids = sorted(scored_task_ids & selected_all_set)
    slice_mixed_ids = sorted(scored_task_ids & selected_mixed_set)

    slice_metrics: Dict[str, Dict[str, Any]] = {}
    if args.task_slice == "all" or emit_all_slice_report:
        slice_metrics["all"] = _evaluate_slice(
            task_ids=slice_all_ids,
            by_task=by_task,
            rng_seed=args.seed + 17,
            marker_terms=marker_terms,
            n_bootstrap=args.n_bootstrap,
            bootstrap_seed=bootstrap_seed,
        )

    slice_metrics["mixed"] = _evaluate_slice(
        task_ids=slice_mixed_ids,
        by_task=by_task,
        rng_seed=args.seed + 29,
        marker_terms=marker_terms,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=bootstrap_seed + 1,
    )

    selected_slice = args.task_slice
    selected_metrics = slice_metrics.get(selected_slice, {"n_tasks": 0, "delta_vs_baselines": {"rm_minus_best_baseline": 0.0}, "delta_ci": {"ci_low": 0.0, "ci_high": 0.0, "bootstrap_mean": 0.0, "n_bootstrap": args.n_bootstrap, "seed": bootstrap_seed}})

    scores_np = np.array([float(x.get("rm_score", 0.0)) for x in scored], dtype=np.float64)
    steps_np = np.array([float(x.get("steps_len", 0.0)) for x in scored], dtype=np.float64)
    tokens_np = np.array([float(x.get("token_proxy", 0.0)) for x in scored], dtype=np.float64)
    audit_metrics = {
        "rm_score_steps_len_corr": _safe_corr(scores_np, steps_np),
        "rm_score_token_proxy_corr": _safe_corr(scores_np, tokens_np),
    }

    delta_selected = float(selected_metrics.get("delta_vs_baselines", {}).get("rm_minus_best_baseline", 0.0))
    delta_ci = selected_metrics.get("delta_ci", {"bootstrap_mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_bootstrap": args.n_bootstrap, "seed": bootstrap_seed})

    checks = {
        "slice_has_tasks": int(selected_metrics.get("n_tasks", 0)) > 0,
        "delta_threshold_ok": delta_selected >= float(args.success_delta_threshold),
        "delta_ci_lower_gt_zero": float(delta_ci.get("ci_low", 0.0)) > 0.0,
        "audit_steps_corr_ok": abs(float(audit_metrics["rm_score_steps_len_corr"])) < float(args.audit_abs_corr_threshold),
        "audit_token_corr_ok": abs(float(audit_metrics["rm_score_token_proxy_corr"])) < float(args.audit_abs_corr_threshold),
    }
    passed = all(checks.values())

    pass_rules = {
        "task_slice": selected_slice,
        "success_delta_threshold": float(args.success_delta_threshold),
        "delta_ci_rule": "ci_low > 0",
        "audit_abs_corr_threshold": float(args.audit_abs_corr_threshold),
    }

    selected_success = selected_metrics.get("success_rate_by_policy", {})
    selected_eff = selected_metrics.get("efficiency_metrics", {})
    selected_failure = selected_metrics.get("failure_type_distribution", {})
    selected_delta = selected_metrics.get("delta_vs_baselines", {})

    report = {
        "raw_path": str(args.raw_path),
        "ckpt": str(args.ckpt),
        "k_candidates": args.k_candidates,
        "requested_n_tasks": args.n_tasks,
        "seed": args.seed,
        "selected_slice": selected_slice,
        "coverage": {
            "eligible_task_count_all": int(built["coverage"]["eligible_task_count_all"]),
            "eligible_task_count_mixed": int(built["coverage"]["eligible_task_count_mixed"]),
            "used_task_count_all": int(built["coverage"]["used_task_count_all"]),
            "used_task_count_mixed": int(built["coverage"]["used_task_count_mixed"]),
            "used_task_count_union": int(built["coverage"]["used_task_count_union"]),
            "scored_task_count_all": len(slice_all_ids),
            "scored_task_count_mixed": len(slice_mixed_ids),
            "skipped_due_empty_text": int(built["coverage"]["skipped_due_empty_text"]),
        },
        "task_mix_stats": built["task_mix_stats"],
        "slice_metrics": {
            "mixed": slice_metrics.get("mixed", {}),
            "all": slice_metrics.get("all", {}),
        },
        "success_rate_by_policy": selected_success,
        "delta_vs_baselines": selected_delta,
        "efficiency_metrics": selected_eff,
        "failure_type_distribution": selected_failure,
        "audit_metrics": audit_metrics,
        "delta_ci": delta_ci,
        "pass_rules": pass_rules,
        "checks": checks,
        "artifacts": {
            "candidates_path": str(candidates_path),
            "scores_path": str(scores_path),
        },
        "passed": passed,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report_path": str(report_path),
                "passed": passed,
                "selected_slice": selected_slice,
                "scored_tasks": int(selected_metrics.get("n_tasks", 0)),
                "delta": delta_selected,
                "ci_low": float(delta_ci.get("ci_low", 0.0)),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
