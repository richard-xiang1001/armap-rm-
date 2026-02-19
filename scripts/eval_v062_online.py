#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Allow `python3 scripts/eval_v062_online.py` from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from ingest.rules import load_forbidden_markers


POLICIES = ("baseline_no_planning", "baseline_random_bon", "rm_bon")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _bootstrap_delta_ci(per_task: List[Dict[str, int]], n_bootstrap: int, seed: int) -> Dict[str, Any]:
    n = len(per_task)
    if n <= 0:
        return {
            "bootstrap_mean": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "n_bootstrap": int(n_bootstrap),
            "seed": int(seed),
        }

    no_plan = np.array([float(x["baseline_no_planning"]) for x in per_task], dtype=np.float64)
    rnd = np.array([float(x["baseline_random_bon"]) for x in per_task], dtype=np.float64)
    rm = np.array([float(x["rm_bon"]) for x in per_task], dtype=np.float64)

    rng = np.random.default_rng(int(seed))
    deltas: List[float] = []
    for _ in range(max(int(n_bootstrap), 1)):
        idx = rng.integers(0, n, size=n)
        b0 = float(np.mean(no_plan[idx]))
        b1 = float(np.mean(rnd[idx]))
        r = float(np.mean(rm[idx]))
        deltas.append(r - max(b0, b1))

    arr = np.array(deltas, dtype=np.float64)
    return {
        "bootstrap_mean": float(np.mean(arr)),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }


def _load_terms(path: Path) -> List[str]:
    out: List[str] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        token = str(line or "").strip().lower()
        if not token or token.startswith("#"):
            continue
        out.append(token)
    return out


def _hit_any(text: str, marker_terms: List[str]) -> bool:
    body = str(text or "").lower()
    return any(term in body for term in marker_terms)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate v0.6.2 online traces with CI and audit gates.")
    ap.add_argument("--traces_path", type=str, required=True)
    ap.add_argument("--output_json", type=str, required=True)
    ap.add_argument("--n_bootstrap", type=int, default=2000)
    ap.add_argument("--bootstrap_seed", type=int, default=42)
    ap.add_argument("--success_delta_threshold", type=float, default=0.03)
    ap.add_argument("--audit_abs_corr_threshold", type=float, default=0.20)
    ap.add_argument("--min_runnable_task_rate", type=float, default=0.85)
    ap.add_argument("--min_test_exec_success_rate", type=float, default=0.80)
    ap.add_argument("--max_timeout_rate", type=float, default=0.10)
    ap.add_argument("--forbidden_markers_file", type=str, default="configs/forbidden_markers_v060.json")
    ap.add_argument("--leak_terms_file", type=str, default="configs/leak_terms_v050.txt")
    args = ap.parse_args()

    rows = _read_jsonl(Path(args.traces_path))
    by_policy: Dict[str, Dict[str, Dict[str, Any]]] = {k: {} for k in POLICIES}
    for row in rows:
        p = str(row.get("policy") or "")
        t = str(row.get("task_id") or "")
        if p in by_policy and t:
            by_policy[p][t] = row

    common_tasks = sorted(set(by_policy[POLICIES[0]].keys()) & set(by_policy[POLICIES[1]].keys()) & set(by_policy[POLICIES[2]].keys()))

    success_rate_by_policy: Dict[str, float] = {}
    efficiency_metrics: Dict[str, Dict[str, float]] = {}
    per_task_outcomes: List[Dict[str, int]] = []

    for policy in POLICIES:
        subset = [by_policy[policy][task_id] for task_id in common_tasks]
        if subset:
            success_rate_by_policy[policy] = float(np.mean([1.0 if bool(x.get("success", False)) else 0.0 for x in subset]))
            efficiency_metrics[policy] = {
                "avg_steps_len": float(np.mean([float(x.get("n_steps", 0)) for x in subset])),
                "avg_token_proxy": float(np.mean([float(x.get("total_token_proxy", 0)) for x in subset])),
                "avg_rm_forward_calls": float(np.mean([float(x.get("total_rm_forward_calls", 0)) for x in subset])),
                "avg_wall_time_sec": float(np.mean([float(x.get("wall_time_sec", 0.0)) for x in subset])),
            }
        else:
            success_rate_by_policy[policy] = 0.0
            efficiency_metrics[policy] = {
                "avg_steps_len": 0.0,
                "avg_token_proxy": 0.0,
                "avg_rm_forward_calls": 0.0,
                "avg_wall_time_sec": 0.0,
            }

    for task_id in common_tasks:
        per_task_outcomes.append(
            {
                "task_id": task_id,
                "baseline_no_planning": int(bool(by_policy["baseline_no_planning"][task_id].get("success", False))),
                "baseline_random_bon": int(bool(by_policy["baseline_random_bon"][task_id].get("success", False))),
                "rm_bon": int(bool(by_policy["rm_bon"][task_id].get("success", False))),
            }
        )

    best_baseline = max(float(success_rate_by_policy["baseline_no_planning"]), float(success_rate_by_policy["baseline_random_bon"]))
    rm_rate = float(success_rate_by_policy["rm_bon"])
    delta = rm_rate - best_baseline
    delta_ci = _bootstrap_delta_ci(per_task=per_task_outcomes, n_bootstrap=args.n_bootstrap, seed=args.bootstrap_seed)

    marker_cfg = load_forbidden_markers(ROOT / args.forbidden_markers_file)
    marker_terms = [str(x).lower() for x in marker_cfg.get("hard_leak_markers", []) + marker_cfg.get("soft_marker_audit", [])]
    marker_terms.extend(_load_terms(ROOT / args.leak_terms_file))
    marker_terms = sorted(set([x for x in marker_terms if x]))

    score_vals: List[float] = []
    step_vals: List[float] = []
    token_vals: List[float] = []
    marker_hits: List[float] = []

    for task_id in common_tasks:
        row = by_policy["rm_bon"][task_id]
        for step in list(row.get("steps") or []):
            selected = str(step.get("selected_candidate_id") or "")
            candidates = list(step.get("candidates") or [])
            selected_row = None
            for cand in candidates:
                if str(cand.get("candidate_id") or "") == selected:
                    selected_row = cand
                    break
            if selected_row is None:
                continue
            score = selected_row.get("rm_score")
            if score is None:
                continue
            sim_info = dict(selected_row.get("sim_info") or {})
            step_len = float(sim_info.get("committed_steps_len", 0))
            tok = float(selected_row.get("token_proxy", 0))
            text_a = str(selected_row.get("action_text") or "")
            text_b = str(step.get("env_info", {}).get("committed_text", "") or "")

            score_vals.append(float(score))
            step_vals.append(step_len)
            token_vals.append(tok)
            marker_hits.append(1.0 if _hit_any(text_a + "\n" + text_b, marker_terms=marker_terms) else 0.0)

    score_np = np.array(score_vals, dtype=np.float64)
    step_np = np.array(step_vals, dtype=np.float64)
    token_np = np.array(token_vals, dtype=np.float64)

    audit_metrics = {
        "rm_score_steps_len_corr": _safe_corr(score_np, step_np),
        "rm_score_token_proxy_corr": _safe_corr(score_np, token_np),
        "selected_rm_marker_hit_rate": float(np.mean(marker_hits)) if marker_hits else 0.0,
    }

    planning_checks = {
        "slice_has_tasks": len(common_tasks) > 0,
        "delta_threshold_ok": float(delta) >= float(args.success_delta_threshold),
        "delta_ci_lower_gt_zero": float(delta_ci.get("ci_low", 0.0)) > 0.0,
        "audit_steps_corr_ok": abs(float(audit_metrics["rm_score_steps_len_corr"])) < float(args.audit_abs_corr_threshold),
        "audit_token_corr_ok": abs(float(audit_metrics["rm_score_token_proxy_corr"])) < float(args.audit_abs_corr_threshold),
    }

    rm_infos: List[Dict[str, Any]] = []
    for task_id in common_tasks:
        row = by_policy["rm_bon"][task_id]
        for step in list(row.get("steps") or []):
            rm_infos.append(dict(step.get("env_info") or {}))

    has_executor_fields = any(
        (
            info.get("patch_apply_ok") is not None
            or info.get("tests_passed") is not None
            or str(info.get("exec_error_type") or "").strip().lower() not in {"", "simulated"}
        )
        for info in rm_infos
    )

    executor_health = {
        "gating_in_effect": bool(has_executor_fields),
        "runnable_task_rate": None,
        "patch_apply_ok_rate": None,
        "test_exec_success_rate": None,
        "timeout_rate": None,
        "exec_error_breakdown": {},
    }
    executor_health_checks = {
        "runnable_task_rate_ok": True,
        "test_exec_success_rate_ok": True,
        "timeout_rate_ok": True,
    }
    if has_executor_fields:
        fatal_non_runnable = {"manifest_missing", "prepare_failed", "docker_missing", "not_runnable"}
        runnable_count = 0
        patch_ok_count = 0
        test_ok_count = 0
        timeout_count = 0
        err_counter: Dict[str, int] = {}
        total = len(rm_infos)
        for info in rm_infos:
            err = str(info.get("exec_error_type") or "none").strip() or "none"
            err_counter[err] = int(err_counter.get(err, 0)) + 1
            if err == "timeout":
                timeout_count += 1
            runnable = err not in fatal_non_runnable
            if runnable:
                runnable_count += 1
                if bool(info.get("patch_apply_ok") is True):
                    patch_ok_count += 1
                if bool(info.get("tests_passed") is True):
                    test_ok_count += 1

        runnable_task_rate = float(runnable_count / total) if total > 0 else 0.0
        patch_apply_ok_rate = float(patch_ok_count / runnable_count) if runnable_count > 0 else 0.0
        test_exec_success_rate = float(test_ok_count / runnable_count) if runnable_count > 0 else 0.0
        timeout_rate = float(timeout_count / total) if total > 0 else 0.0

        executor_health = {
            "gating_in_effect": True,
            "runnable_task_rate": runnable_task_rate,
            "patch_apply_ok_rate": patch_apply_ok_rate,
            "test_exec_success_rate": test_exec_success_rate,
            "timeout_rate": timeout_rate,
            "exec_error_breakdown": err_counter,
        }
        executor_health_checks = {
            "runnable_task_rate_ok": runnable_task_rate >= float(args.min_runnable_task_rate),
            "test_exec_success_rate_ok": test_exec_success_rate >= float(args.min_test_exec_success_rate),
            "timeout_rate_ok": timeout_rate <= float(args.max_timeout_rate),
        }

    checks = dict(planning_checks)
    checks.update(executor_health_checks)
    passed = all(planning_checks.values()) and all(executor_health_checks.values())

    report = {
        "traces_path": str(args.traces_path),
        "n_tasks": len(common_tasks),
        "success_rate_by_policy": success_rate_by_policy,
        "delta_vs_baselines": {
            "best_baseline_success_rate": best_baseline,
            "rm_minus_best_baseline": delta,
        },
        "delta_ci": delta_ci,
        "efficiency_metrics": efficiency_metrics,
        "audit_metrics": audit_metrics,
        "executor_health": executor_health,
        "planning_checks": planning_checks,
        "executor_health_checks": executor_health_checks,
        "checks": checks,
        "pass_rules": {
            "success_delta_threshold": float(args.success_delta_threshold),
            "delta_ci_rule": "ci_low > 0",
            "audit_abs_corr_threshold": float(args.audit_abs_corr_threshold),
            "min_runnable_task_rate": float(args.min_runnable_task_rate),
            "min_test_exec_success_rate": float(args.min_test_exec_success_rate),
            "max_timeout_rate": float(args.max_timeout_rate),
        },
        "passed": passed,
        "per_task_outcomes": per_task_outcomes,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(output_path), "passed": passed, "n_tasks": len(common_tasks), "delta": delta}, ensure_ascii=False))


if __name__ == "__main__":
    main()
