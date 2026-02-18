#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap_ci(values: List[float], n_bootstrap: int, seed: int) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    n = len(values)
    samples: List[float] = []
    for _ in range(max(int(n_bootstrap), 1)):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(float(statistics.mean(draw)))
    samples_sorted = sorted(samples)
    low_idx = max(0, int(0.025 * len(samples_sorted)) - 1)
    high_idx = min(len(samples_sorted) - 1, int(0.975 * len(samples_sorted)) - 1)
    return {
        "mean": float(statistics.mean(values)),
        "ci_low": float(samples_sorted[low_idx]),
        "ci_high": float(samples_sorted[high_idx]),
    }


def _render_md(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# v0.6.1 SWE Planning Matrix Summary")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    verdict = summary["verdict"]
    lines.append(f"- passed: `{verdict['passed']}`")
    lines.append(f"- reason: `{verdict['reason']}`")
    lines.append(f"- success_delta_threshold: `{verdict['success_delta_threshold']}`")
    lines.append(f"- audit_abs_corr_threshold: `{verdict['audit_abs_corr_threshold']}`")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    agg = summary["aggregate"]
    lines.append(f"- mean_delta: `{agg['mean_delta']:.4f}`")
    lines.append(f"- std_delta: `{agg['std_delta']:.4f}`")
    lines.append(f"- bootstrap_ci_95: `[{agg['bootstrap_ci']['ci_low']:.4f}, {agg['bootstrap_ci']['ci_high']:.4f}]`")
    lines.append("")
    lines.append("## Per Seed")
    lines.append("")
    lines.append("| seed | passed | delta_rm_vs_best | delta_ci_low | delta_ci_high | steps_corr | token_corr |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for row in summary["per_seed_results"]:
        lines.append(
            f"| {row['seed']} | {row['passed']} | {row['delta_rm_vs_best']:.4f} | {row['delta_ci_low']:.4f} | "
            f"{row['delta_ci_high']:.4f} | {row['rm_score_steps_len_corr']:.4f} | {row['rm_score_token_proxy_corr']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize v0.6.1 SWE planning multi-seed matrix.")
    ap.add_argument("--matrix_json", type=str, default="results/v061_swe_planning_matrix/matrix_raw.json")
    ap.add_argument("--output_json", type=str, default="results/v061_swe_planning_matrix/matrix_summary.json")
    ap.add_argument("--output_md", type=str, default="results/v061_swe_planning_matrix/matrix_summary.md")
    ap.add_argument("--n_bootstrap", type=int, default=5000)
    ap.add_argument("--bootstrap_seed", type=int, default=123)
    ap.add_argument("--success_delta_threshold", type=float, default=0.03)
    ap.add_argument("--audit_abs_corr_threshold", type=float, default=0.20)
    args = ap.parse_args()

    matrix = _load_json(Path(args.matrix_json))
    per_seed = matrix.get("per_seed_reports", [])
    deltas = [float(x.get("delta_rm_vs_best", 0.0)) for x in per_seed]
    if deltas:
        mean_delta = float(statistics.mean(deltas))
        std_delta = float(statistics.pstdev(deltas)) if len(deltas) > 1 else 0.0
    else:
        mean_delta = 0.0
        std_delta = 0.0

    boot = _bootstrap_ci(values=deltas, n_bootstrap=args.n_bootstrap, seed=args.bootstrap_seed)

    audit_ok = all(
        abs(float(x.get("rm_score_steps_len_corr", 0.0))) < args.audit_abs_corr_threshold
        and abs(float(x.get("rm_score_token_proxy_corr", 0.0))) < args.audit_abs_corr_threshold
        for x in per_seed
    )
    passed = (
        bool(per_seed)
        and mean_delta >= args.success_delta_threshold
        and float(boot["ci_low"]) > 0.0
        and audit_ok
    )
    reason = "all_rules_passed" if passed else "one_or_more_rules_failed"

    summary = {
        "matrix_json": str(args.matrix_json),
        "ckpt_selection": matrix.get("ckpt_selection", {}),
        "per_seed_results": per_seed,
        "aggregate": {
            "mean_delta": mean_delta,
            "std_delta": std_delta,
            "bootstrap_ci": {
                "mean": float(boot["mean"]),
                "ci_low": float(boot["ci_low"]),
                "ci_high": float(boot["ci_high"]),
                "n_bootstrap": args.n_bootstrap,
                "seed": args.bootstrap_seed,
            },
        },
        "verdict": {
            "passed": passed,
            "reason": reason,
            "success_delta_threshold": args.success_delta_threshold,
            "audit_abs_corr_threshold": args.audit_abs_corr_threshold,
            "checks": {
                "has_seed_runs": bool(per_seed),
                "mean_delta_ok": mean_delta >= args.success_delta_threshold,
                "bootstrap_ci_low_gt_zero": float(boot["ci_low"]) > 0.0,
                "audit_corr_all_ok": audit_ok,
            },
        },
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_render_md(summary), encoding="utf-8")
    print(json.dumps({"output_json": str(out_json), "output_md": str(out_md), "passed": passed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
