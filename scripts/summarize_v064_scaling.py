#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_non_decreasing(values: List[float], eps: float = 1e-9) -> bool:
    for i in range(1, len(values)):
        if float(values[i]) + eps < float(values[i - 1]):
            return False
    return True


def _is_cost_increasing(values: List[float], tolerance: float) -> bool:
    if tolerance < 0:
        tolerance = 0.0
    for i in range(1, len(values)):
        prev = float(values[i - 1])
        cur = float(values[i])
        if prev <= 0:
            if cur < 0:
                return False
            continue
        if cur < prev * (1.0 - tolerance):
            return False
    return True


def _render_md(summary: Dict[str, Any], curve_points: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# v0.6.4 Scaling Summary")
    lines.append("")
    lines.append("## Conclusions")
    lines.append("")
    lines.append(f"- delta_monotonic_non_decreasing: `{bool(summary.get('delta_monotonic_non_decreasing', False))}`")
    lines.append(f"- cost_monotonic_increasing: `{bool(summary.get('cost_monotonic_increasing', False))}`")
    lines.append(f"- best_tradeoff_bon_n: `{summary.get('best_tradeoff_bon_n')}`")
    lines.append("")
    lines.append("## Curve Points")
    lines.append("")
    lines.append("| bon_n | delta_mean | ci_low_mean | avg_cost | avg_rm_forward_calls | timeout_rate_mean |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in curve_points:
        lines.append(
            f"| {int(row.get('bon_n', 0))} | {float(row.get('delta_mean', 0.0)):.4f} | "
            f"{float(row.get('ci_low_mean', 0.0)):.4f} | {float(row.get('avg_cost', 0.0)):.4f} | "
            f"{float(row.get('avg_rm_forward_calls_mean', 0.0)):.2f} | {float(row.get('timeout_rate_mean', 0.0)):.4f} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize v0.6.4 scaling matrix results.")
    ap.add_argument("--matrix_json", type=str, required=True)
    ap.add_argument("--output_json", type=str, default="")
    ap.add_argument("--output_md", type=str, default="")
    ap.add_argument("--curve_points_json", type=str, default="")
    ap.add_argument("--cost_tolerance", type=float, default=0.05)
    args = ap.parse_args()

    matrix_path = Path(args.matrix_json)
    matrix = _read_json(matrix_path)

    if args.output_json.strip():
        output_json = Path(args.output_json)
    else:
        output_json = matrix_path.parent / "matrix_summary.json"
    if args.output_md.strip():
        output_md = Path(args.output_md)
    else:
        output_md = matrix_path.parent / "matrix_summary.md"
    if args.curve_points_json.strip():
        curve_points_json = Path(args.curve_points_json)
    else:
        curve_points_json = matrix_path.parent / "curve_points.json"

    aggs = list(matrix.get("aggregate_by_bon_n", []))
    aggs = sorted(aggs, key=lambda x: int(x.get("bon_n", 0)))

    curve_points: List[Dict[str, Any]] = []
    for item in aggs:
        curve_points.append(
            {
                "bon_n": int(item.get("bon_n", 0)),
                "delta_mean": float(item.get("delta_mean", 0.0)),
                "ci_low_mean": float(item.get("ci_low_mean", 0.0)),
                "avg_cost": float(item.get("avg_wall_time_sec_mean", 0.0)),
                "avg_rm_forward_calls_mean": float(item.get("avg_rm_forward_calls_mean", 0.0)),
                "timeout_rate_mean": float(item.get("timeout_rate_mean", 0.0)),
                "n_cells": int(item.get("n_cells", 0)),
            }
        )

    delta_vals = [float(x["delta_mean"]) for x in curve_points]
    cost_vals = [float(x["avg_cost"]) for x in curve_points]

    delta_monotonic = _is_non_decreasing(delta_vals) if curve_points else False
    cost_monotonic = _is_cost_increasing(cost_vals, tolerance=float(args.cost_tolerance)) if curve_points else False

    eligible = [x for x in curve_points if float(x.get("ci_low_mean", 0.0)) > 0.0]
    pool = eligible if eligible else curve_points
    best_tradeoff = None
    if pool:
        best_tradeoff = sorted(
            pool,
            key=lambda x: (
                -(float(x.get("delta_mean", 0.0)) / max(float(x.get("avg_cost", 0.0)), 1e-6)),
                -float(x.get("delta_mean", 0.0)),
                float(x.get("avg_cost", 0.0)),
                int(x.get("bon_n", 0)),
            ),
        )[0]

    by_bon = {int(x.get("bon_n", 0)): x for x in curve_points}
    endpoint_checks = {
        "has_bon_1_and_8": (1 in by_bon and 8 in by_bon),
        "delta_mean_8_gt_1": (float(by_bon[8]["delta_mean"]) > float(by_bon[1]["delta_mean"])) if (1 in by_bon and 8 in by_bon) else False,
        "ci_low_mean_8_ge_1": (float(by_bon[8]["ci_low_mean"]) >= float(by_bon[1]["ci_low_mean"])) if (1 in by_bon and 8 in by_bon) else False,
    }

    summary = {
        "matrix_json": str(matrix_path),
        "n_cells": int(len(matrix.get("cells", []))),
        "n_curve_points": len(curve_points),
        "delta_monotonic_non_decreasing": bool(delta_monotonic),
        "cost_monotonic_increasing": bool(cost_monotonic),
        "best_tradeoff_bon_n": (int(best_tradeoff["bon_n"]) if best_tradeoff else None),
        "best_tradeoff_reason": (
            "maximize delta/cost among points with ci_low_mean>0; fallback to all points" if best_tradeoff else "no valid points"
        ),
        "endpoint_checks": endpoint_checks,
        "cost_tolerance": float(args.cost_tolerance),
        "curve_points_path": str(curve_points_json),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    curve_points_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    curve_points_json.write_text(json.dumps(curve_points, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(_render_md(summary=summary, curve_points=curve_points), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_md": str(output_md),
                "curve_points_json": str(curve_points_json),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
