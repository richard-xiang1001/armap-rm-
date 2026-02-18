#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _slice(rep: Dict[str, Any], name: str) -> Dict[str, Any]:
    obj = rep.get("slice_metrics", {}).get(name)
    if isinstance(obj, dict) and obj:
        return obj
    return {}


def _render_slice(lines: List[str], name: str, obj: Dict[str, Any]) -> None:
    if not obj:
        return
    rates = obj.get("success_rate_by_policy", {})
    eff = obj.get("efficiency_metrics", {})
    delta = obj.get("delta_vs_baselines", {})
    ci = obj.get("delta_ci", {})
    lines.append(f"## Slice `{name}`")
    lines.append("")
    lines.append(f"- n_tasks: `{obj.get('n_tasks', 0)}`")
    lines.append(f"- rm_minus_best_baseline: `{float(delta.get('rm_minus_best_baseline', 0.0)):.4f}`")
    lines.append(f"- delta_ci_95: `[{float(ci.get('ci_low', 0.0)):.4f}, {float(ci.get('ci_high', 0.0)):.4f}]`")
    lines.append("")
    lines.append("| policy | success_rate | avg_steps_len | avg_token_proxy |")
    lines.append("|---|---:|---:|---:|")
    for key in ("baseline_random", "baseline_shortest", "rm_rerank"):
        lines.append(
            f"| {key} | {float(rates.get(key, 0.0)):.4f} | "
            f"{float(eff.get(key, {}).get('avg_steps_len', 0.0)):.2f} | "
            f"{float(eff.get(key, {}).get('avg_token_proxy', 0.0)):.2f} |"
        )
    lines.append("")


def _render_md(rep: Dict[str, Any]) -> str:
    cov = rep.get("coverage", {})
    audit = rep.get("audit_metrics", {})
    checks = rep.get("checks", {})
    rules = rep.get("pass_rules", {})
    delta_ci = rep.get("delta_ci", {})

    lines: List[str] = []
    lines.append("# v0.6.1 SWE Planning Eval Summary")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- passed: `{rep.get('passed', False)}`")
    lines.append(f"- selected_slice: `{rep.get('selected_slice', 'mixed')}`")
    lines.append(f"- rm_minus_best_baseline: `{float(rep.get('delta_vs_baselines', {}).get('rm_minus_best_baseline', 0.0)):.4f}`")
    lines.append(f"- delta_ci_95: `[{float(delta_ci.get('ci_low', 0.0)):.4f}, {float(delta_ci.get('ci_high', 0.0)):.4f}]`")
    lines.append(f"- pass_rules: `{json.dumps(rules, ensure_ascii=False)}`")
    lines.append(f"- checks: `{json.dumps(checks, ensure_ascii=False)}`")
    lines.append("")

    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- eligible_task_count_all: `{cov.get('eligible_task_count_all', 0)}`")
    lines.append(f"- eligible_task_count_mixed: `{cov.get('eligible_task_count_mixed', 0)}`")
    lines.append(f"- scored_task_count_all: `{cov.get('scored_task_count_all', 0)}`")
    lines.append(f"- scored_task_count_mixed: `{cov.get('scored_task_count_mixed', 0)}`")
    lines.append("")

    _render_slice(lines, "mixed", _slice(rep, "mixed"))
    _render_slice(lines, "all", _slice(rep, "all"))

    lines.append("## Audit")
    lines.append("")
    lines.append(f"- rm_score_steps_len_corr: `{float(audit.get('rm_score_steps_len_corr', 0.0)):.4f}`")
    lines.append(f"- rm_score_token_proxy_corr: `{float(audit.get('rm_score_token_proxy_corr', 0.0)):.4f}`")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize v0.6.1 SWE planning eval JSON to markdown.")
    ap.add_argument("--report_path", type=str, default="results/v061_swe_planning/planning_eval.json")
    ap.add_argument("--output_json", type=str, default="results/v061_swe_planning/planning_eval.json")
    ap.add_argument("--output_md", type=str, default="results/v061_swe_planning/planning_eval.md")
    args = ap.parse_args()

    rep = _load_json(Path(args.report_path))
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_render_md(rep), encoding="utf-8")
    print(json.dumps({"output_json": str(out_json), "output_md": str(out_md), "passed": bool(rep.get("passed", False))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
