#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _render_md(rep: Dict[str, Any]) -> str:
    rates = rep.get("success_rate_by_policy", {})
    eff = rep.get("efficiency_metrics", {})
    delta = rep.get("delta_vs_baselines", {})
    ci = rep.get("delta_ci", {})
    checks = rep.get("checks", {})
    audit = rep.get("audit_metrics", {})
    ex = rep.get("executor_health", {})

    lines: List[str] = []
    lines.append("# v0.6.2 Online BoN Summary")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- passed: `{rep.get('passed', False)}`")
    lines.append(f"- n_tasks: `{rep.get('n_tasks', 0)}`")
    lines.append(f"- rm_minus_best_baseline: `{float(delta.get('rm_minus_best_baseline', 0.0)):.4f}`")
    lines.append(f"- delta_ci_95: `[{float(ci.get('ci_low', 0.0)):.4f}, {float(ci.get('ci_high', 0.0)):.4f}]`")
    lines.append(f"- checks: `{json.dumps(checks, ensure_ascii=False)}`")
    lines.append("")

    lines.append("## Policy Metrics")
    lines.append("")
    lines.append("| policy | success_rate | avg_steps_len | avg_token_proxy | avg_rm_forward_calls | avg_wall_time_sec |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for key in ("baseline_no_planning", "baseline_random_bon", "rm_bon"):
        item = eff.get(key, {})
        lines.append(
            f"| {key} | {float(rates.get(key, 0.0)):.4f} | {float(item.get('avg_steps_len', 0.0)):.2f} | "
            f"{float(item.get('avg_token_proxy', 0.0)):.2f} | {float(item.get('avg_rm_forward_calls', 0.0)):.2f} | "
            f"{float(item.get('avg_wall_time_sec', 0.0)):.3f} |"
        )
    lines.append("")

    lines.append("## Audit")
    lines.append("")
    lines.append(f"- rm_score_steps_len_corr: `{float(audit.get('rm_score_steps_len_corr', 0.0)):.4f}`")
    lines.append(f"- rm_score_token_proxy_corr: `{float(audit.get('rm_score_token_proxy_corr', 0.0)):.4f}`")
    lines.append(f"- selected_rm_marker_hit_rate: `{float(audit.get('selected_rm_marker_hit_rate', 0.0)):.4f}`")
    lines.append("")
    if ex:
        lines.append("## Executor Health")
        lines.append("")
        lines.append(f"- gating_in_effect: `{bool(ex.get('gating_in_effect', False))}`")
        if ex.get("runnable_task_rate") is not None:
            lines.append(f"- runnable_task_rate: `{float(ex.get('runnable_task_rate', 0.0)):.4f}`")
        if ex.get("patch_apply_ok_rate") is not None:
            lines.append(f"- patch_apply_ok_rate: `{float(ex.get('patch_apply_ok_rate', 0.0)):.4f}`")
        if ex.get("test_exec_success_rate") is not None:
            lines.append(f"- test_exec_success_rate: `{float(ex.get('test_exec_success_rate', 0.0)):.4f}`")
        if ex.get("timeout_rate") is not None:
            lines.append(f"- timeout_rate: `{float(ex.get('timeout_rate', 0.0)):.4f}`")
        lines.append(f"- exec_error_breakdown: `{json.dumps(ex.get('exec_error_breakdown', {}), ensure_ascii=False)}`")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize v0.6.2 online eval JSON to markdown.")
    ap.add_argument("--eval_json", type=str, required=True)
    ap.add_argument("--output_json", type=str, required=True)
    ap.add_argument("--output_md", type=str, required=True)
    args = ap.parse_args()

    rep = _load_json(Path(args.eval_json))
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_render_md(rep), encoding="utf-8")
    print(json.dumps({"output_json": str(out_json), "output_md": str(out_md), "passed": bool(rep.get("passed", False))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
