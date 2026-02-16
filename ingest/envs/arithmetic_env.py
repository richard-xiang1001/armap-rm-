from __future__ import annotations

import ast
import re
from typing import Dict, List, Optional

from ingest.envs.protocol import EnvJudge, JudgeResult


_COMPUTE_EXPR_PATTERN = re.compile(r"(?i)\bcompute\s+([^\.\?]+)")
_VALUE_EXPR_PATTERN = re.compile(r"(?i)\bvalue\s+of\s*:\s*([^\.\?]+)")
_FINAL_PATTERN = re.compile(r"(?i)^\s*final\s*:\s*([-+]?\d+)")
_VERIFY_PATTERN = re.compile(r"(?i)candidate\s+answer\s+([-+]?\d+)")
_SUCCESS_PATTERN = re.compile(r"(?i)^\s*success\s*:\s*(true|false|1|0|yes|no)")


class _SafeArithmeticEvaluator(ast.NodeVisitor):
    """Small evaluator for +,-,* and integer constants."""

    def visit_Expression(self, node: ast.Expression) -> int:  # noqa: N802
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> int:  # noqa: N802
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        raise ValueError("unsupported operator")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> int:  # noqa: N802
        val = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.UAdd):
            return val
        raise ValueError("unsupported unary operator")

    def visit_Constant(self, node: ast.Constant) -> int:  # noqa: N802
        if isinstance(node.value, int):
            return int(node.value)
        raise ValueError("constant must be int")

    def generic_visit(self, node):  # noqa: D401
        raise ValueError(f"unsupported node: {type(node).__name__}")


def extract_expression(instruction: str) -> Optional[str]:
    text = str(instruction or "")
    m = _VALUE_EXPR_PATTERN.search(text)
    if m:
        return m.group(1).strip()

    m = _COMPUTE_EXPR_PATTERN.search(text)
    if not m:
        return None
    expr = m.group(1).strip()
    expr = re.sub(r"(?i)^the\s+value\s+of\s*:\s*", "", expr).strip()
    return expr


def eval_expression(expr: str) -> Optional[int]:
    try:
        tree = ast.parse(expr, mode="eval")
        return int(_SafeArithmeticEvaluator().visit(tree))
    except Exception:
        return None


def parse_target_from_instruction(instruction: str) -> tuple[Optional[int], Dict]:
    expr = extract_expression(instruction)
    if not expr:
        return None, {"expr": None, "parse_error": "expr_not_found"}
    target = eval_expression(expr)
    if target is None:
        return None, {"expr": expr, "parse_error": "expr_eval_failed"}
    return target, {"expr": expr}


def parse_final_from_steps(steps: List[str]) -> tuple[Optional[int], int]:
    final_value: Optional[int] = None
    final_idx = -1
    for idx, line in enumerate(steps):
        m = _FINAL_PATTERN.match(str(line))
        if m:
            final_value = int(m.group(1))
            final_idx = idx
    return final_value, final_idx


def parse_verify_candidate(steps: List[str]) -> Optional[int]:
    for line in reversed(steps):
        m = _VERIFY_PATTERN.search(str(line))
        if m:
            return int(m.group(1))
    return None


def parse_success_flag(steps: List[str]) -> Optional[bool]:
    for line in reversed(steps):
        m = _SUCCESS_PATTERN.match(str(line))
        if not m:
            continue
        val = m.group(1).lower()
        return val in {"true", "1", "yes"}
    return None


def judge_arithmetic_episode(instruction: str, steps: List[str], meta: Dict | None = None) -> JudgeResult:
    target, info = parse_target_from_instruction(instruction)
    if target is None:
        return JudgeResult(success=None, terminal=None, score=None, replay_ok=False, reason="target_parse_failed", extras=info)

    final_value, final_idx = parse_final_from_steps(steps)
    if final_value is None:
        extras = {**info, "target": target, "final": None}
        return JudgeResult(success=False, terminal=False, score=0.0, replay_ok=True, reason="final_missing", extras=extras)

    non_empty = [str(x).strip() for x in steps if str(x).strip()]
    final_is_last = bool(non_empty and _FINAL_PATTERN.match(non_empty[-1]))
    verify_candidate = parse_verify_candidate(steps)
    success_flag = parse_success_flag(steps)

    value_ok = final_value == target
    verify_ok = verify_candidate is None or verify_candidate == final_value
    success_flag_ok = success_flag is None or success_flag == value_ok
    structure_ok = final_is_last

    judged_success = bool(value_ok and verify_ok and success_flag_ok and structure_ok)
    reason = "ok" if judged_success else "constraint_failed"

    extras = {
        **info,
        "target": target,
        "final": final_value,
        "final_idx": final_idx,
        "final_is_last": final_is_last,
        "verify_candidate": verify_candidate,
        "success_flag": success_flag,
        "value_ok": value_ok,
        "verify_ok": verify_ok,
        "success_flag_ok": success_flag_ok,
        "structure_ok": structure_ok,
    }
    return JudgeResult(
        success=judged_success,
        terminal=final_is_last,
        score=1.0 if judged_success else 0.0,
        replay_ok=True,
        reason=reason,
        extras=extras,
    )


class ArithmeticEnvJudge(EnvJudge):
    name = "arithmetic"
    protocol_version = "v0.4.1"

    def judge(self, instruction: str, steps: List[str], meta: Dict | None = None) -> JudgeResult:
        return judge_arithmetic_episode(instruction=instruction, steps=steps, meta=meta)
