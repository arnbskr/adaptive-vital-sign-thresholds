"""Tool: calculatrice_medicale.

A deterministic, demonstration-only arithmetic tool. It exists so the single
agent can choose a *calculation* tool when a question is purely arithmetic,
instead of querying ChromaDB or the MIMIC-IV tools -- showing dynamic tool
selection. It is NOT a clinical calculator and must never be used for diagnosis.

Security: it never uses ``eval``. The expression is parsed with ``ast`` and only
numeric literals and the basic binary/unary arithmetic operators are evaluated.
Any function call, attribute access, name, string, import, etc. is rejected.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        return _BINARY_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Only basic arithmetic on numbers is allowed (+ - * / // % ** and parentheses).")


def calculatrice_medicale(expression: str) -> dict[str, Any]:
    """Evaluate a simple numeric arithmetic expression. Demonstration tool only.

    Returns a JSON-serializable dict:
      {"tool": "calculatrice_medicale", "expression": "104 * 2", "result": 208, "status": "ok"}
    or, on rejection:
      {"tool": "calculatrice_medicale", "expression": "...", "status": "error", "error": "invalid_expression"}
    """

    try:
        tree = ast.parse(str(expression), mode="eval")
        result = _safe_eval(tree.body)
    except Exception:  # noqa: BLE001 - structured error, never crash
        return {
            "tool": "calculatrice_medicale",
            "expression": expression,
            "status": "error",
            "error": "invalid_expression",
        }
    return {
        "tool": "calculatrice_medicale",
        "expression": expression,
        "result": result,
        "status": "ok",
        "note": "Demonstration arithmetic tool; not a clinical calculation.",
    }
