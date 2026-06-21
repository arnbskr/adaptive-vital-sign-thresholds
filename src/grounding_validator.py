"""Lightweight numeric grounding validator (Phase 3 guardrail).

A descriptive, non-clinical answer must not invent numbers: every figure it
states should come from a deterministic tool output (or the evidence card built
from one). This module checks exactly that, cheaply and deterministically, with
no LLM call.

Approach: collect every number that appears anywhere in the recorded tool trace
(values, counts, percentiles, itemids, age-group bounds, window hours -- all of
it), then verify each number in the final answer is supported by that set, with
simple rounding tolerance (so ``1.40`` matches ``1.4``). Structural numbers
(age groups like ``65-74``, windows like ``first_24h``, metric names like
``p90``) are naturally grounded because they also appear in the tool payloads.

Design choices (kept simple and defensible):
- It is **advisory**: it never blocks an answer, it only adds a warning.
- It is **lenient** (rounding tolerance, ignores phase numbers / years) so it
  flags clearly-invented numbers, not formatting noise.
"""

from __future__ import annotations

import json
import re
from typing import Any

# A number token: optional sign, digits, optional decimals. Percent sign is left
# out (we compare the bare number).
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
# Boilerplate stripped before extracting answer numbers (these carry non-data
# digits): "Phase 1/2/3" labels and 4-digit calendar years.
_PHASE_RE = re.compile(r"(?i)phase\s*\d+")
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")


def _to_float(token: str) -> float | None:
    try:
        return float(token)
    except (TypeError, ValueError):
        return None


def extract_numbers_from_text(text: str) -> list[str]:
    """Return the data-number tokens in a piece of text (answer)."""

    if not text:
        return []
    cleaned = _PHASE_RE.sub(" ", str(text))
    numbers: list[str] = []
    for match in _NUMBER_RE.finditer(cleaned):
        start = match.start()
        # Skip digits that are part of a label token (e.g. "Q1", "Q3", "P90",
        # "p25"): a number glued to a preceding letter is a label, not a value.
        if start > 0 and cleaned[start - 1].isalpha():
            continue
        token = match.group(0)
        if _YEAR_RE.match(token.lstrip("-")):
            continue  # ignore calendar years (2024, 2026, ...)
        numbers.append(token)
    return numbers


def _numbers_in_object(obj: Any) -> set[float]:
    """Every number anywhere in a JSON-serializable object (keys + values)."""

    try:
        text = json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(obj)
    out: set[float] = set()
    for match in _NUMBER_RE.finditer(text):
        value = _to_float(match.group(0))
        if value is not None:
            out.add(value)
    return out


def extract_numbers_from_tool_outputs(tool_trace: list[dict]) -> set[float]:
    """Collect every number from the inputs and outputs of all tool calls."""

    supported: set[float] = set()
    for entry in tool_trace or []:
        for key in ("outputs", "inputs"):
            if key in entry:
                supported |= _numbers_in_object(entry.get(key))
    return supported


def _is_supported(value: float, supported: set[float]) -> bool:
    """Supported if equal to a tool number under simple rounding tolerance."""

    for target in supported:
        if abs(value - target) <= 1e-6:
            return True
        if round(value, 1) == round(target, 1):  # 1.40 vs 1.4
            return True
        if round(value) == round(target):        # 79 reported for 79.2
            return True
    return False


def validate_numeric_grounding(answer: str, tool_trace: list[dict]) -> dict[str, Any]:
    """Check that the numbers in ``answer`` are supported by the tool trace.

    Returns a dict with ``is_grounded`` (bool), the lists of numbers found,
    supported and unsupported, and a ``warning`` (or ``None``). Advisory only --
    it never rewrites or blocks the answer.
    """

    supported_set = extract_numbers_from_tool_outputs(tool_trace)
    answer_numbers = extract_numbers_from_text(answer)

    supported: list[str] = []
    unsupported: list[str] = []
    for token in answer_numbers:
        value = _to_float(token)
        if value is None:
            continue
        (supported if _is_supported(value, supported_set) else unsupported).append(token)

    # If there is no tool trace at all, we cannot judge -- stay silent (do not
    # block). Likewise an answer with no numbers is trivially grounded.
    if not answer_numbers or not supported_set:
        return {
            "is_grounded": True,
            "numbers_in_answer": answer_numbers,
            "numbers_supported": supported,
            "numbers_unsupported": [],
            "warning": None,
        }

    is_grounded = not unsupported
    warning = None
    if unsupported:
        warning = (
            f"Numeric grounding: {len(unsupported)} number(s) in the answer are not found in the "
            f"tool outputs ({', '.join(unsupported)}). Treat them with caution."
        )
    return {
        "is_grounded": is_grounded,
        "numbers_in_answer": answer_numbers,
        "numbers_supported": supported,
        "numbers_unsupported": unsupported,
        "warning": warning,
    }
