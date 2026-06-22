"""Controlled, session-only Phase 3 memory for simple follow-up questions.

Scope is deliberately tiny and safe:
- **Session-only** (lives in ``st.session_state``), no long-term store.
- Stores **only the last query context** — never raw MIMIC data, never
  patient-level rows, never sensitive values: ``last_variable``,
  ``last_age_group``, ``last_time_window``, ``last_metric``, ``last_intent``,
  ``last_tool``.

It lets a user ask "What about creatinine?" after "Summarize lactate for 75-84 in
the first 24h." by **rewriting** the follow-up into a full, standalone Phase 3
question that the existing intent detector already understands. This is NOT a
conversational memory or a reasoning store — just last-context substitution.
"""

from __future__ import annotations

import re
from typing import Any

from .agent import _extract_age_group, _extract_metric, _extract_time_window, _resolve_variable

MEMORY_FIELDS = ("last_variable", "last_age_group", "last_time_window", "last_metric", "last_intent", "last_tool")

_FOLLOWUP_TRIGGERS = (
    "what about", "and for", "how about", "what if", "now ", "also for", "instead",
    "same for", "what's the", "now p",
)
_AGE_COMPARE = ("across age", "age groups", "by age group", "which age group", "between age")
_WINDOW_COMPARE = ("across time", "time windows", "over time", "between time")


def empty_phase3_memory() -> dict[str, Any]:
    return {}


def clear_phase3_memory() -> dict[str, Any]:
    """Return a fresh, empty memory (caller assigns it back to the session)."""

    return empty_phase3_memory()


def update_phase3_memory(memory: dict[str, Any] | None, result: dict[str, Any]) -> dict[str, Any]:
    """Update the last-context memory from a Phase 3 agent result.

    Only updates on Phase 3 data answers (not refusals / non-Phase-3). Reads the
    intent (graph engine) or the evidence card (both engines). Stores only strings.
    """

    memory = dict(memory or {})
    question_type = str(result.get("question_type", ""))
    if not question_type.startswith("phase3_"):
        return memory

    intent = result.get("intent") or {}
    card = result.get("evidence_card") or {}

    def _pick(intent_key: str, card_key: str) -> str | None:
        value = intent.get(intent_key)
        if isinstance(value, str) and value:
            return value
        cval = card.get(card_key)
        return cval if isinstance(cval, str) and cval else None

    variable = _pick("variable_name", "variable")
    age_group = _pick("age_group", "age_group")
    time_window = _pick("time_window", "time_window")
    metric = intent.get("metric") if isinstance(intent.get("metric"), str) else None

    if variable:
        memory["last_variable"] = variable
    if age_group:
        memory["last_age_group"] = age_group
    if time_window:
        memory["last_time_window"] = time_window
    if metric:
        memory["last_metric"] = metric
    memory["last_intent"] = question_type.replace("phase3_", "")
    tools = result.get("tools_called") or []
    data_tools = [t for t in tools if t not in ("detect_clinical_advice_request", "generate_evidence_card")]
    if data_tools:
        memory["last_tool"] = data_tools[-1]
    return memory


def resolve_followup_question(question: str, memory: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    """Rewrite a follow-up into a full Phase 3 question using session memory.

    Returns ``(question, resolved_context)``. If the question is already a
    complete standalone Phase 3 question, or there is nothing to resolve from,
    returns it unchanged with an empty ``resolved_context``.
    """

    if not memory or not memory.get("last_variable"):
        return question, {}

    q = question.lower()
    new_var = _resolve_variable(q)
    new_age = _extract_age_group(q)
    new_window = _extract_time_window(q)
    new_metric = _extract_metric(q)
    compare_age = any(k in q for k in _AGE_COMPARE)
    compare_window = any(k in q for k in _WINDOW_COMPARE)
    mentions_it = bool(re.search(r"\bit\b", q))
    explicit_followup = mentions_it or any(t in q for t in _FOLLOWUP_TRIGGERS)

    has_context = bool(new_age or new_window or compare_age or compare_window)
    # A complete, standalone question (variable + context, no follow-up cue) is left as is.
    if new_var and has_context and not explicit_followup:
        return question, {}
    # If it carries no follow-up cue and no new slot at all, do not touch it.
    if not explicit_followup and not (new_var or new_age or new_window or new_metric or compare_age or compare_window):
        return question, {}

    variable = new_var or memory.get("last_variable")
    age_group = new_age or memory.get("last_age_group")
    time_window = new_window or memory.get("last_time_window")
    metric = new_metric or memory.get("last_metric")
    if not variable:
        return question, {}

    if compare_age:
        intent = "compare_age_groups"
        rewritten = f"Compare {variable} across age groups in {time_window or 'first_24h'}."
    elif compare_window:
        intent = "compare_time_windows"
        rewritten = f"Compare {variable} across time windows for {age_group or '75-84'}."
    elif metric and metric not in ("median",):
        intent = "variable_summary"
        rewritten = f"What is the {metric} of {variable} for {age_group or '75-84'} in {time_window or 'first_24h'}?"
    else:
        intent = "variable_summary"
        rewritten = f"Summarize {variable} for {age_group or '75-84'} in {time_window or 'first_24h'}."

    resolved = {
        "variable": variable, "age_group": age_group, "time_window": time_window,
        "metric": metric, "intent": intent, "rewritten_question": rewritten,
    }
    return rewritten, resolved
