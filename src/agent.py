"""Phase 2 single-agent orchestrator.

ONE LLM agent. It extracts the patient context, classifies the question, decides
which deterministic tools to call, invokes them through the MCP boundary
(``src.mcp_server.call_tool``) while recording an auditable trace, then asks the
local LLM to phrase the final grounded answer from the tool outputs. The LLM
never performs the calculations -- it only reads tool results and writes prose.

This is deliberately a single orchestrator: no sub-agents, no supervisor, no
long-term memory, no real-time monitoring. Non-clinical by construction.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from .config import DEFAULT_TOOL_BACKEND
from .semantic_rag import LLM_MODEL, build_ollama_client, infer_patient_context
from .tool_client import ToolClient, get_tool_client, is_remote_backend
from .tool_trace import ToolTrace, save_trace

LOGGER = logging.getLogger(__name__)

MODE = "Phase 2 Agentic RAG with tools"
CLINICAL_WARNING = (
    "This response is for academic interpretation only and is not a clinical diagnosis "
    "or treatment recommendation."
)

# Keyword sets used for deterministic question classification.
_CONCEPT_KEYWORDS = ("difference", "threshold", "adaptive", "percentile", "standard", "what is", "explain", "define")
_DATASET_KEYWORDS = ("mimic", "table", "tables", "dataset", "chartevents", "d_items", "icustays", "patients", "itemid")
_PIPELINE_KEYWORDS = ("pipeline", "ingest", "chunk", "index", "embedding", "preprocessing", "retrieval", "rag", "why should", "why are")
_HIGH_KEYWORDS = ("high", "elevated", "above", "too high", "tachy", "hyper")
_LOW_KEYWORDS = ("low", "below", "too low", "hypo", "drop")


def _classify(question: str, context: dict[str, Any]) -> tuple[str, list[str]]:
    """Return (question_type, missing_fields). Patient questions take priority."""

    lowered = question.lower()
    patient_intent = bool(
        context.get("vital_sign")
        and context.get("value") is not None
        and (context.get("age") is not None or context.get("age_group"))
    )
    if patient_intent:
        required = {
            "vital_sign": context.get("vital_sign"),
            "age_group": context.get("age_group"),
            "value": context.get("value"),
            "time_window": context.get("time_window"),
        }
        missing = [name for name, value in required.items() if not value and value != 0]
        if missing:
            return "unsupported_or_missing_data", missing
        return "patient_value_question", []

    if any(keyword in lowered for keyword in _DATASET_KEYWORDS):
        return "dataset_question", []
    if any(keyword in lowered for keyword in _PIPELINE_KEYWORDS):
        return "pipeline_question", []
    return "concept_question", []


def _direction(question: str) -> str:
    lowered = question.lower()
    if any(keyword in lowered for keyword in _HIGH_KEYWORDS):
        return "high"
    if any(keyword in lowered for keyword in _LOW_KEYWORDS):
        return "low"
    return "neutral"


def _is_threshold_concept(question: str) -> bool:
    lowered = question.lower()
    return "threshold" in lowered or "percentile" in lowered or "adaptive" in lowered


# Pure-arithmetic detection: lets the agent pick `calculatrice_medicale` instead
# of the RAG/MIMIC tools when a question is just a calculation.
_CALC_TRIGGERS = (
    "what is", "what's", "calculate", "compute", "how much", "result of", "the result",
    "multiply", "multiplied", "times", "divide", "divided", "plus", " add ", "added",
    "sum of", "subtract", "minus",
)
_WORD_OPS = [
    (("multiplied by", "multiply", "times", "product of"), "*"),
    (("divided by", "divide", "divided", "over"), "/"),
    (("plus", "add ", "added", "sum of", "increase", "increased by"), "+"),
    (("minus", "subtract", "subtracted", "decrease", "decreased by", "less"), "-"),
]


def _extract_calculation(question: str) -> str | None:
    """Return a safe arithmetic expression if the question is purely arithmetic.

    Patient-value questions are classified earlier and never reach this; this only
    fires when an arithmetic trigger word is present, so ranges like '65-74' in a
    concept question are not mistaken for a subtraction.
    """

    text = question.lower()
    if not any(trigger in text for trigger in _CALC_TRIGGERS):
        return None

    # Symbolic expression using * / % + (these never appear in age-group ranges).
    match = re.search(r"\d+(?:\.\d+)?(?:\s*[*/%+]\s*\d+(?:\.\d+)?)+", question)
    if match:
        return match.group(0).strip()

    # Spaced minus distinguishes "104 - 2" from a range like "65-74".
    match = re.search(r"\d+(?:\.\d+)?\s+-\s+\d+(?:\.\d+)?", question)
    if match:
        return match.group(0).strip()

    # Natural language: an operation word plus the first two numbers.
    numbers = re.findall(r"\d+(?:\.\d+)?", question)
    if len(numbers) >= 2:
        for words, op in _WORD_OPS:
            if any(word in text for word in words):
                return f"{numbers[0]} {op} {numbers[1]}"
    return None


# --------------------------------------------------------------------------- #
# Phase 3 intent detection (multi-variable ICU explorer). Deterministic regex /
# keyword routing over the variables in icu_feature_summary.csv. Patient-value,
# concept, dataset, pipeline and calculator routing are untouched; Phase 3 only
# claims questions that would otherwise fall to the generic concept/dataset path.
# --------------------------------------------------------------------------- #
_VARIABLE_ALIASES: list[tuple[str, str]] = [
    ("mean arterial pressure", "map"), ("arterial pressure", "map"), ("map", "map"),
    ("systolic blood pressure", "sbp"), ("systolic", "sbp"), ("sbp", "sbp"),
    ("diastolic blood pressure", "dbp"), ("diastolic", "dbp"), ("dbp", "dbp"),
    ("heart rate", "heart_rate"), ("heart_rate", "heart_rate"), ("hr", "heart_rate"),
    ("respiratory rate", "respiratory_rate"), ("respiratory_rate", "respiratory_rate"),
    ("resp rate", "respiratory_rate"), ("rr", "respiratory_rate"),
    ("temperature", "temperature"), ("temp", "temperature"),
    ("oxygen saturation", "spo2"), ("o2 saturation", "spo2"), ("spo2", "spo2"),
    ("oxygen flow", "o2_flow"), ("o2 flow", "o2_flow"), ("o2_flow", "o2_flow"),
    ("glasgow coma", "gcs_total"), ("gcs_total", "gcs_total"), ("gcs", "gcs_total"),
    ("central venous", "cvp"), ("cvp", "cvp"), ("fio2", "fio2"), ("glucose", "glucose"),
    ("lactate", "lactate"), ("creatinine", "creatinine"), ("bilirubin", "bilirubin_total"),
    ("platelet", "platelets"), ("white blood", "wbc"), ("wbc", "wbc"),
    ("hemoglobin", "hemoglobin"), ("haemoglobin", "hemoglobin"), ("hgb", "hemoglobin"),
    ("sodium", "sodium"), ("potassium", "potassium"), ("bicarbonate", "bicarbonate"),
    ("blood ph", "ph_blood"), ("ph_blood", "ph_blood"), ("ph", "ph_blood"),
    ("pao2", "pao2"), ("paco2", "paco2"), ("inr", "inr"),
]
_VARIABLE_ALIASES.sort(key=lambda kv: len(kv[0]), reverse=True)

_AVAILABLE_PATTERNS = (
    "what variables", "which variables", "list variables", "list icu variables",
    "what icu variables", "variables are available", "available variables",
    "variables do you have", "what data is available", "what labs", "labs are available",
    "labs do you have", "what can you analyze", "what variables can",
)
_AGE_COMPARE_KEYWORDS = ("across age", "between age", "each age", "by age group", "which age group", "age groups")
_WINDOW_COMPARE_KEYWORDS = ("across time", "between time", "time window", "over time", "evolve", "evolution")
_METRICS = ("p05", "p25", "p50", "p75", "p90", "p95")

# A "strong" Phase 3 signal lets a multi-variable question win even when the
# patient-value classifier would otherwise grab it (e.g. a variable name plus an
# age range that looks like a numeric value). Genuine patient-value questions
# ("is HR 104 high?") contain none of these, so they are untouched.
_STRONG_PHASE3_SIGNALS = (
    "compare", "across", "summarize", "summarise", "statistics", "statistic",
    "evolve", "evolution", "distribution", "available", "list ", "which age",
    "time window", "over time", "highest", "lowest", "by age group",
    "p05", "p25", "p50", "p75", "p90", "p95",
)


def _is_strong_phase3(question: str) -> bool:
    t = question.lower()
    return any(signal in t for signal in _STRONG_PHASE3_SIGNALS)


def _resolve_variable(text: str) -> str | None:
    padded = f" {text.lower()} "
    for alias, canonical in _VARIABLE_ALIASES:
        if re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", padded):
            return canonical
    return None


def _extract_age_group(text: str) -> str | None:
    t = text.lower()
    if "85+" in t or re.search(r"\b85\s*(\+|plus|and older|or older|and over)\b", t):
        return "85+"
    if "75-84" in t or re.search(r"\b75\s*(?:-|–|to)\s*84\b", t):
        return "75-84"
    if "65-74" in t or re.search(r"\b65\s*(?:-|–|to)\s*74\b", t):
        return "65-74"
    match = re.search(r"\b(?:aged|age)\s*(\d{2,3})\b", t)
    if match:
        age = int(match.group(1))
        if age >= 85:
            return "85+"
        if age >= 75:
            return "75-84"
        if age >= 65:
            return "65-74"
    return None


def _windows_mentioned(text: str) -> list[str]:
    t = text.lower().replace("_", " ")
    found = []
    if re.search(r"\b6\s*h(?:ours?)?\b", t):
        found.append("first_6h")
    if re.search(r"\b12\s*h(?:ours?)?\b", t):
        found.append("first_12h")
    if re.search(r"\b24\s*h(?:ours?)?\b", t):
        found.append("first_24h")
    return found


def _extract_time_window(text: str) -> str | None:
    windows = _windows_mentioned(text)
    return windows[0] if len(windows) == 1 else None


def _extract_metric(text: str) -> str | None:
    t = text.lower()
    for metric in _METRICS:
        if metric in t:
            return metric
    if "median" in t:
        return "median"
    if "mean" in t or "average" in t:
        return "mean"
    return None


def _infer_phase3_intent(question: str) -> dict[str, Any] | None:
    """Return a Phase 3 intent dict, or None if the question is not Phase 3."""

    t = question.lower()
    if any(p in t for p in _AVAILABLE_PATTERNS):
        category = "lab" if ("lab" in t and "vital" not in t) else (
            "vital_sign" if ("vital" in t or "charted" in t) else None)
        return {"intent": "available_variables", "variable_category": category}

    variable = _resolve_variable(t)
    if not variable:
        return None

    age_group = _extract_age_group(t)
    time_window = _extract_time_window(t)
    metric = _extract_metric(t)

    age_compare = any(k in t for k in _AGE_COMPARE_KEYWORDS) or (
        "age group" in t and any(k in t for k in ("compare", "highest", "lowest", "higher", "lower")))
    window_compare = any(k in t for k in _WINDOW_COMPARE_KEYWORDS) or len(_windows_mentioned(t)) >= 2

    if age_compare:
        return {"intent": "compare_age_groups", "variable_name": variable,
                "time_window": time_window or "first_24h", "metric": metric}
    if window_compare:
        return {"intent": "compare_time_windows", "variable_name": variable,
                "age_group": age_group or "75-84", "metric": metric}
    if age_group and time_window:
        return {"intent": "variable_summary", "variable_name": variable,
                "age_group": age_group, "time_window": time_window, "metric": metric}
    return {"intent": "cohort_statistics", "variable_name": variable,
            "age_group": age_group, "time_window": time_window, "metric": metric}


def _call_llm(prompt: str) -> tuple[str, str | None]:
    """Call the local LLM. Returns (text, warning). On failure returns ("", warning)."""

    try:
        client = build_ollama_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content or "", None
    except Exception as exc:  # noqa: BLE001 - degrade gracefully if Ollama/model missing
        LOGGER.warning("LLM call failed (%s); returning deterministic tool-based answer.", exc)
        return "", f"LLM unavailable ({exc}); returned a deterministic answer assembled directly from the tool outputs."


# --------------------------------------------------------------------------- #
# Flow builders
# --------------------------------------------------------------------------- #

def _patient_prompt(question: str, context: dict[str, Any], report: dict[str, Any]) -> str:
    return f"""You are the single academic assistant agent for the ICU Trajectory RAG project.
Rephrase the deterministic tool results below into a clear, concise, non-clinical answer.
Do NOT invent numbers; use ONLY the values provided. Answer in the user's language.
Never use words like "critical", "dangerous", "emergency", "at risk", or any treatment advice.

Detected patient context: age={context.get('age')}, age_group={context.get('age_group')}, vital_sign={context.get('vital_sign')}, value={context.get('value')}, time_window={context.get('time_window')}.

Deterministic findings (from tools):
- Standard-threshold comparison: {report.get('standard_threshold_interpretation')}
- MIMIC-IV percentile comparison: {report.get('mimic_percentile_interpretation')}
- Structured short answer: {report.get('short_answer')}
- Sources used: {report.get('sources_used')}

Write:
1. A short direct answer (is the value high/low/within range, descriptively).
2. The standard-threshold interpretation.
3. The MIMIC-IV percentile interpretation.
4. A one-line limitation.
End with: "{CLINICAL_WARNING}"

Question: {question}
"""


def _context_prompt(question: str, rag_context: dict[str, Any], extra: str = "") -> str:
    blocks = []
    for chunk in rag_context.get("chunks", []):
        blocks.append(
            f"Source: {chunk.get('source_file')} ({chunk.get('source_type')})\n{chunk.get('chunk_text', '')}"
        )
    context_text = "\n\n---\n\n".join(blocks) if blocks else "No context retrieved."
    extra_block = f"\n\nAdditional deterministic explanation:\n{extra}" if extra else ""
    return f"""You are the single academic assistant agent for the ICU Trajectory RAG project.
Answer ONLY from the retrieved context below. If it is insufficient, say so explicitly.
Be grounded, concise, and non-clinical. Answer in the user's language.
End your answer with: "{CLINICAL_WARNING}"

Question: {question}

Retrieved context:
{context_text}{extra_block}
"""


def _phase3_prompt(question: str, intent: str, payload: dict[str, Any], evidence_card: dict[str, Any] | None) -> str:
    card_text = evidence_card.get("text", "") if isinstance(evidence_card, dict) else ""
    return f"""You are the single academic assistant agent for the ICU Multi-Data Explorer (Phase 3).
Rephrase the deterministic tool output below into a clear, concise, NON-CLINICAL answer.
Use ONLY the numbers provided; never invent values. Answer in the user's language.
Never diagnose, never recommend treatment, never use words like "critical", "dangerous", or "at risk".
If a missing-rate warning is present, mention that the figures reflect a capped sample.

Intent: {intent}
Question: {question}

Tool output (JSON-like):
{payload}

{card_text}

End your answer with: "{CLINICAL_WARNING}"
"""


def _phase3_deterministic_answer(intent: str, payload: dict[str, Any], evidence_card: dict[str, Any] | None) -> str:
    """Assemble an answer from tool output when the LLM is unavailable."""

    if payload.get("error"):
        known = payload.get("known_variables")
        extra = f" Known variables: {', '.join(known)}." if known else ""
        return f"{payload['error']}{extra}\n\n{CLINICAL_WARNING}"
    if intent == "available_variables":
        names = ", ".join(v["variable_name"] for v in payload.get("variables", []))
        body = f"{payload.get('count', 0)} ICU variables available ({names})."
    elif intent in ("variable_summary", "cohort_statistics"):
        body = payload.get("readable_summary") or str({k: payload.get(k) for k in ("variable_name", "median", "p90", "unit")})
    else:  # compare_*
        body = payload.get("descriptive", str(payload))
    card_text = f"\n\n{evidence_card.get('text', '')}" if isinstance(evidence_card, dict) and evidence_card.get("text") else ""
    return f"{body}{card_text}\n\n{CLINICAL_WARNING}"


def _run_phase3(
    question: str, intent: dict[str, Any], client: ToolClient, trace: ToolTrace, warnings: list[str]
) -> tuple[str, list[str], dict[str, Any] | None]:
    """Execute the right Phase 3 tool(s), build an evidence card, reformulate."""

    kind = intent["intent"]
    variable = intent.get("variable_name")
    age_group = intent.get("age_group")
    time_window = intent.get("time_window")
    metric = intent.get("metric")
    evidence_card: dict[str, Any] | None = None

    def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
        return trace.record(name, args, lambda: client.call_tool(name, args))

    if kind == "available_variables":
        payload = _call("list_available_variables", {"variable_category": intent.get("variable_category")})
    elif kind == "compare_age_groups":
        payload = _call("compare_age_groups", {"variable_name": variable, "time_window": time_window, "metric": metric})
        evidence_card = _call("generate_evidence_card", {
            "variable_name": variable, "time_window": time_window,
            "tool_name": "compare_age_groups", "main_metric": metric})
    elif kind == "compare_time_windows":
        payload = _call("compare_time_windows", {"variable_name": variable, "age_group": age_group, "metric": metric})
        evidence_card = _call("generate_evidence_card", {
            "variable_name": variable, "age_group": age_group,
            "tool_name": "compare_time_windows", "main_metric": metric})
    elif kind == "variable_summary":
        payload = _call("get_variable_summary", {
            "variable_name": variable, "age_group": age_group, "time_window": time_window})
        evidence_card = _call("generate_evidence_card", {
            "variable_name": variable, "age_group": age_group, "time_window": time_window,
            "tool_name": "get_variable_summary", "main_metric": metric})
    else:  # cohort_statistics
        payload = _call("query_cohort_statistics", {
            "variable_name": variable, "age_group": age_group, "time_window": time_window, "metric": metric})
        evidence_card = _call("generate_evidence_card", {
            "variable_name": variable, "age_group": age_group, "time_window": time_window,
            "tool_name": "query_cohort_statistics", "main_metric": metric})

    sources_used = [payload["source"]] if isinstance(payload, dict) and payload.get("source") else []
    if isinstance(payload, dict) and payload.get("error"):
        warnings.append(payload["error"])

    llm_answer, warning = _call_llm(_phase3_prompt(question, kind, payload, evidence_card))
    if warning:
        warnings.append(warning)
        answer = _phase3_deterministic_answer(kind, payload, evidence_card)
    else:
        answer = llm_answer
    return answer, sources_used, evidence_card


def _safe_retrieve_context(
    trace: ToolTrace, client: ToolClient, query: str, top_k: int
) -> dict[str, Any] | None:
    """Call the RAG tool through the backend, tolerating retrieval failures (e.g. Ollama down)."""

    try:
        return trace.record(
            "retrieve_project_context",
            {"query": query, "top_k": top_k},
            lambda: client.call_tool("retrieve_project_context", {"query": query, "top_k": top_k}),
        )
    except Exception as exc:  # noqa: BLE001 - trace already recorded the error
        LOGGER.warning("retrieve_project_context failed: %s", exc)
        return None


def _resolve_tool_client(
    tool_backend: str | None,
    tool_client: ToolClient | None,
    allow_fallback: bool,
) -> tuple[ToolClient, bool, list[str]]:
    """Return (client, owns_client, warnings).

    A caller-supplied ``tool_client`` is used as-is (and not closed here). Otherwise
    the requested backend is built; if a remote backend cannot be reached and
    ``allow_fallback`` is set, we degrade to the local in-process backend with a
    clear warning instead of crashing.
    """

    if tool_client is not None:
        return tool_client, False, []

    requested = tool_backend or DEFAULT_TOOL_BACKEND
    try:
        return get_tool_client(requested), True, []
    except Exception as exc:  # noqa: BLE001 - remote unreachable / SDK missing
        if is_remote_backend(requested) and allow_fallback:
            warning = (
                f"Remote MCP backend unavailable ({exc}); fell back to the local "
                "in-process backend. Start `python src/server_mcp.py` to use the remote backend."
            )
            LOGGER.warning(warning)
            return get_tool_client("local"), True, [warning]
        raise


def run_agent(
    question: str,
    top_k: int = 5,
    persist: bool = True,
    tool_backend: str | None = None,
    tool_client: ToolClient | None = None,
    allow_fallback: bool = True,
) -> dict[str, Any]:
    """Orchestrate tools + LLM for one question and return an auditable result.

    ``tool_backend`` selects the tool backend ("local" / "mcp_remote"); it
    defaults to ``DEFAULT_TOOL_BACKEND`` (env ``PHASE2_TOOL_BACKEND``). Pass a
    ready ``tool_client`` to reuse an existing connection. The orchestration,
    medical logic and non-clinical guarantees are identical across backends.
    """

    client, owns_client, warnings = _resolve_tool_client(tool_backend, tool_client, allow_fallback)
    try:
        return _run_agent_with_client(question, top_k, persist, client, warnings)
    finally:
        if owns_client:
            client.close()


def _run_agent_with_client(
    question: str,
    top_k: int,
    persist: bool,
    client: ToolClient,
    warnings: list[str],
) -> dict[str, Any]:
    trace = ToolTrace(backend=client.name)

    # Safety gate FIRST (deterministic, never depends on the LLM): a diagnosis /
    # treatment request is refused non-clinically before any data tool runs.
    safety = trace.record(
        "detect_clinical_advice_request", {"question": question},
        lambda: client.call_tool("detect_clinical_advice_request", {"question": question}),
    )
    clinical_refused = bool(isinstance(safety, dict) and safety.get("is_clinical_advice"))

    context = infer_patient_context(question)
    base_type, missing_fields = _classify(question, context)
    question_type = base_type

    phase3_intent: dict[str, Any] | None = None
    calc_expression: str | None = None
    if clinical_refused:
        question_type = "clinical_advice_refused"
    else:
        phase3_intent = _infer_phase3_intent(question)
        if phase3_intent and _is_strong_phase3(question):
            # A strong multi-variable signal overrides patient/concept routing.
            question_type = "phase3_" + phase3_intent["intent"]
        elif base_type in {"concept_question", "dataset_question", "pipeline_question"}:
            # Within the generic path, a (weaker) Phase 3 intent still wins over
            # RAG; otherwise a purely arithmetic question routes to the calculator.
            if phase3_intent:
                question_type = "phase3_" + phase3_intent["intent"]
            else:
                phase3_intent = None
                calc_expression = _extract_calculation(question)
                if calc_expression:
                    question_type = "calculator_question"
        else:
            phase3_intent = None  # keep patient/unsupported routing untouched
    patient_context = {
        "is_patient_value_question": context.get("is_patient_value_question"),
        "age": context.get("age"),
        "age_group": context.get("age_group"),
        "vital_sign": context.get("vital_sign"),
        "value": context.get("value"),
        "time_window": context.get("time_window"),
    }
    sources_used: list[str] = []
    answer = ""
    evidence_card: dict[str, Any] | None = None

    if question_type == "clinical_advice_refused":
        answer = safety.get("refusal_message", CLINICAL_WARNING) + f"\n\n{CLINICAL_WARNING}"

    elif question_type.startswith("phase3_"):
        answer, sources_used, evidence_card = _run_phase3(
            question, phase3_intent, client, trace, warnings)

    elif question_type == "patient_value_question":
        vital_sign = context["vital_sign"]
        age_group = context["age_group"]
        time_window = context["time_window"]
        value = float(context["value"])
        direction = _direction(question)

        availability = trace.record(
            "check_data_availability",
            {"vital_sign": vital_sign, "age_group": age_group, "time_window": time_window},
            lambda: client.call_tool("check_data_availability", {
                "vital_sign": vital_sign, "age_group": age_group, "time_window": time_window}),
        )

        if not availability.get("available"):
            question_type = "unsupported_or_missing_data"
            warnings.append(availability.get("message", "Exact summary unavailable."))
            answer = (
                f"The exact MIMIC-IV summary for {vital_sign} / {age_group} / {time_window} is not available, "
                "so no descriptive comparison can be made. No substitute vital sign or subgroup is used.\n\n"
                + CLINICAL_WARNING
            )
        else:
            summary = trace.record(
                "get_vital_summary",
                {"vital_sign": vital_sign, "age_group": age_group, "time_window": time_window},
                lambda: client.call_tool("get_vital_summary", {
                    "vital_sign": vital_sign, "age_group": age_group, "time_window": time_window}),
            )
            standard_comparison = trace.record(
                "compare_to_standard_threshold",
                {"vital_sign": vital_sign, "value": value,
                 "standard_low": summary.get("standard_low"), "standard_high": summary.get("standard_high"),
                 "unitname": summary.get("unitname", "")},
                lambda: client.call_tool("compare_to_standard_threshold", {
                    "vital_sign": vital_sign, "value": value,
                    "standard_low": summary.get("standard_low"), "standard_high": summary.get("standard_high"),
                    "unitname": summary.get("unitname", "")}),
            )
            percentile_comparison = trace.record(
                "compare_to_percentiles",
                {"value": value, "summary": summary, "direction": direction},
                lambda: client.call_tool("compare_to_percentiles", {
                    "value": value, "summary": summary, "direction": direction}),
            )
            rag_context = _safe_retrieve_context(trace, client, question, top_k=3)
            if rag_context is None:
                warnings.append("Supporting RAG retrieval was unavailable; answer relies on the exact summary only.")

            report = trace.record(
                "generate_patient_interpretation_report",
                {"question": question, "patient_context": patient_context, "summary": "<summary>",
                 "standard_comparison": "<standard>", "percentile_comparison": "<percentile>"},
                lambda: client.call_tool("generate_patient_interpretation_report", {
                    "question": question, "patient_context": patient_context, "summary": summary,
                    "standard_comparison": standard_comparison, "percentile_comparison": percentile_comparison,
                    "rag_context": rag_context}),
            )
            sources_used = report.get("sources_used", [])

            llm_answer, warning = _call_llm(_patient_prompt(question, context, report))
            if warning:
                warnings.append(warning)
                answer = (
                    f"{report.get('short_answer')}\n\n"
                    f"- Standard threshold: {report.get('standard_threshold_interpretation')}\n"
                    f"- MIMIC-IV percentiles: {report.get('mimic_percentile_interpretation')}\n"
                    f"- Limitation: {report.get('limitations')}\n\n{CLINICAL_WARNING}"
                )
            else:
                answer = llm_answer

    elif question_type == "calculator_question":
        calc = trace.record(
            "calculatrice_medicale",
            {"expression": calc_expression},
            lambda: client.call_tool("calculatrice_medicale", {"expression": calc_expression}),
        )
        if calc.get("status") == "ok":
            answer = (
                f"The result of {calc.get('expression')} is {calc.get('result')}. "
                "(Arithmetic helper tool; not a clinical calculation.)"
            )
        else:
            warnings.append(f"Could not evaluate the arithmetic expression '{calc_expression}'.")
            answer = (
                f"I could not evaluate the expression '{calc_expression}'. Please provide a simple "
                "arithmetic expression such as '104 * 2'."
            )

    elif question_type == "unsupported_or_missing_data":
        warnings.append(
            "Incomplete patient context; missing: " + ", ".join(missing_fields) + "."
        )
        answer = (
            "The question looks like a patient-value comparison but is missing required information ("
            + ", ".join(missing_fields)
            + "). Please provide it so an exact MIMIC-IV summary can be matched. No comparison is forced.\n\n"
            + CLINICAL_WARNING
        )

    else:  # concept_question / dataset_question / pipeline_question
        rag_context = _safe_retrieve_context(trace, client, question, top_k=top_k)
        extra = ""
        if question_type == "concept_question" and _is_threshold_concept(question):
            explanation = trace.record(
                "explain_threshold_type", {}, lambda: client.call_tool("explain_threshold_type", {})
            )
            extra = (
                f"Standard threshold: {explanation['standard_threshold']}\n"
                f"Adaptive percentile threshold: {explanation['adaptive_percentile_threshold']}\n"
                f"Project interpretation: {explanation['project_interpretation']}\n"
                f"Limitations: {explanation['limitations']}"
            )
        if rag_context is None:
            warnings.append("RAG retrieval was unavailable (is Ollama running and the index built?).")
            rag_context = {"chunks": [], "sources": []}
        sources_used = [src.get("source_file") for src in rag_context.get("sources", []) if src.get("source_file")]

        llm_answer, warning = _call_llm(_context_prompt(question, rag_context, extra))
        if warning:
            warnings.append(warning)
            if extra:
                answer = extra + f"\n\n{CLINICAL_WARNING}"
            elif rag_context.get("chunks"):
                joined = "\n\n".join(
                    f"- {c.get('source_file')}: {c.get('chunk_preview')}" for c in rag_context["chunks"][:top_k]
                )
                answer = (
                    "LLM unavailable; here are the most relevant retrieved sources:\n\n"
                    + joined
                    + f"\n\n{CLINICAL_WARNING}"
                )
            else:
                answer = "No answer could be produced (LLM and retrieval both unavailable).\n\n" + CLINICAL_WARNING
        else:
            answer = llm_answer

    # Purely arithmetic answers stay concise and skip the heavy clinical warning.
    if question_type != "calculator_question" and CLINICAL_WARNING not in answer:
        answer = f"{answer.rstrip()}\n\n{CLINICAL_WARNING}"

    result = {
        "mode": MODE,
        "question": question,
        "question_type": question_type,
        "answer": answer,
        "patient_context": patient_context,
        "evidence_card": evidence_card,
        "sources_used": sources_used,
        "tool_backend": client.name,
        "tool_trace": trace.as_list(),
        "tools_called": trace.tool_names(),
        "warnings": warnings,
        "tool_call_success_rate": trace.success_rate(),
        "average_tool_latency_ms": trace.average_latency_ms(),
    }

    if persist:
        try:
            path = save_trace(result, datetime.now())
            result["trace_file"] = str(path)
        except Exception as exc:  # noqa: BLE001 - persistence is best-effort
            LOGGER.warning("Failed to persist agent trace: %s", exc)

    return result
