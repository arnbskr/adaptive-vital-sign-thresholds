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
from datetime import datetime
from typing import Any

from .mcp_server import call_tool
from .semantic_rag import LLM_MODEL, build_ollama_client, infer_patient_context
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


def _safe_retrieve_context(trace: ToolTrace, query: str, top_k: int) -> dict[str, Any] | None:
    """Call the RAG tool through MCP, tolerating retrieval failures (e.g. Ollama down)."""

    try:
        return trace.record(
            "retrieve_project_context",
            {"query": query, "top_k": top_k},
            lambda: call_tool("retrieve_project_context", {"query": query, "top_k": top_k}),
        )
    except Exception as exc:  # noqa: BLE001 - trace already recorded the error
        LOGGER.warning("retrieve_project_context failed: %s", exc)
        return None


def run_agent(question: str, top_k: int = 5, persist: bool = True) -> dict[str, Any]:
    """Orchestrate tools + LLM for one question and return an auditable result."""

    trace = ToolTrace()
    warnings: list[str] = []
    context = infer_patient_context(question)
    question_type, missing_fields = _classify(question, context)
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

    if question_type == "patient_value_question":
        vital_sign = context["vital_sign"]
        age_group = context["age_group"]
        time_window = context["time_window"]
        value = float(context["value"])
        direction = _direction(question)

        availability = trace.record(
            "check_data_availability",
            {"vital_sign": vital_sign, "age_group": age_group, "time_window": time_window},
            lambda: call_tool("check_data_availability", {
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
                lambda: call_tool("get_vital_summary", {
                    "vital_sign": vital_sign, "age_group": age_group, "time_window": time_window}),
            )
            standard_comparison = trace.record(
                "compare_to_standard_threshold",
                {"vital_sign": vital_sign, "value": value,
                 "standard_low": summary.get("standard_low"), "standard_high": summary.get("standard_high"),
                 "unitname": summary.get("unitname", "")},
                lambda: call_tool("compare_to_standard_threshold", {
                    "vital_sign": vital_sign, "value": value,
                    "standard_low": summary.get("standard_low"), "standard_high": summary.get("standard_high"),
                    "unitname": summary.get("unitname", "")}),
            )
            percentile_comparison = trace.record(
                "compare_to_percentiles",
                {"value": value, "summary": summary, "direction": direction},
                lambda: call_tool("compare_to_percentiles", {
                    "value": value, "summary": summary, "direction": direction}),
            )
            rag_context = _safe_retrieve_context(trace, question, top_k=3)
            if rag_context is None:
                warnings.append("Supporting RAG retrieval was unavailable; answer relies on the exact summary only.")

            report = trace.record(
                "generate_patient_interpretation_report",
                {"question": question, "patient_context": patient_context, "summary": "<summary>",
                 "standard_comparison": "<standard>", "percentile_comparison": "<percentile>"},
                lambda: call_tool("generate_patient_interpretation_report", {
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
        rag_context = _safe_retrieve_context(trace, question, top_k=top_k)
        extra = ""
        if question_type == "concept_question" and _is_threshold_concept(question):
            explanation = trace.record(
                "explain_threshold_type", {}, lambda: call_tool("explain_threshold_type", {})
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

    if CLINICAL_WARNING not in answer:
        answer = f"{answer.rstrip()}\n\n{CLINICAL_WARNING}"

    result = {
        "mode": MODE,
        "question": question,
        "question_type": question_type,
        "answer": answer,
        "patient_context": patient_context,
        "sources_used": sources_used,
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
