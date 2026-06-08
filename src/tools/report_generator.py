"""Tool: generate_patient_interpretation_report.

Deterministically assembles a structured, non-clinical interpretation from the
outputs of the other tools. It performs NO calculation and NO LLM call: it only
arranges already-computed deterministic results into an auditable structure.
The agent later reformulates this skeleton into prose with the LLM.
"""

from __future__ import annotations

from typing import Any

CLINICAL_WARNING = (
    "This response is for academic interpretation only and is not a clinical diagnosis "
    "or treatment recommendation."
)

LIMITATIONS = (
    "Interpretations are descriptive summaries of historical MIMIC-IV data for elderly ICU "
    "patients. They depend on the completeness of the matched summary, compare against a single "
    "matching subgroup only, and must not be used for diagnosis, triage, or treatment."
)

_DIRECTION_WORD = {
    "above_standard_high": "above the standard high threshold",
    "below_standard_low": "below the standard low threshold",
    "within_reference_range": "within the standard reference range",
    "borderline": "on the standard reference threshold",
    "not_applicable": "without an applicable standard threshold",
}


def _format_patient_context(patient_context: dict[str, Any]) -> str:
    age = patient_context.get("age")
    vital_sign = patient_context.get("vital_sign", "")
    value = patient_context.get("value")
    age_group = patient_context.get("age_group", "")
    time_window = patient_context.get("time_window", "")
    parts = []
    if age is not None:
        parts.append(f"age {age} (group {age_group})" if age_group else f"age {age}")
    if vital_sign:
        parts.append(f"{vital_sign} = {value}" if value is not None else vital_sign)
    if time_window:
        parts.append(f"ICU window {time_window}")
    return ", ".join(parts) if parts else "incomplete patient context"


def _collect_sources(summary: dict[str, Any], rag_context: dict[str, Any] | None) -> list[str]:
    sources: list[str] = []
    summary_source = summary.get("source_file")
    if summary_source:
        sources.append(str(summary_source))
    if rag_context:
        for source in rag_context.get("sources", []):
            label = source.get("source_file") if isinstance(source, dict) else str(source)
            if label and label not in sources:
                sources.append(label)
    return sources


def generate_patient_interpretation_report(
    question: str,
    patient_context: dict[str, Any],
    summary: dict[str, Any],
    standard_comparison: dict[str, Any],
    percentile_comparison: dict[str, Any],
    rag_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the structured, non-clinical interpretation skeleton."""

    vital_sign = patient_context.get("vital_sign") or summary.get("vital_sign", "the vital sign")
    unit = summary.get("unitname", "")
    value = patient_context.get("value")
    value_text = f"{value:g} {unit}".strip() if isinstance(value, (int, float)) else str(value)

    standard_status = str(standard_comparison.get("status", "not_applicable"))
    percentile_position = str(percentile_comparison.get("percentile_position", "unavailable"))

    standard_phrase = _DIRECTION_WORD.get(standard_status, standard_status.replace("_", " "))
    short_answer = (
        f"For this patient, {vital_sign} of {value_text} is {standard_phrase}, and within the "
        f"matching MIMIC-IV subgroup distribution it sits at the '{percentile_position}' position. "
        "This is a descriptive comparison only."
    )

    return {
        "short_answer": short_answer,
        "patient_context": _format_patient_context(patient_context),
        "standard_threshold_interpretation": standard_comparison.get("explanation", ""),
        "mimic_percentile_interpretation": percentile_comparison.get("explanation", ""),
        "sources_used": _collect_sources(summary, rag_context),
        "limitations": LIMITATIONS,
        "clinical_warning": CLINICAL_WARNING,
    }
