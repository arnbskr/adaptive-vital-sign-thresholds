"""Phase 2 agent evaluation.

Runs the single agent over a fixed set of scenarios and reports auditable
metrics: tool-call success, exact context extraction, data-availability
checking, grounded-answer presence, non-clinical-warning presence, and average
tool latency. Results are written to data/evaluation/agent_evaluation.csv and
data/evaluation/agent_summary.md.

    python -m src.evaluate_agent
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .agent import CLINICAL_WARNING, run_agent
from .config import EVALUATION_DIR, ensure_data_directories

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

SCENARIOS: list[dict[str, Any]] = [
    {
        "question": "For a patient aged 82 with mean HR 104 bpm in the first 24h ICU stay, is this value high?",
        "expected_type": "patient_value_question",
        "expected_vital_sign": "Heart Rate",
        "expected_age_group": "75-84",
        "expected_time_window": "first_24h",
        "expected_tools": ["check_data_availability", "get_vital_summary", "compare_to_standard_threshold", "compare_to_percentiles"],
    },
    {
        "question": "For a patient aged 78 with MAP 62 mmHg in the first 24h ICU stay, is this value low?",
        "expected_type": "patient_value_question",
        "expected_vital_sign": "MAP",
        "expected_age_group": "75-84",
        "expected_time_window": "first_24h",
        "expected_tools": ["check_data_availability", "get_vital_summary", "compare_to_standard_threshold", "compare_to_percentiles"],
    },
    {
        "question": "For a patient aged 80 with SpO2 90% in the first 24h ICU stay, is this low?",
        "expected_type": "patient_value_question",
        "expected_vital_sign": "SpO2",
        "expected_age_group": "75-84",
        "expected_time_window": "first_24h",
        "expected_tools": ["check_data_availability"],
    },
    {
        "question": "What is the difference between a standard clinical threshold and an adaptive percentile-based threshold?",
        "expected_type": "concept_question",
        "expected_vital_sign": "",
        "expected_age_group": "",
        "expected_time_window": "",
        "expected_tools": ["retrieve_project_context", "explain_threshold_type"],
    },
    {
        "question": "Which MIMIC-IV tables are used to derive ICU vital-sign summaries?",
        "expected_type": "dataset_question",
        "expected_vital_sign": "",
        "expected_age_group": "",
        "expected_time_window": "",
        "expected_tools": ["retrieve_project_context"],
    },
]


def _context_ok(scenario: dict[str, Any], result: dict[str, Any]) -> bool:
    if scenario["expected_type"] == "patient_value_question":
        ctx = result.get("patient_context", {})
        return (
            str(ctx.get("vital_sign")) == scenario["expected_vital_sign"]
            and str(ctx.get("age_group")) == scenario["expected_age_group"]
            and str(ctx.get("time_window")) == scenario["expected_time_window"]
        )
    return result.get("question_type") == scenario["expected_type"]


def run_agent_evaluation() -> tuple[str, str]:
    ensure_data_directories()
    rows: list[dict[str, Any]] = []

    for scenario in SCENARIOS:
        LOGGER.info("Evaluating: %s", scenario["question"])
        result = run_agent(scenario["question"], persist=True)
        tools_called = result.get("tools_called", [])
        answer = result.get("answer", "")

        expected_tools_called = all(tool in tools_called for tool in scenario["expected_tools"])
        rows.append(
            {
                "question": scenario["question"],
                "expected_type": scenario["expected_type"],
                "question_type": result.get("question_type"),
                "type_match": result.get("question_type") == scenario["expected_type"],
                "tools_called": "|".join(tools_called),
                "expected_tools_called": expected_tools_called,
                "tool_call_success_rate": result.get("tool_call_success_rate", 0.0),
                "exact_context_extraction": _context_ok(scenario, result),
                "data_availability_checked": "check_data_availability" in tools_called,
                "grounded_answer_present": bool(answer.strip()),
                "non_clinical_warning_present": CLINICAL_WARNING in answer,
                "average_tool_latency_ms": result.get("average_tool_latency_ms", 0.0),
                "warnings": " | ".join(result.get("warnings", [])),
            }
        )

    frame = pd.DataFrame(rows)
    csv_path = EVALUATION_DIR / "agent_evaluation.csv"
    frame.to_csv(csv_path, index=False)

    patient_rows = frame[frame["expected_type"] == "patient_value_question"]
    summary = {
        "scenarios": len(frame),
        "tool_call_success_rate": round(frame["tool_call_success_rate"].mean(), 4),
        "exact_context_extraction_rate": round(frame["exact_context_extraction"].mean(), 4),
        "data_availability_checked_rate": round(
            patient_rows["data_availability_checked"].mean() if not patient_rows.empty else 1.0, 4
        ),
        "grounded_answer_rate": round(frame["grounded_answer_present"].mean(), 4),
        "non_clinical_warning_rate": round(frame["non_clinical_warning_present"].mean(), 4),
        "expected_tools_called_rate": round(frame["expected_tools_called"].mean(), 4),
        "average_tool_latency_ms": round(frame["average_tool_latency_ms"].mean(), 2),
    }

    md_lines = [
        "# Phase 2 Agent Evaluation",
        "",
        "Single-agent orchestration over deterministic MCP tools. Descriptive, non-clinical.",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Scenarios | {summary['scenarios']} |",
        f"| Tool-call success rate | {summary['tool_call_success_rate']} |",
        f"| Exact context-extraction rate | {summary['exact_context_extraction_rate']} |",
        f"| Data-availability checked (patient questions) | {summary['data_availability_checked_rate']} |",
        f"| Grounded answer present | {summary['grounded_answer_rate']} |",
        f"| Non-clinical warning present | {summary['non_clinical_warning_rate']} |",
        f"| Expected tools called | {summary['expected_tools_called_rate']} |",
        f"| Average tool latency (ms) | {summary['average_tool_latency_ms']} |",
        "",
        "## Per-scenario",
        "",
        "| Question | Type (got/expected) | Tools called | Success | Context OK | Warning |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['question'][:60]}… | {row['question_type']}/{row['expected_type']} | "
            f"{row['tools_called']} | {row['tool_call_success_rate']} | "
            f"{row['exact_context_extraction']} | {row['non_clinical_warning_present']} |"
        )

    md_path = EVALUATION_DIR / "agent_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    LOGGER.info("Saved agent evaluation to %s and %s", csv_path, md_path)
    return str(csv_path), str(md_path)


def main() -> None:
    run_agent_evaluation()


if __name__ == "__main__":
    main()
