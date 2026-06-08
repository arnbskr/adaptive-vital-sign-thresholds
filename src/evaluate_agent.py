"""Phase 2 agent evaluation.

Runs the single agent over a fixed set of scenarios and reports auditable
metrics: tool-call success, exact context extraction, data-availability
checking, grounded-answer presence, non-clinical-warning presence, and average
tool latency. Results are written to data/evaluation/agent_evaluation.csv and
data/evaluation/agent_summary.md.

    python -m src.evaluate_agent --backend local
    python -m src.evaluate_agent --backend mcp_remote
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import pandas as pd

from .agent import CLINICAL_WARNING, run_agent
from .config import EVALUATION_DIR, ensure_data_directories
from .tool_client import ToolBackendError, get_tool_client, is_remote_backend

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_PATIENT_TOOLS = ["check_data_availability", "get_vital_summary", "compare_to_standard_threshold", "compare_to_percentiles"]
_MIMIC_RAG_TOOLS = ["retrieve_project_context", "get_vital_summary", "compare_to_percentiles", "compare_to_standard_threshold"]

SCENARIOS: list[dict[str, Any]] = [
    {
        "question": "For a patient aged 82 with mean HR 104 bpm in the first 24h ICU stay, is this value high?",
        "expected_type": "patient_value_question",
        "expected_vital_sign": "Heart Rate",
        "expected_age_group": "75-84",
        "expected_time_window": "first_24h",
        "expected_tools": _PATIENT_TOOLS,
        "forbidden_tools": ["calculatrice_medicale"],
        "expect_clinical_warning": True,
    },
    {
        "question": "What is the difference between a standard clinical threshold and an adaptive percentile-based threshold?",
        "expected_type": "concept_question",
        "expected_vital_sign": "",
        "expected_age_group": "",
        "expected_time_window": "",
        "expected_tools": ["retrieve_project_context", "explain_threshold_type"],
        "forbidden_tools": ["calculatrice_medicale", "get_vital_summary"],
        "expect_clinical_warning": True,
    },
    {
        "question": "What is 104 * 2?",
        "expected_type": "calculator_question",
        "expected_vital_sign": "",
        "expected_age_group": "",
        "expected_time_window": "",
        "expected_tools": ["calculatrice_medicale"],
        "forbidden_tools": _MIMIC_RAG_TOOLS,
        "expected_result": 208,
        "expect_clinical_warning": False,
    },
    {
        "question": "For a patient aged 50 with mean HR 104 bpm in the first 24h ICU stay, is this value high?",
        "expected_type": "unsupported_or_missing_data",
        "expected_vital_sign": "",
        "expected_age_group": "",
        "expected_time_window": "",
        "expected_tools": [],
        "forbidden_tools": ["calculatrice_medicale"],
        "expect_clinical_warning": True,
    },
    {
        "question": "Which MIMIC-IV tables are used to derive ICU vital-sign summaries?",
        "expected_type": "dataset_question",
        "expected_vital_sign": "",
        "expected_age_group": "",
        "expected_time_window": "",
        "expected_tools": ["retrieve_project_context"],
        "forbidden_tools": ["calculatrice_medicale"],
        "expect_clinical_warning": True,
    },
]


def _calculator_result_correct(scenario: dict[str, Any], result: dict[str, Any]) -> bool | str:
    """For a calculator scenario, did the tool return the expected numeric result?"""

    if "expected_result" not in scenario:
        return "n/a"
    for entry in result.get("tool_trace", []):
        if entry.get("tool_name") == "calculatrice_medicale":
            output = entry.get("outputs", {}) or {}
            return output.get("status") == "ok" and output.get("result") == scenario["expected_result"]
    return False


def _context_ok(scenario: dict[str, Any], result: dict[str, Any]) -> bool:
    if scenario["expected_type"] == "patient_value_question":
        ctx = result.get("patient_context", {})
        return (
            str(ctx.get("vital_sign")) == scenario["expected_vital_sign"]
            and str(ctx.get("age_group")) == scenario["expected_age_group"]
            and str(ctx.get("time_window")) == scenario["expected_time_window"]
        )
    return result.get("question_type") == scenario["expected_type"]


def _check_remote_backend() -> tuple[bool, bool, int]:
    """Return (reachable, discovery_ok, tool_count) for the remote MCP backend."""

    try:
        client = get_tool_client("mcp_remote")
    except ToolBackendError:
        return False, False, 0
    try:
        tools = client.list_tools()
        return True, len(tools) > 0, len(tools)
    finally:
        client.close()


def run_agent_evaluation(tool_backend: str = "local") -> tuple[str, str] | None:
    ensure_data_directories()

    remote = is_remote_backend(tool_backend)
    remote_reachable, discovery_ok, tool_count = (True, True, 0)
    if remote:
        remote_reachable, discovery_ok, tool_count = _check_remote_backend()
        if not remote_reachable:
            LOGGER.warning(
                "Remote MCP backend is not reachable at the configured URL. "
                "Start it with `python src/server_mcp.py`, then re-run with "
                "`--backend mcp_remote`. Skipping the remote evaluation."
            )
            return None
        LOGGER.info("Remote MCP backend reachable; discovered %s tools.", tool_count)

    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        LOGGER.info("[%s] Evaluating: %s", tool_backend, scenario["question"])
        # For an explicit remote run we disable fallback so a real remote failure
        # is visible rather than silently masked by the local backend.
        result = run_agent(
            scenario["question"],
            persist=True,
            tool_backend=tool_backend,
            allow_fallback=not remote,
        )
        tools_called = result.get("tools_called", [])
        answer = result.get("answer", "")
        result_warnings = result.get("warnings", [])

        expected_tool_called = all(tool in tools_called for tool in scenario["expected_tools"])
        forbidden = scenario.get("forbidden_tools", [])
        forbidden_tools_not_called = not any(tool in tools_called for tool in forbidden)
        warning_present = CLINICAL_WARNING in answer
        fallback_used = any("fell back to the local" in w for w in result_warnings)
        rows.append(
            {
                "question": scenario["question"],
                "expected_type": scenario["expected_type"],
                "question_type": result.get("question_type"),
                "type_match": result.get("question_type") == scenario["expected_type"],
                "backend_used": result.get("tool_backend"),
                "fallback_used": fallback_used,
                "remote_server_reachable": remote_reachable if remote else "n/a",
                "tool_discovery_success": discovery_ok if remote else "n/a",
                "tools_called": "|".join(tools_called),
                "expected_tool_called": expected_tool_called,
                "forbidden_tools_not_called": forbidden_tools_not_called,
                "calculator_result_correct": _calculator_result_correct(scenario, result),
                "tool_call_success_rate": result.get("tool_call_success_rate", 0.0),
                "exact_context_extraction": _context_ok(scenario, result),
                "data_availability_checked": "check_data_availability" in tools_called,
                "grounded_answer_present": bool(answer.strip()),
                "non_clinical_warning_present": warning_present,
                "non_clinical_warning_when_needed": warning_present == scenario.get("expect_clinical_warning", True),
                "average_tool_latency_ms": result.get("average_tool_latency_ms", 0.0),
                "warnings": " | ".join(result_warnings),
            }
        )

    frame = pd.DataFrame(rows)
    suffix = "_mcp_remote" if remote else ""
    csv_path = EVALUATION_DIR / f"agent_evaluation{suffix}.csv"
    frame.to_csv(csv_path, index=False)

    calc_rows = frame[frame["calculator_result_correct"] != "n/a"]
    summary = {
        "scenarios": len(frame),
        "backend_used": frame["backend_used"].iloc[0] if not frame.empty else tool_backend,
        "fallback_used_rate": round(frame["fallback_used"].mean(), 4),
        "tool_call_success_rate": round(frame["tool_call_success_rate"].mean(), 4),
        "exact_context_extraction_rate": round(frame["exact_context_extraction"].mean(), 4),
        "grounded_answer_rate": round(frame["grounded_answer_present"].mean(), 4),
        "non_clinical_warning_when_needed_rate": round(frame["non_clinical_warning_when_needed"].mean(), 4),
        "expected_tool_called_rate": round(frame["expected_tool_called"].mean(), 4),
        "forbidden_tools_not_called_rate": round(frame["forbidden_tools_not_called"].mean(), 4),
        "calculator_result_correct": (
            bool(calc_rows["calculator_result_correct"].all()) if not calc_rows.empty else "n/a"
        ),
        "average_tool_latency_ms": round(frame["average_tool_latency_ms"].mean(), 2),
    }

    md_lines = [
        "# Phase 2 Agent Evaluation",
        "",
        "Single-agent orchestration over deterministic MCP tools. Descriptive, non-clinical.",
        "",
        f"Tool backend: **{tool_backend}**"
        + (f" (remote reachable: {remote_reachable}, discovered tools: {tool_count})" if remote else "")
        + ".",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Tool backend | {tool_backend} |",
        f"| Backend actually used | {summary['backend_used']} |",
        f"| Fallback-to-local rate | {summary['fallback_used_rate']} |",
        f"| Scenarios | {summary['scenarios']} |",
        f"| Tool-call success rate | {summary['tool_call_success_rate']} |",
        f"| Exact context-extraction rate | {summary['exact_context_extraction_rate']} |",
        f"| Grounded answer present | {summary['grounded_answer_rate']} |",
        f"| Non-clinical warning when needed | {summary['non_clinical_warning_when_needed_rate']} |",
        f"| Expected tool called | {summary['expected_tool_called_rate']} |",
        f"| Forbidden tools not called | {summary['forbidden_tools_not_called_rate']} |",
        f"| Calculator result correct | {summary['calculator_result_correct']} |",
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

    md_path = EVALUATION_DIR / f"agent_summary{suffix}.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    LOGGER.info("Saved agent evaluation to %s and %s", csv_path, md_path)
    return str(csv_path), str(md_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 agent evaluation.")
    parser.add_argument(
        "--backend",
        default="local",
        help="Tool backend: 'local' (default) or 'mcp_remote'.",
    )
    args = parser.parse_args()
    result = run_agent_evaluation(tool_backend=args.backend)
    if result is None:
        LOGGER.warning("Evaluation skipped (remote backend unavailable).")


if __name__ == "__main__":
    main()
