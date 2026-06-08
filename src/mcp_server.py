"""Local MCP layer that exposes the Phase 2 deterministic tools.

This module is the single boundary through which the agent invokes tools. It
provides:

* ``TOOL_SPECS`` / ``list_tools()`` -- an MCP-style catalogue (name, description,
  JSON-schema-like input spec) for the seven deterministic tools.
* ``call_tool(name, arguments)`` -- a deterministic dispatcher used by the agent.
* ``main()`` -- runs a real MCP stdio server when the official ``mcp`` SDK is
  installed; otherwise it prints the catalogue and explains that the server is
  running in MCP-compatible local mode.

The same registry backs both paths, so the agent behaves identically whether or
not the heavyweight MCP runtime is present. No multi-agent logic lives here.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .tools import (
    calculatrice_medicale,
    check_data_availability,
    compare_to_percentiles,
    compare_to_standard_threshold,
    explain_threshold_type,
    generate_patient_interpretation_report,
    get_vital_summary,
    retrieve_project_context,
)


class ToolSpec:
    """An MCP-style tool description bound to its deterministic implementation."""

    def __init__(self, name: str, description: str, parameters: dict[str, str], func: Callable[..., Any]) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters  # param name -> short type/description
        self.func = func

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        "check_data_availability",
        "Check whether an exact MIMIC-IV statistical summary exists for a vital sign, age group, and time window.",
        {"vital_sign": "str", "age_group": "str", "time_window": "str"},
        check_data_availability,
    ),
    ToolSpec(
        "get_vital_summary",
        "Retrieve the exact MIMIC-IV statistical summary (count, mean, median, percentiles, thresholds).",
        {"vital_sign": "str", "age_group": "str", "time_window": "str"},
        get_vital_summary,
    ),
    ToolSpec(
        "compare_to_standard_threshold",
        "Compare a value to the predefined standard reference thresholds (descriptive only).",
        {"vital_sign": "str", "value": "float", "standard_low": "float|null", "standard_high": "float|null", "unitname": "str (optional)"},
        compare_to_standard_threshold,
    ),
    ToolSpec(
        "compare_to_percentiles",
        "Locate a value within the MIMIC-IV percentile distribution of an exact summary.",
        {"value": "float", "summary": "dict", "direction": "str|null (high|low|neutral)"},
        compare_to_percentiles,
    ),
    ToolSpec(
        "retrieve_project_context",
        "Retrieve documentary context from the Phase 1 semantic RAG index.",
        {"query": "str", "top_k": "int (optional)"},
        retrieve_project_context,
    ),
    ToolSpec(
        "explain_threshold_type",
        "Explain the difference between standard and adaptive percentile-based thresholds.",
        {},
        explain_threshold_type,
    ),
    ToolSpec(
        "generate_patient_interpretation_report",
        "Assemble a structured, non-clinical interpretation from the other tools' outputs.",
        {
            "question": "str",
            "patient_context": "dict",
            "summary": "dict",
            "standard_comparison": "dict",
            "percentile_comparison": "dict",
            "rag_context": "dict|null (optional)",
        },
        generate_patient_interpretation_report,
    ),
    ToolSpec(
        "calculatrice_medicale",
        "Demonstration-only safe calculator: evaluate a simple arithmetic expression "
        "(numbers and + - * / // % ** and parentheses) via a strict AST, never eval(). "
        "Use this for purely arithmetic questions instead of the RAG/MIMIC tools.",
        {"expression": "str"},
        calculatrice_medicale,
    ),
]

TOOL_REGISTRY: dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_SPECS}


def list_tools() -> list[dict[str, Any]]:
    """Return the MCP-style tool catalogue."""

    return [spec.to_dict() for spec in TOOL_SPECS]


def call_tool(name: str, arguments: dict[str, Any]) -> Any:
    """Dispatch a tool call by name with keyword arguments (the MCP boundary)."""

    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        raise KeyError(f"Unknown tool: {name}. Available tools: {', '.join(TOOL_REGISTRY)}")
    return spec.func(**arguments)


def _run_real_mcp_server() -> bool:
    """Start an official MCP stdio server if the SDK is installed. Returns True if it ran."""

    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore
    except Exception:  # noqa: BLE001 - SDK not installed
        return False

    server = FastMCP("icu-trajectory-tools")
    for spec in TOOL_SPECS:
        # FastMCP introspects the wrapped function's signature and docstring.
        server.add_tool(spec.func, name=spec.name, description=spec.description)
    server.run()
    return True


def main() -> None:
    if _run_real_mcp_server():
        return
    print("MCP-compatible local tool server (official `mcp` SDK not installed).")
    print("Running in MCP-compatible mode: tools are callable via src.mcp_server.call_tool(name, arguments).")
    print("Install `mcp` (see requirements.txt) to expose these over a real MCP stdio transport.\n")
    print("Available tools:")
    print(json.dumps(list_tools(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
