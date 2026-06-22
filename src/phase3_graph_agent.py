"""Phase 3 workflow expressed as a LangGraph StateGraph (role-based nodes).

This is the **same** single-agent Phase 3 logic as ``src/agent.py``, re-expressed
as an explicit state graph so the workflow is the project's architecture, not a
side demo. Each node is a **specialized role** (safety, intent, data, evidence,
answer, grounding), not an autonomous LLM — it is a role-based decomposition for
auditability and separation of responsibilities, NOT a multi-agent debate.

Graph (note: ``answer`` runs before ``grounding`` because grounding validates the
numbers in the produced answer):

    START
      → safety_agent     -- detect_clinical_advice_request; refuse + stop if clinical
      → (conditional)    -- refused → END ; otherwise → intent
      → intent_agent     -- reuse agent._infer_phase3_intent (+ session memory fill)
      → data_agent       -- call the right deterministic tool via ToolClient + trace
      → evidence_agent   -- generate_evidence_card when relevant
      → answer_agent     -- LLM reformulation (local), deterministic fallback
      → grounding_agent  -- validate_numeric_grounding over the tool trace
      → END

It degrades gracefully: if ``langgraph`` is not installed, ``run_phase3_graph_agent``
falls back to the classic ``run_agent`` with a clear warning (never crashes).
Non-clinical, deterministic tools, auditable trace — unchanged.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from typing import Any, TypedDict

from .agent import (
    CLINICAL_WARNING,
    _call_llm,
    _infer_phase3_intent,
    _phase3_deterministic_answer,
    _phase3_prompt,
    _resolve_tool_client,
    run_agent,
)
from .grounding_validator import validate_numeric_grounding
from .tool_trace import ToolTrace, save_trace

LOGGER = logging.getLogger(__name__)

# Roles, in execution order — shown in logs / Streamlit.
NODE_ROLES = ["safety_agent", "intent_agent", "data_agent", "evidence_agent", "answer_agent", "grounding_agent"]


def is_langgraph_available() -> bool:
    try:
        import langgraph.graph  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - SDK not installed
        return False


class Phase3GraphState(TypedDict, total=False):
    question: str
    intent: dict
    safety_result: dict
    tool_payload: dict
    evidence_card: dict
    grounding_validation: dict
    answer: str
    warnings: list
    tool_trace: list
    sources_used: list
    memory_context: dict
    question_type: str
    nodes_executed: list


def _merge_intent_with_memory(intent: dict[str, Any], memory: dict[str, Any] | None) -> dict[str, Any]:
    """Fill missing intent slots (age_group/time_window/metric) from session memory."""

    if not memory:
        return intent
    merged = dict(intent)
    for field in ("age_group", "time_window", "metric"):
        if not merged.get(field) and memory.get(field):
            merged[field] = memory[field]
    return merged


# --------------------------------------------------------------------------- #
# Node builders (closures over the shared ToolClient + ToolTrace).
# --------------------------------------------------------------------------- #
def _build_nodes(client, trace: ToolTrace, memory_context: dict[str, Any] | None):

    def _record(name: str, args: dict[str, Any]) -> dict[str, Any]:
        return trace.record(name, args, lambda: client.call_tool(name, args))

    def safety_agent(state: Phase3GraphState) -> dict[str, Any]:
        question = state["question"]
        safety = _record("detect_clinical_advice_request", {"question": question})
        executed = state.get("nodes_executed", []) + ["safety_agent"]
        if isinstance(safety, dict) and safety.get("is_clinical_advice"):
            answer = safety.get("refusal_message", CLINICAL_WARNING) + f"\n\n{CLINICAL_WARNING}"
            return {"safety_result": safety, "question_type": "clinical_advice_refused",
                    "answer": answer, "nodes_executed": executed}
        return {"safety_result": safety, "nodes_executed": executed}

    def intent_agent(state: Phase3GraphState) -> dict[str, Any]:
        intent = _infer_phase3_intent(state["question"]) or {"intent": "cohort_statistics"}
        intent = _merge_intent_with_memory(intent, memory_context)
        return {"intent": intent, "question_type": "phase3_" + intent["intent"],
                "nodes_executed": state.get("nodes_executed", []) + ["intent_agent"]}

    def data_agent(state: Phase3GraphState) -> dict[str, Any]:
        intent = state["intent"]
        kind = intent["intent"]
        v, ag, tw, metric = (intent.get("variable_name"), intent.get("age_group"),
                             intent.get("time_window"), intent.get("metric"))
        if kind == "available_variables":
            payload = _record("list_available_variables", {"variable_category": intent.get("variable_category")})
        elif kind == "compare_age_groups":
            payload = _record("compare_age_groups", {"variable_name": v, "time_window": tw, "metric": metric})
        elif kind == "compare_time_windows":
            payload = _record("compare_time_windows", {"variable_name": v, "age_group": ag, "metric": metric})
        elif kind == "variable_summary":
            payload = _record("get_variable_summary", {"variable_name": v, "age_group": ag, "time_window": tw})
        else:
            payload = _record("query_cohort_statistics",
                              {"variable_name": v, "age_group": ag, "time_window": tw, "metric": metric})
        sources = [payload["source"]] if isinstance(payload, dict) and payload.get("source") else []
        warnings = state.get("warnings", [])
        if isinstance(payload, dict) and payload.get("error"):
            warnings = warnings + [payload["error"]]
        return {"tool_payload": payload, "sources_used": sources, "warnings": warnings,
                "nodes_executed": state.get("nodes_executed", []) + ["data_agent"]}

    def evidence_agent(state: Phase3GraphState) -> dict[str, Any]:
        intent = state["intent"]
        kind = intent["intent"]
        executed = state.get("nodes_executed", []) + ["evidence_agent"]
        if kind == "available_variables":
            return {"nodes_executed": executed}  # catalogue needs no per-variable card
        card = _record("generate_evidence_card", {
            "variable_name": intent.get("variable_name"), "age_group": intent.get("age_group"),
            "time_window": intent.get("time_window"), "tool_name": kind, "main_metric": intent.get("metric")})
        return {"evidence_card": card, "nodes_executed": executed}

    def answer_agent(state: Phase3GraphState) -> dict[str, Any]:
        kind = state["intent"]["intent"]
        payload = state.get("tool_payload", {})
        card = state.get("evidence_card")
        warnings = state.get("warnings", [])
        llm_answer, warning = _call_llm(_phase3_prompt(state["question"], kind, payload, card))
        if warning:
            warnings = warnings + [warning]
            answer = _phase3_deterministic_answer(kind, payload, card)
        else:
            answer = llm_answer
        if CLINICAL_WARNING not in answer:
            answer = f"{answer.rstrip()}\n\n{CLINICAL_WARNING}"
        return {"answer": answer, "warnings": warnings,
                "nodes_executed": state.get("nodes_executed", []) + ["answer_agent"]}

    def grounding_agent(state: Phase3GraphState) -> dict[str, Any]:
        grounding = validate_numeric_grounding(state.get("answer", ""), trace.as_list())
        warnings = state.get("warnings", [])
        if grounding.get("warning"):
            warnings = warnings + [grounding["warning"]]
        return {"grounding_validation": grounding, "warnings": warnings,
                "nodes_executed": state.get("nodes_executed", []) + ["grounding_agent"]}

    return safety_agent, intent_agent, data_agent, evidence_agent, answer_agent, grounding_agent


def _build_graph(client, trace: ToolTrace, memory_context: dict[str, Any] | None):
    from langgraph.graph import END, START, StateGraph

    safety, intent, data, evidence, answer, grounding = _build_nodes(client, trace, memory_context)
    graph = StateGraph(Phase3GraphState)
    graph.add_node("safety_agent", safety)
    graph.add_node("intent_agent", intent)
    graph.add_node("data_agent", data)
    graph.add_node("evidence_agent", evidence)
    graph.add_node("answer_agent", answer)
    graph.add_node("grounding_agent", grounding)

    graph.add_edge(START, "safety_agent")
    graph.add_conditional_edges(
        "safety_agent",
        lambda s: "refused" if s.get("question_type") == "clinical_advice_refused" else "continue",
        {"refused": END, "continue": "intent_agent"},
    )
    graph.add_edge("intent_agent", "data_agent")
    graph.add_edge("data_agent", "evidence_agent")
    graph.add_edge("evidence_agent", "answer_agent")
    graph.add_edge("answer_agent", "grounding_agent")
    graph.add_edge("grounding_agent", END)
    return graph.compile()


def run_phase3_graph_agent(
    question: str,
    tool_backend: str = "local",
    memory_context: dict[str, Any] | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Run the Phase 3 workflow as a LangGraph state graph.

    Returns the same dict shape as ``run_agent`` (plus ``engine`` and
    ``nodes_executed``). Falls back to the classic agent if langgraph is missing.
    """

    if not is_langgraph_available():
        LOGGER.warning("langgraph not installed; falling back to the classic agent.")
        result = run_agent(question, tool_backend=tool_backend, persist=persist)
        result["engine"] = "classic (langgraph unavailable)"
        result["nodes_executed"] = []
        result["warnings"] = result.get("warnings", []) + [
            "LangGraph is not installed; used the classic agent. Install with: pip install langgraph"
        ]
        return result

    # Non-Phase-3, non-clinical questions are delegated to the classic agent so
    # this engine stays focused on the Phase 3 multi-variable workflow.
    intent = _infer_phase3_intent(question)
    client, owns_client, warnings = _resolve_tool_client(tool_backend, None, True)
    try:
        trace = ToolTrace(backend=client.name)
        if intent is None:
            probe = trace.record("detect_clinical_advice_request", {"question": question},
                                 lambda: client.call_tool("detect_clinical_advice_request", {"question": question}))
            if not (isinstance(probe, dict) and probe.get("is_clinical_advice")):
                LOGGER.info("Not a Phase 3 question; delegating to the classic agent.")
                result = run_agent(question, tool_backend=tool_backend, persist=persist)
                result["engine"] = "classic (non-phase3 question)"
                result.setdefault("nodes_executed", [])
                return result
            trace = ToolTrace(backend=client.name)  # reset so the graph records cleanly

        app = _build_graph(client, trace, memory_context)
        final: Phase3GraphState = app.invoke({"question": question, "warnings": list(warnings),
                                              "memory_context": memory_context or {}})

        result = {
            "engine": "langgraph",
            "mode": "Phase 3 LangGraph (role-based nodes)",
            "question": question,
            "question_type": final.get("question_type", "phase3_cohort_statistics"),
            "answer": final.get("answer", ""),
            "intent": final.get("intent"),
            "evidence_card": final.get("evidence_card"),
            "grounding_validation": final.get("grounding_validation"),
            "memory_context": memory_context or {},
            "nodes_executed": final.get("nodes_executed", []),
            "sources_used": final.get("sources_used", []),
            "tool_backend": client.name,
            "tool_trace": trace.as_list(),
            "tools_called": trace.tool_names(),
            "tool_call_success_rate": trace.success_rate(),
            "average_tool_latency_ms": trace.average_latency_ms(),
            "warnings": final.get("warnings", list(warnings)),
        }
        if persist:
            try:
                result["trace_file"] = str(save_trace(result, datetime.now()))
            except Exception as exc:  # noqa: BLE001 - persistence is best-effort
                LOGGER.warning("Failed to persist graph trace: %s", exc)
        return result
    finally:
        if owns_client:
            client.close()


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print('Usage: python -m src.phase3_graph_agent "your Phase 3 question"')
        return
    if not is_langgraph_available():
        print("LangGraph is not installed. Install it with:\n    pip install langgraph")
        print("Falling back to the classic agent for this run.\n")
    result = run_phase3_graph_agent(" ".join(args))
    print(f"engine: {result.get('engine')}")
    print(f"nodes_executed: {result.get('nodes_executed')}")
    print(f"question_type: {result.get('question_type')}")
    print(f"tools_called: {result.get('tools_called')}")
    grounding = result.get("grounding_validation") or {}
    print(f"grounding: is_grounded={grounding.get('is_grounded')} unsupported={grounding.get('numbers_unsupported')}")
    print("\n--- answer ---")
    print(result.get("answer", ""))


if __name__ == "__main__":
    main()
