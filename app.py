from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.agent import run_agent
from src.config import EVALUATION_DIR
from src.evaluate_retrieval import run_retrieval_evaluation
from src.mcp_server import list_tools as list_agent_tools
from src.semantic_rag import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    build_chroma_collection,
    build_ollama_client,
    generate_grounded_answer,
    is_allowed_source_file,
    retrieve_semantic_chunks,
)

PHASE_1 = "Phase 1 — Semantic RAG Explorer"
PHASE_2 = "Phase 2 — Agentic RAG with Tools"
PHASE_3 = "Phase 3 — ICU Multi-Data Explorer"

PHASE_1_DESC = "Retrieves relevant chunks from ChromaDB and generates a grounded answer."
PHASE_2_DESC = "Uses a single LLM agent to call deterministic tools before generating the final answer."
PHASE_3_DESC = (
    "Same single agent over 25 MIMIC-IV ICU variables (labs + charted): lists variables, "
    "summarizes one, compares age groups / time windows, and returns an evidence card."
)

NON_CLINICAL_WARNING = (
    "This response is for academic interpretation only and is not a clinical diagnosis "
    "or treatment recommendation."
)

PATIENT_QUESTIONS = [
    "For a patient aged 82 with mean HR 104 bpm in the first 24h ICU stay, is this value high?",
    "For a patient aged 78 with MAP 62 mmHg in the first 24h ICU stay, is this value low?",
    "For a patient aged 86 with respiratory rate 24 in the first 12h ICU stay, is this elevated?",
    "For a patient aged 80 with SpO2 90% in the first 24h ICU stay, is this low?",
    "For a patient aged 75 with systolic blood pressure 145 mmHg in the first 6h ICU stay, is this high?",
    "For a patient aged 88 with temperature 38.5°C in the first 12h ICU stay, is this high?",
]
CONCEPT_QUESTIONS = [
    "What is the difference between a standard clinical threshold and an adaptive percentile-based threshold?",
    "Which MIMIC-IV tables are used to derive ICU vital-sign summaries?",
    "Why should raw chartevents not be indexed directly in a RAG system?",
    "Why are alarm items excluded from the vital sign pipeline?",
    "What are the limitations of using MIMIC-IV for clinical decision support?",
]
CALCULATOR_QUESTIONS = [
    "What is 104 * 2?",
    "If the value is 104 and we add 10, what is the result?",
]
PHASE3_QUESTIONS = [
    "What variables are available?",
    "What labs are available?",
    "Summarize lactate for patients aged 75-84 in the first 24h.",
    "Compare creatinine across age groups in first_24h.",
    "Compare MAP across time windows for patients aged 75-84.",
    "Show heart_rate statistics for age group 85+ in first_6h.",
    "Which age group has the highest median lactate in first_24h?",
    "What is the p90 of respiratory_rate for 65-74 in first_12h?",
    "Should this patient receive treatment for high lactate?",
]
SAMPLE_QUESTIONS = PATIENT_QUESTIONS + CONCEPT_QUESTIONS + CALCULATOR_QUESTIONS

# Tool catalogue shown in the UI. Same set for both backends (local & mcp_remote).
AGENT_TOOLS = list_agent_tools()

# Conceptual agent workflow -> trace tool name. ``None`` markers are handled
# specially (context extraction and final LLM phrasing are not MCP tools).
WORKFLOW_STEPS = [
    ("1. Extract patient context", "_context"),
    ("2. Check data availability", "check_data_availability"),
    ("3. Retrieve exact vital summary", "get_vital_summary"),
    ("4. Compare to standard threshold", "compare_to_standard_threshold"),
    ("5. Compare to MIMIC-IV percentiles", "compare_to_percentiles"),
    ("6. Generate structured interpretation", "generate_patient_interpretation_report"),
    ("7. Final LLM phrasing", "_llm"),
]


# --------------------------------------------------------------------------- #
# Small helpers (display only -- no business logic)
# --------------------------------------------------------------------------- #

def _short(obj: object, limit: int = 90) -> str:
    """Compact one-line preview of an input/output payload for tables."""

    try:
        text = json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(obj)
    return text[:limit] + ("…" if len(text) > limit else "")


def _strip_warning(text: str) -> str:
    """Remove the trailing non-clinical warning so it is shown only once."""

    if not text:
        return text
    return text.replace(NON_CLINICAL_WARNING, "").rstrip()


def _kv_table(rows: list[tuple[str, object]]) -> None:
    # Coerce to strings so the "Value" column is uniformly typed (avoids Arrow
    # serialization warnings when mixing numbers, strings and None).
    display_rows = [(field, "—" if value is None else str(value)) for field, value in rows]
    frame = pd.DataFrame(display_rows, columns=["Field", "Value"])
    st.dataframe(frame, hide_index=True, width="stretch")


def _raw_json(label: str, obj: object, demo_friendly: bool) -> None:
    """Show raw JSON inside an expander (collapsed in demo mode)."""

    with st.expander(label, expanded=not demo_friendly):
        st.json(obj)


def _why_selected(chunk: dict) -> str:
    """Human-readable reason a chunk was retrieved, from existing metadata."""

    if chunk.get("is_exact_match"):
        return "exact vital sign + age group + time window match"

    reasons: list[str] = []
    if chunk.get("detected_vital_sign") and chunk.get("vital_sign") == chunk.get("detected_vital_sign"):
        reasons.append("exact vital sign match")
    if chunk.get("detected_age_group") and chunk.get("age_group") == chunk.get("detected_age_group"):
        reasons.append("exact age group match")
    if chunk.get("detected_time_window") and chunk.get("time_window") == chunk.get("detected_time_window"):
        reasons.append("exact time window match")
    if reasons:
        return "; ".join(reasons)

    source_type = str(chunk.get("source_type", ""))
    if source_type in {"project_report", "documentation"}:
        return "project documentation"
    if source_type in {"article", "guideline"}:
        return "concept source"
    return "semantic similarity"


@st.cache_data(ttl=60, show_spinner=False)
def _system_status() -> dict[str, tuple[bool, str]]:
    """Best-effort technical status checklist. Never raises."""

    status: dict[str, tuple[bool, str]] = {}

    try:
        collection = build_chroma_collection()
        count = collection.count()
        status["ChromaDB index"] = (True, f"{count} indexed chunks")
    except Exception as exc:  # noqa: BLE001 - status panel must not crash
        status["ChromaDB index"] = (False, str(exc)[:80])

    model_names: list[str] = []
    try:
        client = build_ollama_client()
        model_names = [model.id for model in client.models.list().data]
        status["Ollama reachable"] = (True, f"{len(model_names)} models")
    except Exception as exc:  # noqa: BLE001
        status["Ollama reachable"] = (False, str(exc)[:80])

    has_bge = any("bge-m3" in name for name in model_names)
    has_qwen = any("qwen2.5" in name for name in model_names)
    status["bge-m3 embedding"] = (has_bge, EMBEDDING_MODEL if has_bge else "not pulled")
    status["qwen2.5 LLM"] = (has_qwen, LLM_MODEL if has_qwen else "not pulled")

    try:
        from src.mcp_server import list_tools

        tool_count = len(list_tools())
        status["Phase 2 tools registered"] = (tool_count > 0, f"{tool_count} tools")
    except Exception as exc:  # noqa: BLE001
        status["Phase 2 tools registered"] = (False, str(exc)[:80])

    try:
        from src.mcp_server import call_tool  # noqa: F401

        try:
            import mcp  # noqa: F401

            detail = "local call_tool + mcp SDK"
        except Exception:  # noqa: BLE001 - SDK optional
            detail = "local call_tool (mcp SDK optional)"
        status["MCP-compatible layer"] = (True, detail)
    except Exception as exc:  # noqa: BLE001
        status["MCP-compatible layer"] = (False, str(exc)[:80])

    return status


# --------------------------------------------------------------------------- #
# Page setup + styles
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="ICU Trajectory RAG Assistant", page_icon="ICU", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 76, 129, 0.14), transparent 24%),
                radial-gradient(circle at top right, rgba(23, 120, 103, 0.12), transparent 20%),
                linear-gradient(180deg, #f7f9fc 0%, #eef3f8 100%);
        }
        .hero {
            padding: 1.2rem 1.4rem;
            border-radius: 1rem;
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
        }
        .card {
            padding: 1rem 1.1rem;
            border-radius: 0.9rem;
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
        }
        .small-muted { color: #5b6472; font-size: 0.92rem; }
        .section-label {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #4a5568;
            font-size: 0.75rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }
        .chip {
            display: inline-block;
            padding: 0.26rem 0.7rem;
            margin: 0.35rem 0.35rem 0 0;
            border-radius: 999px;
            background: rgba(15, 76, 129, 0.08);
            border: 1px solid rgba(15, 23, 42, 0.08);
            font-size: 0.8rem;
            color: #26405c;
        }
        .chip b { color: #0f4c81; }
        .chip-safe { background: rgba(23, 120, 103, 0.10); color: #176767; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Sidebar: mode, demo toggle, query builder, controls, status
# --------------------------------------------------------------------------- #

if "question" not in st.session_state:
    st.session_state.question = SAMPLE_QUESTIONS[0]

st.sidebar.markdown("### Pipeline mode")
mode = st.sidebar.radio(
    "Choose the pipeline",
    [PHASE_1, PHASE_2, PHASE_3],
    label_visibility="collapsed",
)
is_phase2 = mode == PHASE_2
is_phase3 = mode == PHASE_3
is_phase1 = mode == PHASE_1
uses_agent = is_phase2 or is_phase3
st.sidebar.caption({PHASE_1: PHASE_1_DESC, PHASE_2: PHASE_2_DESC, PHASE_3: PHASE_3_DESC}[mode])

demo_friendly = st.sidebar.toggle(
    "Demo-friendly display",
    value=True,
    help="ON: cards, short tables and explanations, raw JSON tucked into expanders. "
    "OFF: more technical detail expanded for debugging.",
)

TOOL_BACKEND_LABELS = {
    "Local backend": "local",
    "MCP remote backend": "mcp_remote",
}
tool_backend = "local"
if uses_agent:
    with st.sidebar.expander("Advanced settings (agent backend)", expanded=False):
        backend_label = st.radio(
            "Tool execution backend",
            list(TOOL_BACKEND_LABELS),
            captions=[
                "Runs tools in-process. Stable default mode.",
                "Calls the same tools through a real MCP HTTP server.",
            ],
        )
        tool_backend = TOOL_BACKEND_LABELS[backend_label]
        if tool_backend == "mcp_remote":
            st.caption(
                "Start `python src/server_mcp.py` first. If unreachable, the agent "
                "falls back to the local backend with a warning."
            )

st.sidebar.markdown("### Suggested questions")
suggested_options = (
    [("Custom — keep my own question", None)]
    + [(f"Patient · {q}", q) for q in PATIENT_QUESTIONS]
    + [(f"Concept · {q}", q) for q in CONCEPT_QUESTIONS]
    + [(f"Calculator · {q}", q) for q in CALCULATOR_QUESTIONS]
    + [(f"Phase 3 · {q}", q) for q in PHASE3_QUESTIONS]
)
suggested_labels = [label for label, _ in suggested_options]
suggested_map = dict(suggested_options)
chosen_label = st.sidebar.selectbox("Pick an example", suggested_labels, label_visibility="collapsed")
chosen_question = suggested_map[chosen_label]
if chosen_question:
    st.session_state.question = chosen_question

st.sidebar.markdown("### Retrieval controls")
st.sidebar.caption("These shape retrieval only — they do not change tool calculations.")
top_k = st.sidebar.slider("Top-k chunks", min_value=3, max_value=10, value=5, step=1)
source_type_filter = st.sidebar.selectbox(
    "Source type",
    ["All", "mimic_stats", "project_report", "documentation", "article", "guideline"],
)
vital_sign_filter = st.sidebar.selectbox(
    "Vital sign",
    ["All", "Heart Rate", "Respiratory Rate", "MAP", "Systolic Blood Pressure", "Diastolic Blood Pressure", "Temperature", "SpO2"],
)
age_group_filter = st.sidebar.selectbox("Age group", ["All", "65-74", "75-84", "85+"])
time_window_filter = st.sidebar.selectbox("Time window", ["All", "first_6h", "first_12h", "first_24h"])

run_col, reset_col = st.sidebar.columns(2)
ask_clicked = run_col.button(
    "Run agent" if uses_agent else "Run retrieval", type="primary", use_container_width=True
)
reset_clicked = reset_col.button("Reset", use_container_width=True)
if reset_clicked:
    st.session_state.question = SAMPLE_QUESTIONS[0]
    st.rerun()

st.sidebar.markdown("### Technical status")
for label, (ok, detail) in _system_status().items():
    icon = "✅" if ok else "⚠️"
    st.sidebar.markdown(f"{icon} **{label}** — <span class='small-muted'>{detail}</span>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Hero / summary banner
# --------------------------------------------------------------------------- #

mode_badge = {
    PHASE_1: "Phase 1 — Semantic RAG",
    PHASE_2: "Phase 2 — Agentic RAG",
    PHASE_3: "Phase 3 — ICU Multi-Data Explorer",
}[mode]
st.markdown(
    f"""
    <div class="hero">
        <div class="section-label">ICU Trajectory RAG Assistant</div>
        <h1 style="margin:0 0 0.35rem 0;">ICU Trajectory RAG Assistant</h1>
        <p style="margin:0 0 0.4rem 0; max-width: 900px;" class="small-muted">
            Local academic assistant for exploring MIMIC-IV-derived ICU vital-sign summaries.
        </p>
        <div>
            <span class="chip">Mode: <b>{mode_badge}</b></span>
            <span class="chip">Index: <b>ChromaDB</b></span>
            <span class="chip">Embedding: <b>{EMBEDDING_MODEL}</b></span>
            <span class="chip">LLM: <b>{LLM_MODEL}</b></span>
            <span class="chip chip-safe">Safety: Non-clinical academic interpretation</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

question = st.text_area(
    "Question",
    value=st.session_state.question,
    height=120,
    placeholder="Ask about a vital sign, a threshold, or a dataset detail...",
)
st.session_state.question = question


# --------------------------------------------------------------------------- #
# Phase 1 — Semantic RAG
# --------------------------------------------------------------------------- #

if ask_clicked and is_phase1:
    with st.spinner("Retrieving local chunks..."):
        try:
            retrieved = retrieve_semantic_chunks(
                question,
                top_k=top_k,
                source_type_filter=source_type_filter,
                vital_sign_filter=vital_sign_filter,
                age_group_filter=age_group_filter,
                time_window_filter=time_window_filter,
            )
        except FileNotFoundError as exc:
            st.error(f"{exc}\n\nRun `python src/ingest.py` first to rebuild ChromaDB.")
            st.stop()

    if not retrieved:
        st.warning("No semantic chunks matched the current filters. Try broadening the source or vital-sign filters.")
        st.stop()

    with st.spinner(f"Generating grounded answer with {LLM_MODEL}..."):
        answer_text = generate_grounded_answer(question, retrieved)

    st.markdown("### Final answer")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(_strip_warning(answer_text))
    st.caption("Grounded in retrieved ChromaDB chunks only.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.warning(NON_CLINICAL_WARNING)

    retrieval_info = retrieved[0]
    quality_cols = st.columns(4)
    quality_cols[0].metric("Retrieval latency", f"{float(retrieval_info.get('retrieval_latency_ms', 0.0)):.1f} ms")
    quality_cols[1].metric("Candidates retrieved", int(retrieval_info.get("candidate_count_requested", len(retrieved))))
    quality_cols[2].metric("Question intent", str(retrieval_info.get("question_intent", "general_question")))
    quality_cols[3].metric("Exact metadata match", "yes" if retrieval_info.get("is_exact_match") else "no")

    st.markdown("### RAG sources used")
    sources_rows = [
        {
            "Rank": item.get("rank"),
            "Source file": item.get("source_file"),
            "Source type": item.get("source_type"),
            "Vital sign": item.get("vital_sign"),
            "Age group": item.get("age_group"),
            "Time window": item.get("time_window"),
            "Final score": round(float(item.get("final_score", item.get("semantic_score", 0.0))), 4),
            "Why selected": _why_selected(item),
            "Preview": str(item.get("chunk_preview", ""))[:200],
        }
        for item in retrieved
        if is_allowed_source_file(str(item.get("source_file", "")))
    ]
    if sources_rows:
        st.dataframe(pd.DataFrame(sources_rows), width="stretch", hide_index=True)
    else:
        st.info("No sources available for the current query.")

    with st.expander("Retrieved chunks and reranking detail", expanded=not demo_friendly):
        chunks_view = pd.DataFrame(
            [
                {
                    "rank": item.get("rank"),
                    "semantic_rank": item.get("semantic_rank"),
                    "source_file": item.get("source_file"),
                    "source_type": item.get("source_type"),
                    "vital_sign": item.get("vital_sign"),
                    "age_group": item.get("age_group"),
                    "time_window": item.get("time_window"),
                    "semantic_score": round(float(item.get("semantic_score", 0.0)), 4),
                    "metadata_bonus": round(float(item.get("metadata_bonus", 0.0)), 4),
                    "mismatch_penalty": round(float(item.get("mismatch_penalty", 0.0)), 4),
                    "final_score": round(float(item.get("final_score", 0.0)), 4),
                    "title": item.get("title"),
                }
                for item in retrieved
            ]
        )
        st.dataframe(chunks_view, width="stretch", hide_index=True)

    with st.expander("Method and framing", expanded=False):
        st.write(
            "This Phase 1 response is a research aid for interpretation only and must not be used as a clinical "
            "decision. It is grounded in the retrieved ChromaDB context and does not call agents or tools."
        )


# --------------------------------------------------------------------------- #
# Phase 2 — Agentic RAG with tools
# --------------------------------------------------------------------------- #

elif ask_clicked and is_phase2:
    with st.spinner("Running the single agent (deterministic tools + grounded LLM)..."):
        agent_result = run_agent(question, top_k=top_k, tool_backend=tool_backend)

    trace = agent_result.get("tool_trace", [])
    warnings = agent_result.get("warnings", [])
    patient_context = agent_result.get("patient_context", {})
    trace_by_name = {entry.get("tool_name"): entry for entry in trace}
    is_calc = agent_result.get("question_type") == "calculator_question"

    # 3. Final answer ------------------------------------------------------- #
    st.markdown("### Final answer")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(_strip_warning(agent_result.get("answer", "")))
    if is_calc:
        st.caption("Computed by the `calculatrice_medicale` tool (arithmetic helper, not clinical).")
    else:
        st.caption("Grounded in deterministic tool outputs and retrieved sources only.")
    st.markdown("</div>", unsafe_allow_html=True)
    # Heavy non-clinical warning only for medical questions, not pure arithmetic.
    if not is_calc:
        st.warning(NON_CLINICAL_WARNING)

    for warning in warnings:
        st.warning(warning)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Question type", str(agent_result.get("question_type", "")))
    metric_cols[1].metric("Tools called", len(agent_result.get("tools_called", [])))
    metric_cols[2].metric("Tool success", f"{agent_result.get('tool_call_success_rate', 0.0) * 100:.0f}%")
    metric_cols[3].metric("Tool backend", str(agent_result.get("tool_backend", "local")))
    if tool_backend == "mcp_remote":
        st.caption(
            f"Requested backend: `mcp_remote` · actually used: `{agent_result.get('tool_backend')}` "
            f"(avg tool latency {agent_result.get('average_tool_latency_ms', 0.0):.1f} ms)."
        )

    # Tools available to the agent (same catalogue for both backends).
    with st.expander("Tools available to the agent", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [{"Tool": t["name"], "Description": t["description"]} for t in AGENT_TOOLS]
            ),
            width="stretch",
            hide_index=True,
        )

    def _find_tool_output(name: str):
        entry = trace_by_name.get(name)
        if entry is not None and entry.get("status") == "success":
            return entry.get("outputs")
        return None

    if is_calc:
        # 4'. Calculator result — concise, no patient context / MIMIC comparison. #
        calc_output = _find_tool_output("calculatrice_medicale") or {}
        st.markdown("### Calculator result")
        _kv_table(
            [
                ("Tool", "calculatrice_medicale"),
                ("Expression", calc_output.get("expression")),
                ("Result", calc_output.get("result")),
                ("Status", calc_output.get("status")),
            ]
        )
        st.caption("The agent selected an arithmetic tool — no ChromaDB / MIMIC-IV lookup was needed.")
    else:
        # 4. Detected patient context -------------------------------------- #
        st.markdown("### Detected patient context")
        _kv_table(
            [
                ("Question type", agent_result.get("question_type", "")),
                ("Patient age", patient_context.get("age")),
                ("Age group", patient_context.get("age_group")),
                ("Vital sign", patient_context.get("vital_sign")),
                ("Value", patient_context.get("value")),
                ("Time window", patient_context.get("time_window")),
            ]
        )
        _raw_json("Raw patient context", patient_context, demo_friendly)

        # 5. Agent workflow ------------------------------------------------ #
        st.markdown("### Agent workflow")
        llm_fallback = any("LLM unavailable" in w for w in warnings)
        has_patient = bool(patient_context.get("vital_sign"))

        workflow_rows = []
        for step_label, marker in WORKFLOW_STEPS:
            if marker == "_context":
                status = "✓ Done" if has_patient else "— Not applicable"
                detail = "Parsed from the question" if has_patient else "Non-patient question"
            elif marker == "_llm":
                status = "✓ Done" if not llm_fallback else "⚠ Fallback (deterministic)"
                detail = "qwen2.5 phrased the answer" if not llm_fallback else "LLM unavailable; deterministic phrasing"
            else:
                entry = trace_by_name.get(marker)
                if entry is None:
                    status, detail = "— Not applicable", "Tool not called for this question"
                elif entry.get("status") == "success":
                    status, detail = "✓ Done", f"{entry.get('latency_ms', 0)} ms"
                else:
                    status, detail = "⚠ Error", str(entry.get("outputs_summary", ""))[:80]
            workflow_rows.append({"Step": step_label, "Status": status, "Detail": detail})
        st.dataframe(pd.DataFrame(workflow_rows), width="stretch", hide_index=True)

    standard_output = _find_tool_output("compare_to_standard_threshold")
    percentile_output = _find_tool_output("compare_to_percentiles")
    summary_output = _find_tool_output("get_vital_summary")

    if standard_output or percentile_output:
        st.markdown("### Threshold comparisons")
        comp_a, comp_b = st.columns(2)

        with comp_a:
            st.markdown("#### Standard-threshold comparison")
            if standard_output:
                unit = (summary_output or {}).get("unitname", "")
                value = patient_context.get("value")
                _kv_table(
                    [
                        ("Status", standard_output.get("status")),
                        ("Value", f"{value} {unit}".strip() if value is not None else "—"),
                        ("Standard low", standard_output.get("standard_low")),
                        ("Standard high", standard_output.get("standard_high")),
                        ("Explanation", standard_output.get("explanation")),
                    ]
                )
                _raw_json("Raw standard-threshold output", standard_output, demo_friendly)
            else:
                st.caption("Not applicable for this question.")

        with comp_b:
            st.markdown("#### MIMIC-IV percentile comparison")
            if percentile_output:
                pct = percentile_output.get("used_percentiles", {}) or {}
                _kv_table(
                    [
                        ("Percentile position", percentile_output.get("percentile_position")),
                        ("P5", pct.get("p5")),
                        ("P25", pct.get("p25")),
                        ("P50", pct.get("p50")),
                        ("P75", pct.get("p75")),
                        ("P90", pct.get("p90")),
                        ("Explanation", percentile_output.get("explanation")),
                    ]
                )
                _raw_json("Raw percentile output", percentile_output, demo_friendly)
            else:
                st.caption("Not applicable for this question.")

    # 6. Tool trace -------------------------------------------------------- #
    st.markdown("### Agent tool trace")
    if trace:
        trace_df = pd.DataFrame(
            [
                {
                    "Step": entry.get("step"),
                    "Tool name": entry.get("tool_name"),
                    "Backend": entry.get("backend", "local"),
                    "Status": entry.get("status"),
                    "Latency (ms)": entry.get("latency_ms"),
                    "Input summary": _short(entry.get("inputs", {})),
                    "Output summary": _short(entry.get("outputs_summary", "")),
                }
                for entry in trace
            ]
        )
        st.dataframe(trace_df, width="stretch", hide_index=True)

        for entry in trace:
            with st.expander(
                f"Step {entry.get('step')} — {entry.get('tool_name')} — show raw input/output",
                expanded=not demo_friendly,
            ):
                st.markdown("**Input**")
                st.json(entry.get("inputs", {}))
                st.markdown("**Output**")
                st.json(entry.get("outputs", {}))
    else:
        st.info("No tools were called for this question.")

    # 8. RAG sources used -------------------------------------------------- #
    st.markdown("### RAG sources used")
    sources_used = agent_result.get("sources_used", [])
    if sources_used:
        st.dataframe(
            pd.DataFrame({"Rank": range(1, len(sources_used) + 1), "Source file": sources_used}),
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("No documentary sources were used for this question.")

    if agent_result.get("trace_file"):
        st.caption(f"Auditable trace saved to {agent_result['trace_file']}")

    # 10. Why this is Phase 2 ---------------------------------------------- #
    st.markdown("### Why this is Phase 2")
    st.info(
        "This mode uses a single LLM agent to orchestrate deterministic tools. The RAG retrieves knowledge, "
        "while tools perform data availability checks, threshold comparisons and percentile comparisons. "
        "The final answer is grounded and auditable through the tool trace."
    )


# --------------------------------------------------------------------------- #
# Phase 3 — ICU Multi-Data Explorer (same agent, multi-variable tools)
# --------------------------------------------------------------------------- #

elif ask_clicked and is_phase3:
    with st.spinner("Running the ICU Multi-Data agent (deterministic tools + grounded LLM)..."):
        agent_result = run_agent(question, top_k=top_k, tool_backend=tool_backend)

    trace = agent_result.get("tool_trace", [])
    warnings = agent_result.get("warnings", [])
    trace_by_name = {entry.get("tool_name"): entry for entry in trace}
    qtype = str(agent_result.get("question_type", ""))
    evidence_card = agent_result.get("evidence_card")

    st.markdown("### Final answer")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(_strip_warning(agent_result.get("answer", "")))
    st.caption("Descriptive, non-clinical. Grounded in `icu_feature_summary.csv` via deterministic tools.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.warning(NON_CLINICAL_WARNING)
    for warning in warnings:
        st.warning(warning)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Question type", qtype)
    metric_cols[1].metric("Tools called", len(agent_result.get("tools_called", [])))
    metric_cols[2].metric("Tool success", f"{agent_result.get('tool_call_success_rate', 0.0) * 100:.0f}%")
    metric_cols[3].metric("Tool backend", str(agent_result.get("tool_backend", "local")))

    def _p3_output(name: str):
        entry = trace_by_name.get(name)
        if entry is not None and entry.get("status") == "success":
            return entry.get("outputs")
        return None

    if qtype == "clinical_advice_refused":
        st.error("Clinical-advice request refused: this tool is descriptive and non-clinical only.")

    if isinstance(evidence_card, dict) and not evidence_card.get("error"):
        st.markdown("### Evidence card")
        _kv_table([
            ("Variable", evidence_card.get("variable")),
            ("Category", evidence_card.get("category")),
            ("Source table", evidence_card.get("source_table")),
            ("Itemids", evidence_card.get("itemids")),
            ("Unit", evidence_card.get("unit")),
            ("Age group", evidence_card.get("age_group")),
            ("Time window", evidence_card.get("time_window")),
            ("N patients", evidence_card.get("n_patients")),
            ("N measurements", evidence_card.get("n_measurements")),
            ("Main metric", evidence_card.get("main_metric")),
            ("Missing rate warning", evidence_card.get("missing_rate_warning")),
        ])

    avail = _p3_output("list_available_variables")
    if avail and avail.get("variables"):
        st.markdown("### Available ICU variables")
        st.dataframe(pd.DataFrame(avail["variables"]), width="stretch", hide_index=True)

    vsum = _p3_output("get_variable_summary")
    if vsum and not vsum.get("error"):
        st.markdown("### Variable summary")
        _kv_table([
            ("Variable", vsum.get("variable_name")), ("Unit", vsum.get("unit")),
            ("Age group", vsum.get("age_group")), ("Time window", vsum.get("time_window")),
            ("N patients", vsum.get("n_patients")), ("N measurements", vsum.get("n_measurements")),
            ("Mean", vsum.get("mean")), ("Std", vsum.get("std")), ("Median", vsum.get("median")),
            ("P05", vsum.get("p05")), ("P25", vsum.get("p25")), ("P75", vsum.get("p75")),
            ("P90", vsum.get("p90")), ("P95", vsum.get("p95")), ("Missing rate", vsum.get("missing_rate")),
        ])

    cag = _p3_output("compare_age_groups")
    if cag and not cag.get("error"):
        st.markdown("### Comparison across age groups")
        metric_name = cag.get("metric", "value")
        chart_df = pd.DataFrame(
            [{"age_group": k, metric_name: v} for k, v in cag.get("values_by_age_group", {}).items() if v is not None]
        )
        if not chart_df.empty:
            st.bar_chart(chart_df.set_index("age_group"))
            st.dataframe(chart_df, width="stretch", hide_index=True)
        st.caption(cag.get("descriptive", ""))

    ctw = _p3_output("compare_time_windows")
    if ctw and not ctw.get("error"):
        st.markdown("### Comparison across time windows")
        metric_name = ctw.get("metric", "value")
        chart_df = pd.DataFrame(
            [{"time_window": k, metric_name: v} for k, v in ctw.get("values_by_time_window", {}).items() if v is not None]
        )
        if not chart_df.empty:
            st.bar_chart(chart_df.set_index("time_window"))
            st.dataframe(chart_df, width="stretch", hide_index=True)
        st.caption(f"Trend: {ctw.get('trend')} — {ctw.get('descriptive', '')}")

    cohort = _p3_output("query_cohort_statistics")
    if cohort and cohort.get("rows"):
        st.markdown("### Cohort statistics")
        st.dataframe(pd.DataFrame(cohort["rows"]), width="stretch", hide_index=True)

    st.markdown("### Agent tool trace")
    if trace:
        st.dataframe(
            pd.DataFrame([
                {
                    "Step": entry.get("step"),
                    "Tool name": entry.get("tool_name"),
                    "Backend": entry.get("backend", "local"),
                    "Status": entry.get("status"),
                    "Latency (ms)": entry.get("latency_ms"),
                    "Input summary": _short(entry.get("inputs", {})),
                    "Output summary": _short(entry.get("outputs_summary", "")),
                }
                for entry in trace
            ]),
            width="stretch", hide_index=True,
        )
        for entry in trace:
            with st.expander(
                f"Step {entry.get('step')} — {entry.get('tool_name')} — raw input/output",
                expanded=not demo_friendly,
            ):
                st.markdown("**Input**")
                st.json(entry.get("inputs", {}))
                st.markdown("**Output**")
                st.json(entry.get("outputs", {}))
    else:
        st.info("No tools were called for this question.")

    if agent_result.get("trace_file"):
        st.caption(f"Auditable trace saved to {agent_result['trace_file']}")

    st.markdown("### Why this is Phase 3")
    st.info(
        "This mode reuses the SAME single agent, tool client and auditable trace as Phase 2, but over "
        "25 MIMIC-IV ICU variables (13 labs + 12 charted). All tools are deterministic and descriptive; "
        "the agent refuses diagnosis/treatment requests via a non-clinical safety gate."
    )


# --------------------------------------------------------------------------- #
# Idle / quick-start state
# --------------------------------------------------------------------------- #

else:
    st.markdown("### Quick start")
    quick_col_1, quick_col_2 = st.columns(2)
    with quick_col_1:
        st.markdown(
            f"""
            <div class="card">
            <div class="section-label">{PHASE_1}</div>
            <p class="small-muted">{PHASE_1_DESC} Use the suggested patient-value questions to test whether the
            top retrieved chunk matches the right vital sign, age group, and time window.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with quick_col_2:
        st.markdown(
            f"""
            <div class="card">
            <div class="section-label">{PHASE_2}</div>
            <p class="small-muted">{PHASE_2_DESC} The detected patient context, the tools called, the
            standard / percentile comparisons and the auditable tool trace are all shown.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Retrieval evaluation (compact)
# --------------------------------------------------------------------------- #

st.markdown("### Retrieval strategy comparison")
st.caption("This is a small Phase 1 retrieval benchmark, not a clinical validation.")

run_eval = st.button("Run / refresh retrieval evaluation")
if run_eval:
    with st.spinner("Running the strategy comparison benchmark..."):
        csv_path, md_path = run_retrieval_evaluation()
    st.success(f"Saved evaluation results to {csv_path} and {md_path}")

evaluation_summary_path = EVALUATION_DIR / "retrieval_summary.md"
evaluation_results_path = EVALUATION_DIR / "retrieval_evaluation.csv"

if evaluation_results_path.exists():
    evaluation_df = pd.read_csv(evaluation_results_path)
    try:
        from src.evaluate_retrieval import STRATEGY_METADATA

        compact_rows = []
        for strategy_key, group in evaluation_df.groupby("strategy"):
            meta = STRATEGY_METADATA.get(strategy_key, {})
            compact_rows.append(
                {
                    "Strategy": meta.get("name", strategy_key),
                    "Precision proxy (exact@1)": round(float(group["exact_metadata_match_at_1"].mean()), 2),
                    "Recall proxy (exact@k)": round(float(group["exact_metadata_match_at_k"].mean()), 2),
                    "Latency (ms)": round(float(group["retrieval_latency_ms"].mean()), 1),
                    "Cost": meta.get("cost", "—"),
                    "Complexity": meta.get("complexity", "—"),
                }
            )
        st.dataframe(pd.DataFrame(compact_rows), width="stretch", hide_index=True)
    except Exception as exc:  # noqa: BLE001 - summary fallback
        st.caption(f"Could not build the compact summary ({exc}).")

    with st.expander("Interpretation of retrieval strategies", expanded=False):
        if evaluation_summary_path.exists():
            st.markdown(evaluation_summary_path.read_text(encoding="utf-8"))
        st.dataframe(
            evaluation_df[
                [
                    "strategy",
                    "question",
                    "question_type",
                    "top1_source_type_match",
                    "top1_vital_sign_match",
                    "top1_age_group_match",
                    "top1_time_window_match",
                    "exact_metadata_match_at_1",
                    "exact_metadata_match_at_k",
                    "retrieval_latency_ms",
                    "number_of_candidates_retrieved",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
else:
    st.caption("Run the evaluation once to generate the comparison table and summary files.")
