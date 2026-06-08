from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.agent import run_agent
from src.config import EVALUATION_DIR
from src.evaluate_retrieval import run_retrieval_evaluation
from src.semantic_rag import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    generate_grounded_answer,
    is_allowed_source_file,
    retrieve_semantic_chunks,
)

PHASE_1 = "Phase 1 Semantic RAG"
PHASE_2 = "Phase 2 Agentic RAG"


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
            background: rgba(255, 255, 255, 0.82);
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
        .small-muted {
            color: #5b6472;
            font-size: 0.92rem;
        }
        .section-label {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #4a5568;
            font-size: 0.75rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="section-label">ICU Trajectory RAG Assistant</div>
        <h1 style="margin:0 0 0.35rem 0;">Phase 1 Semantic RAG Explorer</h1>
        <p style="margin:0; max-width: 900px;">
            Local semantic retrieval over MIMIC-IV summaries and project documents, with ChromaDB embeddings and a grounded local LLM answer.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

sample_questions = [
    "What is the difference between a standard clinical threshold and an adaptive percentile-based threshold?",
    "For a patient aged 82 with mean HR 104 bpm in the first 24h ICU stay, is this value high?",
    "For a patient aged 78 with MAP 62 mmHg in the first 24h ICU stay, is this low?",
    "For a patient aged 86 with respiratory rate 24 in the first 12h ICU stay, is this elevated?",
    "For a patient aged 80 with SpO2 90% in the first 24h ICU stay, is this low?",
    "For a patient aged 75 with systolic blood pressure 145 mmHg in the first 6h ICU stay, is this high?",
    "For a patient aged 88 with temperature 38.5°C in the first 12h ICU stay, is this high?",
    "Which MIMIC-IV tables are useful for ICU vital signs?",
    "Why should raw chartevents not be indexed directly in a RAG system?",
    "Why are alarm items excluded from the vital sign pipeline?",
    "What are the limitations of using MIMIC-IV for clinical decision support?",
]

if "question" not in st.session_state:
    st.session_state.question = sample_questions[0]
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = []

st.sidebar.markdown("### Mode")
mode = st.sidebar.radio(
    "Pipeline mode",
    [PHASE_1, PHASE_2],
    help="Phase 1: semantic retrieval + grounded answer. Phase 2: a single LLM agent orchestrating deterministic MCP tools.",
)
is_phase2 = mode == PHASE_2

st.sidebar.markdown("### Query Builder")
st.sidebar.caption("Choose a preset or keep your own question. The controls below shape retrieval only.")

sample_choice = st.sidebar.selectbox("Suggested question", ["Custom"] + sample_questions)
if sample_choice != "Custom":
    st.session_state.question = sample_choice

question = st.text_area("Question", value=st.session_state.question, height=140, placeholder="Ask about a vital sign, a threshold, or a dataset detail...")
st.session_state.question = question

st.sidebar.markdown("### Retrieval Controls")
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
    "Run agent" if is_phase2 else "Run retrieval", type="primary", use_container_width=True
)
reset_clicked = reset_col.button("Reset", use_container_width=True)
if reset_clicked:
    st.session_state.question = sample_questions[0]
    st.session_state.last_answer = ""
    st.session_state.last_retrieved = []
    st.rerun()

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Mode", "Phase 2 Agentic" if is_phase2 else "Phase 1 Semantic RAG")
col_b.metric("Index", "ChromaDB")
col_c.metric("Embedding model", EMBEDDING_MODEL)
col_d.metric("LLM", LLM_MODEL)

st.caption("Source focus: MIMIC-IV summaries + project documents")

if ask_clicked and not is_phase2:
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

    with st.spinner("Generating grounded answer with qwen2.5:14b..."):
        answer_text = generate_grounded_answer(question, retrieved)

    st.session_state.last_answer = answer_text
    st.session_state.last_retrieved = retrieved

    answer_col, sources_col = st.columns([1.45, 1])

    with answer_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Answer")
        st.markdown(answer_text)
        st.caption("Grounded in retrieved ChromaDB chunks only.")
        st.markdown('</div>', unsafe_allow_html=True)

    with sources_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Sources used")
        sources_df = pd.DataFrame(
            [
                {
                    "rank": item.get("rank"),
                    "source_file": item.get("source_file"),
                    "source_type": item.get("source_type"),
                    "vital_sign": item.get("vital_sign"),
                    "age_group": item.get("age_group"),
                    "time_window": item.get("time_window"),
                    "semantic_score": round(float(item.get("semantic_score", 0.0)), 4),
                    "metadata_bonus": round(float(item.get("metadata_bonus", 0.0)), 4),
                    "mismatch_penalty": round(float(item.get("mismatch_penalty", 0.0)), 4),
                    "final_score": round(float(item.get("final_score", item.get("semantic_score", 0.0))), 4),
                    "title": item.get("title"),
                    "chunk_preview": item.get("chunk_preview"),
                }
                for item in retrieved
                if is_allowed_source_file(str(item.get("source_file", "")))
            ]
        )
        if not sources_df.empty:
            st.dataframe(sources_df, width="stretch", hide_index=True)
        else:
            st.info("No sources available for the current query.")
        st.markdown('</div>', unsafe_allow_html=True)

    retrieval_info = retrieved[0]
    quality_cols = st.columns(4)
    quality_cols[0].metric("Retrieval latency", f"{float(retrieval_info.get('retrieval_latency_ms', 0.0)):.1f} ms")
    quality_cols[1].metric("Candidates retrieved", int(retrieval_info.get("candidate_count_requested", len(retrieved))))
    quality_cols[2].metric("Question intent", str(retrieval_info.get("question_intent", "general_question")))
    quality_cols[3].metric("Exact metadata match", "yes" if retrieval_info.get("is_exact_match") else "no")

    st.markdown("### Retrieved chunks")
    if retrieved:
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
                    "chunk_preview": item.get("chunk_preview"),
                }
                for item in retrieved
            ]
        )
        with st.expander("Show retrieved chunks and metadata", expanded=True):
            st.dataframe(chunks_view, width="stretch", hide_index=True)
    else:
        st.info("No chunks were retrieved for this query.")

    with st.expander("Method and framing", expanded=False):
        st.write(
            "This response is a research aid for interpretation only and must not be used as a clinical decision. "
            "It is grounded in the retrieved ChromaDB context and does not implement agents, MCP, or function calling."
        )

elif ask_clicked and is_phase2:
    with st.spinner("Running the single agent (deterministic tools + grounded LLM)..."):
        agent_result = run_agent(question, top_k=top_k)

    st.markdown("### Why this is Phase 2")
    st.info(
        "The system now uses a single LLM agent to orchestrate RAG retrieval and deterministic tools. "
        "The agent can check data availability, retrieve exact vital summaries, compare values to thresholds "
        "and percentiles, and produce a grounded answer with an auditable tool trace."
    )

    answer_col, ctx_col = st.columns([1.5, 1])
    with answer_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Final answer")
        st.markdown(agent_result.get("answer", ""))
        st.caption("Grounded in deterministic tool outputs and retrieved sources only.")
        st.markdown('</div>', unsafe_allow_html=True)
    with ctx_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Detected patient context")
        st.json(agent_result.get("patient_context", {}))
        st.markdown('</div>', unsafe_allow_html=True)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Question type", str(agent_result.get("question_type", "")))
    metric_cols[1].metric("Tools called", len(agent_result.get("tools_called", [])))
    metric_cols[2].metric("Tool success", f"{agent_result.get('tool_call_success_rate', 0.0) * 100:.0f}%")
    metric_cols[3].metric("Avg tool latency", f"{agent_result.get('average_tool_latency_ms', 0.0):.1f} ms")

    for warning in agent_result.get("warnings", []):
        st.warning(warning)

    trace = agent_result.get("tool_trace", [])

    def _find_tool_output(name):
        for entry in trace:
            if entry.get("tool_name") == name and entry.get("status") == "success":
                return entry.get("outputs")
        return None

    standard_output = _find_tool_output("compare_to_standard_threshold")
    percentile_output = _find_tool_output("compare_to_percentiles")
    if standard_output or percentile_output:
        comp_a, comp_b = st.columns(2)
        with comp_a:
            st.markdown("#### Standard-threshold comparison")
            if standard_output:
                st.json(standard_output)
            else:
                st.caption("Not applicable for this question.")
        with comp_b:
            st.markdown("#### MIMIC-IV percentile comparison")
            if percentile_output:
                st.json(percentile_output)
            else:
                st.caption("Not applicable for this question.")

    st.markdown("### Agent tool trace")
    if trace:
        trace_df = pd.DataFrame(
            [
                {
                    "step": entry.get("step"),
                    "tool_name": entry.get("tool_name"),
                    "inputs": json.dumps(entry.get("inputs", {}), default=str)[:300],
                    "outputs summary": entry.get("outputs_summary"),
                    "latency_ms": entry.get("latency_ms"),
                    "status": entry.get("status"),
                }
                for entry in trace
            ]
        )
        st.dataframe(trace_df, width="stretch", hide_index=True)
    else:
        st.info("No tools were called for this question.")

    st.markdown("### RAG sources used")
    sources_used = agent_result.get("sources_used", [])
    if sources_used:
        st.dataframe(pd.DataFrame({"source": sources_used}), width="stretch", hide_index=True)
    else:
        st.caption("No documentary sources were used for this question.")

    if agent_result.get("trace_file"):
        st.caption(f"Auditable trace saved to {agent_result['trace_file']}")

else:
    st.markdown("### Quick start")
    quick_col_1, quick_col_2 = st.columns(2)
    with quick_col_1:
        st.markdown(
            """
            <div class="card">
            <div class="section-label">What to ask</div>
            <p class="small-muted">Use the suggested clinical-context questions to test whether the top retrieved chunk matches the right vital sign, age group, and time window.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with quick_col_2:
        st.markdown(
            """
            <div class="card">
            <div class="section-label">How to read the output</div>
            <p class="small-muted">The left panel gives the grounded answer from qwen2.5:14b; the right panel lists the retrieved project sources; the chunk table shows metadata, semantic score, metadata bonuses, and the reranked final score.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("### Retrieval evaluation")
run_eval = st.button("Run / refresh retrieval evaluation")
if run_eval:
    with st.spinner("Running the strategy comparison benchmark..."):
        csv_path, md_path = run_retrieval_evaluation()
    st.success(f"Saved evaluation results to {csv_path} and {md_path}")

evaluation_summary_path = EVALUATION_DIR / "retrieval_summary.md"
evaluation_results_path = EVALUATION_DIR / "retrieval_evaluation.csv"
if evaluation_summary_path.exists():
    st.markdown(evaluation_summary_path.read_text(encoding="utf-8"))
    if evaluation_results_path.exists():
        evaluation_df = pd.read_csv(evaluation_results_path)
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
