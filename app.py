from __future__ import annotations

import pandas as pd
import streamlit as st

from src.semantic_rag import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    generate_grounded_answer,
    is_allowed_source_file,
    retrieve_semantic_chunks,
)


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
ask_clicked = run_col.button("Run retrieval", type="primary", use_container_width=True)
reset_clicked = reset_col.button("Reset", use_container_width=True)
if reset_clicked:
    st.session_state.question = sample_questions[0]
    st.session_state.last_answer = ""
    st.session_state.last_retrieved = []
    st.rerun()

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Mode", "Phase 1 Semantic RAG")
col_b.metric("Index", "ChromaDB")
col_c.metric("Embedding model", EMBEDDING_MODEL)
col_d.metric("LLM", LLM_MODEL)

st.caption("Source focus: MIMIC-IV summaries + project documents")

if ask_clicked:
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
                    "source_file": item.get("source_file"),
                    "source_type": item.get("source_type"),
                    "title": item.get("title"),
                    "score": round(float(item.get("similarity_score", 0.0)), 4),
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

    st.markdown("### Retrieved chunks")
    if retrieved:
        chunks_view = pd.DataFrame(
            [
                {
                    "rank": item.get("rank"),
                    "source_file": item.get("source_file"),
                    "source_type": item.get("source_type"),
                    "vital_sign": item.get("vital_sign"),
                    "age_group": item.get("age_group"),
                    "time_window": item.get("time_window"),
                    "distance": round(float(item.get("distance", 0.0)), 4),
                    "similarity_score": round(float(item.get("similarity_score", 0.0)), 4),
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
            <p class="small-muted">The left panel gives the grounded answer from qwen2.5:14b; the right panel lists the retrieved project sources; the chunk table shows metadata, distance, and preview text.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
