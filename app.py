from __future__ import annotations

import pandas as pd
import streamlit as st

from src.generate_rag_answer import generate_rag_answer
from src.retrieve_chunks import retrieve_chunks
from src.rag_utils import detect_query_intent, detect_threshold_condition, infer_direction_from_query, infer_time_window_from_query, infer_vital_sign_from_query, infer_age_group_from_query


st.set_page_config(page_title="ICU Trajectory RAG Assistant", page_icon="ICU", layout="wide")

st.title("ICU Trajectory RAG Assistant")
st.caption("Phase 1: local RAG only, with TF-IDF retrieval over MIMIC-IV summaries and project documents.")

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

st.sidebar.header("Controls")
sample_choice = st.sidebar.selectbox("Suggested questions", ["Custom"] + sample_questions)
if sample_choice != "Custom":
    st.session_state.question = sample_choice

question = st.text_area("Question", value=st.session_state.question, height=120)
st.session_state.question = question

top_k = st.slider("Top-k chunks", min_value=3, max_value=10, value=5, step=1)
source_type_filter = st.selectbox(
    "Source type filter",
    ["All", "mimic_stats", "project_report", "documentation", "article", "guideline"],
)
vital_sign_filter = st.selectbox(
    "Vital sign filter",
    [
        "All",
        "Heart Rate",
        "Respiratory Rate",
        "MAP",
        "Systolic Blood Pressure",
        "Diastolic Blood Pressure",
        "Temperature",
        "SpO2",
    ],
)
age_group_filter = st.selectbox(
    "Age group filter",
    ["All", "65-74", "75-84", "85+"],
)
time_window_filter = st.selectbox(
    "Time window filter",
    ["All", "first_6h", "first_12h", "first_24h"],
)

ask_clicked = st.button("Ask", type="primary")

if ask_clicked:
    detected_intent = detect_query_intent(question)
    inferred_age_group = infer_age_group_from_query(question)[0]
    inferred_time_window = infer_time_window_from_query(question)[0]
    inferred_vital_sign = infer_vital_sign_from_query(question)[0]
    inferred_direction = infer_direction_from_query(question)
    threshold_condition = detect_threshold_condition(question)

    with st.spinner("Retrieving local chunks..."):
        try:
            retrieved = retrieve_chunks(
                question,
                top_k=top_k,
                source_type_filter=source_type_filter,
                vital_sign_filter=vital_sign_filter,
                age_group_filter=age_group_filter,
                time_window_filter=time_window_filter,
            )
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()

    result = generate_rag_answer(question, retrieved, use_llm=False)

    answer_col, sources_col = st.columns([2, 1])

    with answer_col:
        st.subheader("Answer")
        st.markdown(result["answer"])
        st.caption(
            f"Detected intent: {detected_intent} | effective_intent={result.get('effective_intent')} | age_group={inferred_age_group} | time_window={inferred_time_window} | "
            f"vital_sign={inferred_vital_sign} | direction={inferred_direction} | threshold_condition={threshold_condition}"
        )

    with sources_col:
        st.subheader("Sources used")
        if result["sources"]:
            st.dataframe(pd.DataFrame(result["sources"]), use_container_width=True, hide_index=True)
        else:
            st.info("No sources available for the current query.")

    st.subheader("Retrieved chunks")
    if retrieved:
        chunks_view = pd.DataFrame(
            [
                {
                    "rank": item.get("rank"),
                    "chunk_id": item.get("chunk_id"),
                    "doc_id": item.get("doc_id"),
                    "final_score": item.get("final_score"),
                    "metadata_boost": item.get("metadata_boost"),
                    "mismatch_penalty": item.get("mismatch_penalty"),
                    "tfidf_score": item.get("tfidf_score"),
                    "keyword_bonus": item.get("keyword_bonus"),
                    "source_type": item.get("source_type"),
                    "vital_sign": item.get("vital_sign"),
                    "itemid": item.get("itemid"),
                    "label": item.get("label"),
                    "unitname": item.get("unitname"),
                    "age_group": item.get("age_group"),
                    "time_window": item.get("time_window"),
                    "section": item.get("section"),
                    "title": item.get("title"),
                    "query_intent": item.get("query_intent"),
                    "inferred_age_group": item.get("inferred_age_group"),
                    "inferred_time_window": item.get("inferred_time_window"),
                    "inferred_vital_sign": item.get("inferred_vital_sign"),
                    "inferred_direction": item.get("inferred_direction"),
                    "chunk_text": item.get("chunk_text"),
                }
                for item in retrieved
            ]
        )
        st.dataframe(chunks_view, use_container_width=True, hide_index=True)
    else:
        st.info("No chunks were retrieved for this query.")

    st.subheader("Clinical framing")
    st.write(
        "This response is a research aid for interpretation only and must not be used as a clinical decision. "
        "It distinguishes standard thresholds from observed MIMIC-IV summaries and flags the current evidence limits."
    )
