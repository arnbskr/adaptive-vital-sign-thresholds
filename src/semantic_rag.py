from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Iterable

import chromadb
import pandas as pd
from openai import OpenAI

from .config import ROOT_DIR

LOGGER = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
EMBEDDING_MODEL = "bge-m3:latest"
LLM_MODEL = "qwen2.5:14b"
COLLECTION_NAME = "icu_rag"
CHROMA_DB_DIR = ROOT_DIR / "data" / "chroma_db"

PROJECT_SOURCE_ALLOWLIST = (
    "README.md",
    "Rapport_Final.pdf",
    "data/processed/vital_signs_elderly_icu_summary.csv",
    "data/rag_documents/rag_documents.csv",
    "R/",
)

FORBIDDEN_PATH_PARTS = (
    ".venv",
    "site-packages",
    "streamlit",
    "scipy",
    "__pycache__",
    ".git",
    "data/chroma_db",
    "data/rag_index",
    "data/rag_chunks",
    "node_modules",
)

VITAL_SIGN_PATTERNS = [
    ("Heart Rate", [r"\bhr\b", r"\bheart rate\b", r"\bpulse\b"]),
    ("Respiratory Rate", [r"\brr\b", r"\brespiratory rate\b"]),
    ("MAP", [r"\bmap\b", r"\bmean arterial pressure\b"]),
    ("Systolic Blood Pressure", [r"\bsbp\b", r"\bsystolic blood pressure\b"]),
    ("Diastolic Blood Pressure", [r"\bdbp\b", r"\bdiastolic blood pressure\b"]),
    ("Temperature", [r"\btemperature\b", r"\btemp\b"]),
    ("SpO2", [r"\bspo2\b", r"\boxygen saturation\b", r"\bsaturation\b"]),
]

TIME_WINDOW_PATTERNS = [
    ("first_6h", [r"\bfirst\s+6\s*hours?\b", r"\bfirst\s+6h\b", r"\b6h\b"]),
    ("first_12h", [r"\bfirst\s+12\s*hours?\b", r"\bfirst\s+12h\b", r"\b12h\b"]),
    ("first_24h", [r"\bfirst\s+24\s*hours?\b", r"\bfirst\s+24h\b", r"\b24h\b"]),
]


def _combine_where_clauses(clauses: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    active_clauses = [clause for clause in clauses if clause]
    if not active_clauses:
        return None
    if len(active_clauses) == 1:
        return active_clauses[0]
    return {"$and": active_clauses}


def _normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def _age_group_from_age(age: int | None) -> str:
    if age is None:
        return ""
    if age >= 85:
        return "85+"
    if age >= 75:
        return "75-84"
    if age >= 65:
        return "65-74"
    return ""


def _first_regex_group(patterns: list[str], text: str) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            groups = [group for group in match.groups() if group]
            if groups:
                return groups[0]
    return ""


def infer_patient_context(question: str) -> dict[str, Any]:
    normalized_question = _normalize_text(question)
    lower_question = normalized_question.lower()

    age_patterns = [
        r"\bage(?:d)?\s*(\d{1,3})\b",
        r"\b(?:patient\s+)?aged\s*(\d{1,3})\b",
        r"\b(\d{1,3})\s*(?:year[- ]old|yo)\b",
        r"\b(\d{1,3})-year-old\b",
    ]
    age_text = _first_regex_group(age_patterns, normalized_question)
    age = int(age_text) if age_text.isdigit() else None

    vital_sign = ""
    for canonical_name, patterns in VITAL_SIGN_PATTERNS:
        if any(re.search(pattern, lower_question, flags=re.IGNORECASE) for pattern in patterns):
            vital_sign = canonical_name
            break

    time_window = ""
    for canonical_name, patterns in TIME_WINDOW_PATTERNS:
        if any(re.search(pattern, lower_question, flags=re.IGNORECASE) for pattern in patterns):
            time_window = canonical_name
            break

    value_patterns = [
        r"\b(?:mean\s+)?(?:hr|heart rate|pulse|rr|respiratory rate|map|mean arterial pressure|sbp|systolic blood pressure|dbp|diastolic blood pressure|spo2|oxygen saturation|saturation|temperature|temp)\b[^0-9-]{0,24}(-?\d+(?:\.\d+)?)",
        r"\b(?:patient\s+)?(?:aged\s+\d{1,3}\s+with\s+)?(-?\d+(?:\.\d+)?)\s*(?:bpm|mmhg|%|c|°c)?\b",
    ]
    value_text = _first_regex_group(value_patterns, normalized_question)
    value = float(value_text) if value_text and re.fullmatch(r"-?\d+(?:\.\d+)?", value_text) else None

    age_group = _age_group_from_age(age)
    has_patient_context = bool(age_group and vital_sign and value is not None)

    return {
        "is_patient_value_question": has_patient_context,
        "age": age,
        "age_group": age_group,
        "vital_sign": vital_sign,
        "value": value,
        "time_window": time_window,
        "question_text": normalized_question,
    }


def detect_patient_value_question(question: str) -> bool:
    return bool(infer_patient_context(question)["is_patient_value_question"])


def _match_bonus_and_penalty(metadata: dict[str, Any], context: dict[str, Any]) -> tuple[float, float, bool, bool]:
    source_type = str(metadata.get("source_type", "")).strip()
    vital_sign = str(metadata.get("vital_sign", "")).strip()
    age_group = str(metadata.get("age_group", "")).strip()
    time_window = str(metadata.get("time_window", "")).strip()
    title = str(metadata.get("title", "")).strip()

    inferred_vital_sign = str(context.get("vital_sign", "")).strip()
    inferred_age_group = str(context.get("age_group", "")).strip()
    inferred_time_window = str(context.get("time_window", "")).strip()
    is_patient_question = bool(context.get("is_patient_value_question"))

    metadata_bonus = 0.0
    mismatch_penalty = 0.0

    exact_source_type = source_type == "mimic_stats"
    exact_vital_sign = bool(inferred_vital_sign) and vital_sign == inferred_vital_sign
    exact_age_group = bool(inferred_age_group) and age_group == inferred_age_group
    exact_time_window = bool(inferred_time_window) and time_window == inferred_time_window
    exact_match = bool(is_patient_question and exact_source_type and exact_vital_sign and exact_age_group and exact_time_window)

    if is_patient_question:
        if exact_source_type:
            metadata_bonus += 100.0
        if exact_vital_sign:
            metadata_bonus += 80.0
        if exact_age_group:
            metadata_bonus += 60.0
        if exact_time_window:
            metadata_bonus += 60.0
        if inferred_vital_sign and inferred_vital_sign.lower() in title.lower():
            metadata_bonus += 20.0

        if source_type in {"project_report", "documentation", "article", "guideline"}:
            mismatch_penalty += 50.0
        elif source_type and source_type != "mimic_stats":
            mismatch_penalty += 100.0

        if vital_sign and inferred_vital_sign and not exact_vital_sign:
            mismatch_penalty += 100.0
        if age_group and inferred_age_group and not exact_age_group:
            mismatch_penalty += 80.0
        if time_window and inferred_time_window and not exact_time_window:
            mismatch_penalty += 80.0

    return metadata_bonus, mismatch_penalty, exact_match, exact_source_type and (exact_vital_sign or exact_age_group or exact_time_window)


def _final_rank_key(chunk: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(chunk.get("priority_bucket", 0.0)),
        float(chunk.get("final_score", 0.0)),
        float(chunk.get("semantic_score", 0.0)),
        -float(chunk.get("distance", 0.0)),
    )


def build_ollama_client() -> OpenAI:
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def build_chroma_collection() -> Any:
    if not CHROMA_DB_DIR.exists():
        raise FileNotFoundError(
            "ChromaDB not found. Run python src/ingest.py to build the semantic vector database first."
        )

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError(
            "ChromaDB collection not found. Run python src/ingest.py to rebuild the semantic index."
        ) from exc


def _normalize_filter_values(value: str | Iterable[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized or normalized.lower() == "all":
            return None
        return [normalized]

    normalized = [str(item).strip() for item in value if str(item).strip() and str(item).lower() != "all"]
    return normalized or None


def build_where_clause(
    source_type_filter: str | Iterable[str] | None = None,
    vital_sign_filter: str | Iterable[str] | None = None,
    age_group_filter: str | Iterable[str] | None = None,
    time_window_filter: str | Iterable[str] | None = None,
) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []
    normalized_filters = {
        "source_type": _normalize_filter_values(source_type_filter),
        "vital_sign": _normalize_filter_values(vital_sign_filter),
        "age_group": _normalize_filter_values(age_group_filter),
        "time_window": _normalize_filter_values(time_window_filter),
    }

    for field_name, values in normalized_filters.items():
        if values:
            clauses.append({field_name: {"$in": values}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def is_allowed_source_file(source_file: str | None) -> bool:
    if not source_file:
        return False
    lowered = str(source_file).replace("\\", "/").lower()
    if any(part.lower() in lowered for part in FORBIDDEN_PATH_PARTS):
        return False
    return any(allowed.lower() in lowered for allowed in PROJECT_SOURCE_ALLOWLIST)


def normalize_display_source_file(source_file: str | None) -> str:
    if not source_file:
        return ""
    candidate = Path(str(source_file))
    if candidate.is_absolute():
        try:
            return str(candidate.relative_to(ROOT_DIR)).replace("\\", "/")
        except ValueError:
            return candidate.name
    return str(source_file).replace("\\", "/")


def _chunk_preview(text: str, max_length: int = 280) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[: max_length - 1].rstrip() + "…"


def retrieve_semantic_chunks(
    question: str,
    top_k: int = 5,
    source_type_filter: str | Iterable[str] | None = None,
    vital_sign_filter: str | Iterable[str] | None = None,
    age_group_filter: str | Iterable[str] | None = None,
    time_window_filter: str | Iterable[str] | None = None,
    apply_metadata_reranking: bool = True,
    apply_inferred_filters: bool = False,
) -> list[dict[str, Any]]:
    retrieval_start = time.perf_counter()
    client = build_ollama_client()
    collection = build_chroma_collection()
    context = infer_patient_context(question)

    embedding_response = client.embeddings.create(model=EMBEDDING_MODEL, input=question)
    query_embedding = embedding_response.data[0].embedding
    user_where = build_where_clause(
        source_type_filter=source_type_filter,
        vital_sign_filter=vital_sign_filter,
        age_group_filter=age_group_filter,
        time_window_filter=time_window_filter,
    )
    inferred_where = None
    if apply_inferred_filters and context.get("is_patient_value_question"):
        inferred_where = build_where_clause(
            source_type_filter="mimic_stats",
            vital_sign_filter=context.get("vital_sign") or None,
            age_group_filter=context.get("age_group") or None,
            time_window_filter=context.get("time_window") or None,
        )
    where = _combine_where_clauses([user_where, inferred_where])
    candidate_count = max(top_k * 5, 25)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_count,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0] or []
    metadatas = results.get("metadatas", [[]])[0] or []
    distances = results.get("distances", [[]])[0] or []
    ids = results.get("ids", [[]])[0] or []

    retrieved_chunks: list[dict[str, Any]] = []
    for semantic_rank, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances), start=1):
        source_file = str(metadata.get("source_file", ""))
        if not is_allowed_source_file(source_file):
            LOGGER.warning("Skipping retrieved chunk from forbidden or non-project source: %s", source_file)
            continue
        display_source_file = normalize_display_source_file(source_file)

        distance_value = float(distance) if distance is not None else 0.0
        similarity_score = 1.0 / (1.0 + distance_value)
        metadata_bonus, mismatch_penalty, exact_match, partial_match = _match_bonus_and_penalty(metadata, context)
        if apply_metadata_reranking and context.get("is_patient_value_question"):
            final_score = similarity_score + metadata_bonus - mismatch_penalty
            priority_bucket = 2.0 if exact_match else 1.0 if partial_match else 0.0
        else:
            final_score = similarity_score
            priority_bucket = 0.0

        retrieved_chunks.append(
            {
                "semantic_rank": semantic_rank,
                "chunk_id": doc_id,
                "source_file": display_source_file,
                "source_type": metadata.get("source_type", ""),
                "vital_sign": metadata.get("vital_sign", ""),
                "age_group": metadata.get("age_group", ""),
                "time_window": metadata.get("time_window", ""),
                "itemid": metadata.get("itemid", ""),
                "label": metadata.get("label", ""),
                "unitname": metadata.get("unitname", ""),
                "title": metadata.get("title", Path(source_file).name),
                "doc_id": metadata.get("doc_id", ""),
                "chunk_text": document,
                "distance": distance_value,
                "semantic_score": similarity_score,
                "metadata_bonus": metadata_bonus,
                "mismatch_penalty": mismatch_penalty,
                "final_score": final_score,
                "priority_bucket": priority_bucket,
                "is_exact_match": exact_match,
                "question_intent": "patient_value_question" if context.get("is_patient_value_question") else "general_question",
                "detected_age": context.get("age"),
                "detected_age_group": context.get("age_group", ""),
                "detected_vital_sign": context.get("vital_sign", ""),
                "detected_value": context.get("value"),
                "detected_time_window": context.get("time_window", ""),
                "candidate_count_requested": candidate_count,
                "retrieval_latency_ms": round((time.perf_counter() - retrieval_start) * 1000, 2),
                "chunk_preview": _chunk_preview(document),
            }
        )

    if context.get("is_patient_value_question") and apply_metadata_reranking:
        retrieved_chunks.sort(key=_final_rank_key, reverse=True)
    else:
        retrieved_chunks.sort(key=lambda item: (float(item.get("semantic_score", 0.0)), -float(item.get("distance", 0.0))), reverse=True)

    retrieval_latency_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)
    selected_chunks = retrieved_chunks[:top_k]
    for index, chunk in enumerate(selected_chunks, start=1):
        chunk["rank"] = index
        chunk["retrieval_latency_ms"] = retrieval_latency_ms

    LOGGER.info(
        "Retrieved %s chunks for intent=%s candidates=%s exact_matches=%s latency_ms=%.2f",
        len(selected_chunks),
        selected_chunks[0].get("question_intent", "general_question") if selected_chunks else "general_question",
        candidate_count,
        sum(1 for chunk in selected_chunks if chunk.get("is_exact_match")),
        retrieval_latency_ms,
    )

    return selected_chunks


def build_grounded_prompt(question: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    context = infer_patient_context(question)
    exact_chunks = [chunk for chunk in retrieved_chunks if chunk.get("is_exact_match")]
    prompt_chunks = exact_chunks if context.get("is_patient_value_question") and exact_chunks else retrieved_chunks

    context_blocks: list[str] = []
    for chunk in prompt_chunks:
        context_blocks.append(
            "\n".join(
                [
                    f"Source file: {chunk.get('source_file', '')}",
                    f"Source type: {chunk.get('source_type', '')}",
                    f"Vital sign: {chunk.get('vital_sign', '')}",
                    f"Age group: {chunk.get('age_group', '')}",
                    f"ICU time window: {chunk.get('time_window', '')}",
                    f"Item ID: {chunk.get('itemid', '')}",
                    f"Label: {chunk.get('label', '')}",
                    f"Unit: {chunk.get('unitname', '')}",
                    f"Distance: {chunk.get('distance', 0.0):.4f}",
                    "Context:",
                    str(chunk.get("chunk_text", "")),
                ]
            )
        )

    context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else "No context was retrieved."
    if context.get("is_patient_value_question"):
        ignored_chunks = [chunk for chunk in retrieved_chunks if not chunk.get("is_exact_match")]
        ignored_text = "\n\n".join(
            "\n".join(
                [
                    f"Source file: {chunk.get('source_file', '')}",
                    f"Source type: {chunk.get('source_type', '')}",
                    f"Vital sign: {chunk.get('vital_sign', '')}",
                    f"Age group: {chunk.get('age_group', '')}",
                    f"ICU time window: {chunk.get('time_window', '')}",
                    f"Final score: {chunk.get('final_score', 0.0):.3f}",
                ]
            )
            for chunk in ignored_chunks[:3]
        ) or "No conflicting chunks were retrieved."

        return f"""You are an academic assistant for the ICU Trajectory RAG Assistant project.
Answer in the same language as the user question.
This is a Phase 1 research aid only.
Do not provide clinical diagnosis or treatment recommendations.
Keep the answer grounded, concise, and structured.

Detected patient context:
- age: {context.get('age')}
- age_group: {context.get('age_group', '')}
- vital_sign: {context.get('vital_sign', '')}
- value: {context.get('value')}
- time_window: {context.get('time_window', '')}

Instruction:
Use the matching MIMIC-IV statistical summary for this vital sign, age group, and time window as the primary source.
Do not use summaries from other age groups, other vital signs, or other time windows to make the comparison.
If the exact matching summary is unavailable, say that the exact summary is unavailable.
Do not mention age-group or time-window mismatches as evidence for the direct comparison.

Primary retrieved context:
{context_text}

Retrieved chunks not used for the direct comparison:
{ignored_text}

Question:
{question}
"""

    return f"""You are an academic assistant for the ICU Trajectory RAG Assistant project.
Answer in the same language as the user question.
Use only the retrieved context below.
If the context is insufficient, say so explicitly.
Do not provide clinical diagnosis, treatment recommendations, or unsupported claims.
Keep the answer grounded, concise, and structured.

Question:
{question}

Retrieved context:
{context_text}
"""


def generate_grounded_answer(question: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    client = build_ollama_client()
    prompt = build_grounded_prompt(question, retrieved_chunks)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content
