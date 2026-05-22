from __future__ import annotations

import logging
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
) -> list[dict[str, Any]]:
    client = build_ollama_client()
    collection = build_chroma_collection()

    embedding_response = client.embeddings.create(model=EMBEDDING_MODEL, input=question)
    query_embedding = embedding_response.data[0].embedding
    where = build_where_clause(
        source_type_filter=source_type_filter,
        vital_sign_filter=vital_sign_filter,
        age_group_filter=age_group_filter,
        time_window_filter=time_window_filter,
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0] or []
    metadatas = results.get("metadatas", [[]])[0] or []
    distances = results.get("distances", [[]])[0] or []
    ids = results.get("ids", [[]])[0] or []

    retrieved_chunks: list[dict[str, Any]] = []
    for index, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances), start=1):
        source_file = str(metadata.get("source_file", ""))
        if not is_allowed_source_file(source_file):
            LOGGER.warning("Skipping retrieved chunk from forbidden or non-project source: %s", source_file)
            continue
        display_source_file = normalize_display_source_file(source_file)

        distance_value = float(distance) if distance is not None else 0.0
        retrieved_chunks.append(
            {
                "rank": index,
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
                "similarity_score": 1.0 / (1.0 + distance_value),
                "chunk_preview": _chunk_preview(document),
            }
        )

    return retrieved_chunks


def build_grounded_prompt(question: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    context_blocks: list[str] = []
    for chunk in retrieved_chunks:
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
