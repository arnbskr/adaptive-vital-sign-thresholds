"""Tool: retrieve_project_context.

Thin, deterministic wrapper around the Phase 1 semantic RAG. It reuses
``src.semantic_rag.retrieve_semantic_chunks`` so the agent has a single way to
pull documentary context (concepts, dataset, pipeline) and supporting evidence
for patient questions. No new retrieval logic is introduced here.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..semantic_rag import retrieve_semantic_chunks


def retrieve_project_context(
    query: str,
    top_k: int = 5,
    source_type_filter: str | Iterable[str] | None = None,
    vital_sign_filter: str | Iterable[str] | None = None,
    age_group_filter: str | Iterable[str] | None = None,
    time_window_filter: str | Iterable[str] | None = None,
) -> dict[str, Any]:
    """Retrieve documentary context from the existing semantic index."""

    chunks = retrieve_semantic_chunks(
        query,
        top_k=top_k,
        source_type_filter=source_type_filter,
        vital_sign_filter=vital_sign_filter,
        age_group_filter=age_group_filter,
        time_window_filter=time_window_filter,
    )

    compact_chunks = [
        {
            "rank": chunk.get("rank"),
            "source_file": chunk.get("source_file"),
            "source_type": chunk.get("source_type"),
            "title": chunk.get("title"),
            "semantic_score": round(float(chunk.get("semantic_score", 0.0)), 4),
            "final_score": round(float(chunk.get("final_score", 0.0)), 4),
            "chunk_preview": chunk.get("chunk_preview"),
            "chunk_text": chunk.get("chunk_text"),
        }
        for chunk in chunks
    ]

    sources: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        key = (str(chunk.get("source_file", "")), str(chunk.get("source_type", "")))
        if key in seen or not key[0]:
            continue
        seen.add(key)
        sources.append({"source_file": key[0], "source_type": key[1]})

    first = chunks[0] if chunks else {}
    retrieval_info = {
        "question_intent": first.get("question_intent", "general_question"),
        "candidate_count_requested": first.get("candidate_count_requested"),
        "retrieval_latency_ms": first.get("retrieval_latency_ms"),
        "is_exact_match": bool(first.get("is_exact_match", False)),
        "returned": len(chunks),
    }

    return {"chunks": compact_chunks, "sources": sources, "retrieval_info": retrieval_info}
