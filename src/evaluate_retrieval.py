from __future__ import annotations

import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .config import EVALUATION_DIR, ensure_data_directories
from .semantic_rag import build_chroma_collection, infer_patient_context, retrieve_semantic_chunks

TOP_K = 5

EVALUATION_QUESTIONS: list[dict[str, Any]] = [
    {
        "question": "For a patient aged 82 with mean HR 104 bpm in the first 24h ICU stay, is this value high?",
        "expected_vital_sign": "Heart Rate",
        "expected_age_group": "75-84",
        "expected_time_window": "first_24h",
        "expected_source_type": "mimic_stats",
    },
    {
        "question": "For a patient aged 78 with MAP 62 mmHg in the first 24h ICU stay, is this value low?",
        "expected_vital_sign": "MAP",
        "expected_age_group": "75-84",
        "expected_time_window": "first_24h",
        "expected_source_type": "mimic_stats",
    },
    {
        "question": "For a patient aged 80 with SpO2 90% in the first 24h ICU stay, is this low?",
        "expected_vital_sign": "SpO2",
        "expected_age_group": "75-84",
        "expected_time_window": "first_24h",
        "expected_source_type": "mimic_stats",
    },
    {
        "question": "What is the difference between a standard clinical threshold and an adaptive percentile-based threshold?",
        "expected_source_type": "project_report",
    },
    {
        "question": "Which MIMIC-IV tables are useful for ICU vital signs?",
        "expected_source_type": "project_report",
    },
]

STRATEGY_METADATA = {
    "semantic_only": {
        "name": "Semantic only",
        "notes": "ChromaDB similarity search with bge-m3 embeddings and no reranking.",
        "cost": "Medium",
        "latency": "Medium",
        "complexity": "Low",
    },
    "semantic_reranked": {
        "name": "Semantic + metadata reranking",
        "notes": "Similarity search plus strict metadata-aware reranking for patient-value questions.",
        "cost": "Medium",
        "latency": "Medium",
        "complexity": "Medium",
    },
    "keyword_baseline": {
        "name": "Keyword / lexical baseline",
        "notes": "Simple lexical scoring over indexed documents only; no semantic embedding at retrieval time.",
        "cost": "Very low",
        "latency": "Low",
        "complexity": "Low",
    },
    "metadata_filtered": {
        "name": "Semantic + metadata filtering",
        "notes": "Inference-guided metadata filters before reranking; high precision when metadata is clean.",
        "cost": "Medium",
        "latency": "Medium",
        "complexity": "Medium-high",
    },
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "between",
    "by",
    "for",
    "from",
    "high",
    "how",
    "in",
    "is",
    "it",
    "low",
    "mean",
    "of",
    "on",
    "or",
    "patient",
    "patient",
    "patients",
    "should",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "which",
    "with",
}


def _normalize_tokens(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(text).lower())
    return [token for token in cleaned.split() if token and token not in STOPWORDS]


def _to_bool(value: Any) -> bool:
    return bool(value) and str(value).lower() not in {"", "none", "false", "0"}


def _exact_match(item: dict[str, Any], expected: dict[str, Any]) -> bool:
    if expected.get("expected_source_type") and item.get("source_type") != expected.get("expected_source_type"):
        return False
    if expected.get("expected_vital_sign") and item.get("vital_sign") != expected.get("expected_vital_sign"):
        return False
    if expected.get("expected_age_group") and item.get("age_group") != expected.get("expected_age_group"):
        return False
    if expected.get("expected_time_window") and item.get("time_window") != expected.get("expected_time_window"):
        return False
    return True


def _top1_match_flags(retrieved: list[dict[str, Any]], expected: dict[str, Any]) -> dict[str, bool]:
    top1 = retrieved[0] if retrieved else {}
    return {
        "top1_source_type_match": bool(expected.get("expected_source_type")) and top1.get("source_type") == expected.get("expected_source_type"),
        "top1_vital_sign_match": bool(expected.get("expected_vital_sign")) and top1.get("vital_sign") == expected.get("expected_vital_sign"),
        "top1_age_group_match": bool(expected.get("expected_age_group")) and top1.get("age_group") == expected.get("expected_age_group"),
        "top1_time_window_match": bool(expected.get("expected_time_window")) and top1.get("time_window") == expected.get("expected_time_window"),
    }


def _exact_metadata_scores(retrieved: list[dict[str, Any]], expected: dict[str, Any], top_k: int = TOP_K) -> dict[str, bool]:
    if not expected.get("expected_source_type") and not expected.get("expected_vital_sign"):
        return {"exact_metadata_match_at_1": False, "exact_metadata_match_at_k": False}

    top1 = retrieved[0] if retrieved else {}
    return {
        "exact_metadata_match_at_1": _exact_match(top1, expected),
        "exact_metadata_match_at_k": any(_exact_match(item, expected) for item in retrieved[:top_k]),
    }


def _patient_question(expected: dict[str, Any]) -> bool:
    return bool(expected.get("expected_vital_sign")) and bool(expected.get("expected_age_group"))


def _semantic_only(question: str, top_k: int = TOP_K) -> tuple[list[dict[str, Any]], int]:
    retrieved = retrieve_semantic_chunks(
        question,
        top_k=top_k,
        apply_metadata_reranking=False,
    )
    candidates = retrieved[0].get("candidate_count_requested", max(top_k * 5, 25)) if retrieved else max(top_k * 5, 25)
    return retrieved, int(candidates)


def _semantic_reranked(question: str, top_k: int = TOP_K) -> tuple[list[dict[str, Any]], int]:
    retrieved = retrieve_semantic_chunks(question, top_k=top_k)
    candidates = retrieved[0].get("candidate_count_requested", max(top_k * 5, 25)) if retrieved else max(top_k * 5, 25)
    return retrieved, int(candidates)


def _metadata_filtered(question: str, top_k: int = TOP_K) -> tuple[list[dict[str, Any]], int]:
    retrieved = retrieve_semantic_chunks(
        question,
        top_k=top_k,
        apply_metadata_reranking=True,
        apply_inferred_filters=True,
    )
    candidates = retrieved[0].get("candidate_count_requested", max(top_k * 5, 25)) if retrieved else max(top_k * 5, 25)
    return retrieved, int(candidates)


def _keyword_baseline(question: str, top_k: int = TOP_K) -> tuple[list[dict[str, Any]], int]:
    collection = build_chroma_collection()
    corpus = collection.get(include=["documents", "metadatas"])
    query_tokens = Counter(_normalize_tokens(question))
    retrieved: list[dict[str, Any]] = []

    for index, (doc_id, document, metadata) in enumerate(
        zip(corpus.get("ids", []), corpus.get("documents", []), corpus.get("metadatas", [])),
        start=1,
    ):
        document_text = str(document or "")
        document_tokens = Counter(_normalize_tokens(document_text))
        overlap = sum(min(query_tokens[token], document_tokens[token]) for token in query_tokens if token in document_tokens)
        token_boost = 0.0
        if document_tokens:
            token_boost = overlap / max(1, len(document_tokens))
        retrieved.append(
            {
                "semantic_rank": index,
                "chunk_id": doc_id,
                "source_file": str(metadata.get("source_file", "")),
                "source_type": str(metadata.get("source_type", "")),
                "vital_sign": str(metadata.get("vital_sign", "")),
                "age_group": str(metadata.get("age_group", "")),
                "time_window": str(metadata.get("time_window", "")),
                "itemid": str(metadata.get("itemid", "")),
                "label": str(metadata.get("label", "")),
                "unitname": str(metadata.get("unitname", "")),
                "title": str(metadata.get("title", "")),
                "chunk_text": document_text,
                "distance": 0.0,
                "semantic_score": float(token_boost),
                "metadata_bonus": 0.0,
                "mismatch_penalty": 0.0,
                "final_score": float(overlap + token_boost),
                "priority_bucket": 0.0,
                "is_exact_match": False,
                "question_intent": "keyword_baseline",
                "candidate_count_requested": len(corpus.get("documents", [])),
                "retrieval_latency_ms": 0.0,
                "chunk_preview": document_text[:280],
            }
        )

    retrieved.sort(key=lambda item: (float(item.get("final_score", 0.0)), float(item.get("semantic_score", 0.0))), reverse=True)
    selected = retrieved[:top_k]
    for rank, chunk in enumerate(selected, start=1):
        chunk["rank"] = rank
    return selected, len(corpus.get("documents", []))


def _run_strategy(strategy_name: str, question: str, top_k: int = TOP_K) -> tuple[list[dict[str, Any]], int]:
    if strategy_name == "semantic_only":
        return _semantic_only(question, top_k=top_k)
    if strategy_name == "semantic_reranked":
        return _semantic_reranked(question, top_k=top_k)
    if strategy_name == "metadata_filtered":
        return _metadata_filtered(question, top_k=top_k)
    if strategy_name == "keyword_baseline":
        return _keyword_baseline(question, top_k=top_k)
    raise ValueError(f"Unknown strategy: {strategy_name}")


def _metric_row(
    strategy_name: str,
    evaluation_item: dict[str, Any],
    retrieved: list[dict[str, Any]],
    candidate_count: int,
) -> dict[str, Any]:
    question = evaluation_item["question"]
    expected = dict(evaluation_item)
    patient_question = _patient_question(expected)
    top1_flags = _top1_match_flags(retrieved, expected)
    exact_flags = _exact_metadata_scores(retrieved, expected)
    top1 = retrieved[0] if retrieved else {}

    return {
        "strategy": strategy_name,
        "question": question,
        "question_type": "patient_value_question" if patient_question else "general_question",
        "expected_source_type": expected.get("expected_source_type", ""),
        "expected_vital_sign": expected.get("expected_vital_sign", ""),
        "expected_age_group": expected.get("expected_age_group", ""),
        "expected_time_window": expected.get("expected_time_window", ""),
        "top1_source_file": top1.get("source_file", ""),
        "top1_source_type": top1.get("source_type", ""),
        "top1_vital_sign": top1.get("vital_sign", ""),
        "top1_age_group": top1.get("age_group", ""),
        "top1_time_window": top1.get("time_window", ""),
        **top1_flags,
        **exact_flags,
        "retrieval_latency_ms": float(top1.get("retrieval_latency_ms", 0.0)) if top1 else 0.0,
        "number_of_candidates_retrieved": candidate_count,
        "top_k": TOP_K,
    }


def _aggregate_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {
            "top1_accuracy": 0.0,
            "exact_metadata_match_at_1": 0.0,
            "exact_metadata_match_at_k": 0.0,
            "average_latency_ms": 0.0,
            "precision_proxy": 0.0,
            "recall_proxy": 0.0,
        }

    patient_rows = rows[rows["question_type"] == "patient_value_question"]
    general_rows = rows[rows["question_type"] != "patient_value_question"]
    precision_proxy = float(patient_rows["exact_metadata_match_at_1"].mean()) if not patient_rows.empty else 0.0
    recall_proxy = 0.0

    if not patient_rows.empty:
        recall_proxy = float(patient_rows["exact_metadata_match_at_k"].mean())
    if not general_rows.empty:
        source_precision = float(general_rows["top1_source_type_match"].mean())
        precision_proxy = (precision_proxy + source_precision) / 2.0 if patient_rows.empty is False else source_precision
        recall_proxy = max(recall_proxy, source_precision)

    return {
        "top1_accuracy": float(rows["top1_source_type_match"].mean()),
        "exact_metadata_match_at_1": float(patient_rows["exact_metadata_match_at_1"].mean()) if not patient_rows.empty else 0.0,
        "exact_metadata_match_at_k": float(patient_rows["exact_metadata_match_at_k"].mean()) if not patient_rows.empty else 0.0,
        "average_latency_ms": float(rows["retrieval_latency_ms"].mean()),
        "precision_proxy": precision_proxy,
        "recall_proxy": recall_proxy,
    }


def _format_percentage(value: float) -> str:
    return f"{value * 100:.0f}%"


def _format_latency(value: float) -> str:
    return f"{value:.1f} ms"


def _summary_row(strategy_key: str, metrics: dict[str, Any]) -> str:
    metadata = STRATEGY_METADATA[strategy_key]
    return "| {name} | {precision} | {recall} | {latency} | {cost} | {complexity} | {notes} |".format(
        name=metadata["name"],
        precision=_format_percentage(metrics["precision_proxy"]),
        recall=_format_percentage(metrics["recall_proxy"]),
        latency=_format_latency(metrics["average_latency_ms"]),
        cost=metadata["cost"],
        complexity=metadata["complexity"],
        notes=metadata["notes"],
    )


def run_retrieval_evaluation(output_dir: Path | None = None) -> tuple[Path, Path]:
    ensure_data_directories()
    target_dir = output_dir or EVALUATION_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []

    for strategy_key in STRATEGY_METADATA:
        strategy_start = time.perf_counter()
        strategy_records: list[dict[str, Any]] = []
        for item in EVALUATION_QUESTIONS:
            retrieved, candidate_count = _run_strategy(strategy_key, item["question"], top_k=TOP_K)
            strategy_records.append(_metric_row(strategy_key, item, retrieved, candidate_count))
        strategy_latency_ms = (time.perf_counter() - strategy_start) * 1000
        frame = pd.DataFrame(strategy_records)
        metrics = _aggregate_metrics(frame)
        metrics["strategy"] = strategy_key
        metrics["strategy_latency_ms"] = strategy_latency_ms
        aggregate_rows.append(metrics)
        all_rows.extend(strategy_records)

    results_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(aggregate_rows)

    csv_path = target_dir / "retrieval_evaluation.csv"
    md_path = target_dir / "retrieval_summary.md"
    results_df.to_csv(csv_path, index=False)
    summary_df.to_csv(target_dir / "retrieval_summary.csv", index=False)

    summary_lines = [
        "# Retrieval Strategy Comparison",
        "",
        "This evaluation is intentionally small and Phase 1 only. It compares semantic retrieval, semantic retrieval with strict metadata-aware reranking, a lexical baseline, and an optional metadata-filtered variant.",
        "",
        "| Strategy | Precision proxy | Recall proxy | Latency | Cost | Complexity | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in aggregate_rows:
        summary_lines.append(_summary_row(row["strategy"], row))

    summary_lines.extend(
        [
            "",
            "## Table Handling",
            "",
            "The ICU vital-sign summary CSV is treated as a table-aware source: one CSV row becomes one retrievable RAG document so the age group, vital sign, time window, and threshold statistics stay aligned.",
            "",
            "## Interpretation",
            "",
            "- Lexical baseline: very low cost and low latency, but weak on synonyms and phrasing differences.",
            "- Semantic retrieval: better recall because embeddings can match paraphrases and related concepts.",
            "- Metadata-aware reranking: improves precision for patient-value questions because the age group, vital sign, and time window are explicitly prioritized.",
            "- Metadata filtering: highest precision when metadata is clean, but it can lose recall if metadata is missing or incomplete.",
        ]
    )
    md_path.write_text("\n".join(summary_lines), encoding="utf-8")
    return csv_path, md_path


def main() -> None:
    csv_path, md_path = run_retrieval_evaluation()
    print(f"Saved evaluation CSV to {csv_path}")
    print(f"Saved evaluation summary to {md_path}")


if __name__ == "__main__":
    main()
