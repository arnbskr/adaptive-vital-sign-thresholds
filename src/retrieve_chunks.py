from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .config import RAG_INDEX_DIR
from .rag_utils import (
    INTENT_CONCEPT,
    INTENT_DATASET,
    INTENT_PATIENT_VALUE,
    INTENT_PIPELINE,
    INTENT_VITAL_THRESHOLD,
    detect_query_intent,
    detect_threshold_condition,
    infer_age_from_query,
    infer_age_group_from_query,
    infer_direction_from_query,
    infer_time_window_from_query,
    infer_temperature_itemid_from_query,
    infer_vital_sign_from_query,
    query_mentions_vital_sign,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

KEYWORDS = [
    "heart rate",
    "tachycardia",
    "bradycardia",
    "spo2",
    "map",
    "percentile",
    "threshold",
    "icu",
    "mimic-iv",
    "elderly",
    "first 24h",
    "first 12h",
    "first 6h",
]


@lru_cache(maxsize=1)
def _load_index() -> tuple[pd.DataFrame, object, object]:
    chunks_path = RAG_INDEX_DIR / "chunks_index.csv"
    vectorizer_path = RAG_INDEX_DIR / "tfidf_vectorizer.joblib"
    matrix_path = RAG_INDEX_DIR / "tfidf_matrix.joblib"
    if not chunks_path.exists() or not vectorizer_path.exists() or not matrix_path.exists():
        raise FileNotFoundError(
            "RAG index not found. Build it first with python -m src.build_rag_index after preparing documents."
        )

    chunks_df = pd.read_csv(chunks_path)
    vectorizer = joblib.load(vectorizer_path)
    matrix = joblib.load(matrix_path)
    return chunks_df, vectorizer, matrix


def _normalize_filter(value: str | Iterable[str] | None) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value.lower() == "all" or not value.strip():
            return None
        return {value.lower()}
    normalized = {str(item).lower() for item in value if str(item).strip() and str(item).lower() != "all"}
    return normalized or None


def _keyword_bonus(query: str, chunk_text: str) -> float:
    query_lower = query.lower()
    chunk_lower = chunk_text.lower()
    matching = [keyword for keyword in KEYWORDS if keyword in query_lower and keyword in chunk_lower]
    return 0.10 if matching else 0.0


def _normalize_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _normalize_itemid(value: object) -> int | None:
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _vital_sign_matches(candidate: str, inferred: str | None) -> bool:
    if not inferred:
        return False
    candidate = candidate.strip().lower()
    inferred = inferred.strip().lower()
    if candidate == inferred:
        return True
    if inferred == "temperature":
        return candidate.startswith("temperature")
    if inferred == "blood pressure":
        return "blood pressure" in candidate
    if inferred == "map":
        return candidate == "map"
    return False


def _allowed_source_types(intent: str) -> set[str]:
    if intent in {INTENT_PATIENT_VALUE, INTENT_VITAL_THRESHOLD}:
        return {"mimic_stats", "project_report", "documentation", "article", "guideline"}
    if intent in {INTENT_CONCEPT, INTENT_DATASET, INTENT_PIPELINE}:
        return {"project_report", "documentation", "article", "guideline"}
    return {"mimic_stats", "project_report", "documentation", "article", "guideline"}


def _source_type_bonus(source_type: str, intent: str) -> float:
    if intent in {INTENT_PATIENT_VALUE, INTENT_VITAL_THRESHOLD}:
        if source_type == "mimic_stats":
            return 0.30
        if source_type in {"project_report", "documentation"}:
            return 0.08
        return 0.0
    if intent in {INTENT_CONCEPT, INTENT_DATASET, INTENT_PIPELINE}:
        if source_type in {"project_report", "documentation"}:
            return 0.30
        if source_type == "mimic_stats":
            return -0.40
        return 0.0
    return 0.0


def _source_type_penalty(source_type: str, intent: str) -> float:
    if intent in {INTENT_CONCEPT, INTENT_DATASET, INTENT_PIPELINE} and source_type == "mimic_stats":
        return 0.50
    if intent in {INTENT_PATIENT_VALUE, INTENT_VITAL_THRESHOLD} and source_type in {"project_report", "documentation"}:
        return 0.05
    return 0.0


def retrieve_chunks(
    query: str,
    top_k: int = 5,
    source_type_filter: str | Iterable[str] | None = None,
    vital_sign_filter: str | Iterable[str] | None = None,
    age_group_filter: str | Iterable[str] | None = None,
    time_window_filter: str | Iterable[str] | None = None,
) -> list[dict[str, object]]:
    chunks_df, vectorizer, matrix = _load_index()

    intent = detect_query_intent(query)
    inferred_age_group, has_precise_age = infer_age_group_from_query(query)
    inferred_time_window, has_precise_time_window = infer_time_window_from_query(query)
    inferred_vital_sign, has_precise_vital_sign = infer_vital_sign_from_query(query)
    inferred_temperature_itemid = infer_temperature_itemid_from_query(query)
    inferred_age = infer_age_from_query(query)
    inferred_direction = infer_direction_from_query(query)
    threshold_condition = detect_threshold_condition(query)
    queried_vital_present = query_mentions_vital_sign(query, inferred_vital_sign) if inferred_vital_sign else False

    working = chunks_df.copy()
    source_types = _normalize_filter(source_type_filter)
    vital_signs = _normalize_filter(vital_sign_filter)
    age_groups = _normalize_filter(age_group_filter)
    time_windows = _normalize_filter(time_window_filter)

    allowed_sources = _allowed_source_types(intent)
    working = working[working["source_type"].astype(str).str.lower().isin(allowed_sources)]

    if source_types is not None:
        working = working[working["source_type"].astype(str).str.lower().isin(source_types)]
    if vital_signs is not None:
        working = working[
            working["vital_sign"].astype(str).apply(
                lambda value: any(_vital_sign_matches(value, candidate) for candidate in vital_signs)
            )
        ]
    if age_groups is not None:
        working = working[working["age_group"].astype(str).str.lower().isin(age_groups)]
    if time_windows is not None:
        working = working[working["time_window"].astype(str).str.lower().isin(time_windows)]

    if intent in {INTENT_PATIENT_VALUE, INTENT_VITAL_THRESHOLD} and inferred_vital_sign:
        if intent == INTENT_PATIENT_VALUE:
            working = working[
                ~(
                    (working["source_type"].astype(str).str.lower() == "mimic_stats")
                    & ~working["vital_sign"].astype(str).apply(lambda value: _vital_sign_matches(value, inferred_vital_sign))
                )
            ]
        else:
            working = working[
                ~(
                    (working["source_type"].astype(str).str.lower() == "mimic_stats")
                    & ~working["vital_sign"].astype(str).apply(lambda value: _vital_sign_matches(value, inferred_vital_sign))
                )
            ]

    if intent in {INTENT_CONCEPT, INTENT_DATASET, INTENT_PIPELINE}:
        working = working[working["source_type"].astype(str).str.lower().isin({"project_report", "documentation", "article", "guideline"})]

    if working.empty:
        return []

    matching_vital_stats_found = False
    if inferred_vital_sign:
        matching_vital_stats_found = bool(
            (
                (chunks_df["source_type"].astype(str).str.lower() == "mimic_stats")
                & chunks_df["vital_sign"].astype(str).apply(lambda value: _vital_sign_matches(value, inferred_vital_sign))
            ).any()
        )

    candidate_indices = working.index.to_list()
    candidate_matrix = matrix[candidate_indices]
    query_vector = vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_vector, candidate_matrix).ravel()

    results: list[dict[str, object]] = []
    for candidate_index, tfidf_score in zip(candidate_indices, tfidf_scores):
        row = working.loc[candidate_index].to_dict()
        chunk_text = str(row.get("chunk_text", ""))
        metadata_boost = 0.0
        mismatch_penalty = 0.0

        chunk_age_group = _normalize_cell(row.get("age_group"))
        chunk_time_window = _normalize_cell(row.get("time_window"))
        chunk_vital_sign = _normalize_cell(row.get("vital_sign"))
        chunk_itemid = _normalize_itemid(row.get("itemid"))
        source_type = _normalize_cell(row.get("source_type"))

        if inferred_age_group:
            if chunk_age_group == inferred_age_group.lower():
                metadata_boost += 0.30
            elif has_precise_age and chunk_age_group:
                mismatch_penalty += 0.20

        if inferred_time_window:
            if chunk_time_window == inferred_time_window.lower():
                metadata_boost += 0.30
            elif has_precise_time_window and chunk_time_window:
                mismatch_penalty += 0.15

        if inferred_vital_sign:
            if _vital_sign_matches(chunk_vital_sign, inferred_vital_sign):
                metadata_boost += 0.35
            elif has_precise_vital_sign and chunk_vital_sign:
                mismatch_penalty += 0.35

        if inferred_temperature_itemid is not None and chunk_vital_sign == "temperature":
            if chunk_itemid == inferred_temperature_itemid:
                metadata_boost += 0.35
            elif has_precise_vital_sign:
                mismatch_penalty += 0.20

        if inferred_age is not None and inferred_age_group:
            metadata_boost += 0.02
        if threshold_condition and inferred_direction == "high" and row.get("standard_high") not in {None, ""}:
            metadata_boost += 0.03

        metadata_boost += _source_type_bonus(source_type, intent)
        mismatch_penalty += _source_type_penalty(source_type, intent)

        bonus = _keyword_bonus(query, chunk_text)
        row.update(
            {
                "tfidf_score": float(tfidf_score),
                "metadata_boost": metadata_boost,
                "keyword_bonus": bonus,
                "mismatch_penalty": mismatch_penalty,
                "final_score": float(tfidf_score) + metadata_boost + bonus - mismatch_penalty,
                "inferred_age_group": inferred_age_group,
                "inferred_time_window": inferred_time_window,
                "inferred_vital_sign": inferred_vital_sign,
                "inferred_direction": inferred_direction,
                "query_intent": intent,
                "threshold_condition": threshold_condition,
                "matching_vital_stats_found": matching_vital_stats_found,
            }
        )
        results.append(row)

    results = sorted(results, key=lambda item: item["final_score"], reverse=True)
    for rank, item in enumerate(results[:top_k], start=1):
        item["rank"] = rank
    return results[:top_k]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrieve top chunks for a query.")
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    for chunk in retrieve_chunks(args.query, top_k=args.top_k):
        print(chunk)
