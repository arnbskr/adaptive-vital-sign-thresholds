from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from .config import ROOT_DIR

KEYWORD_TO_VITAL_SIGN = [
    ("heart rate", "Heart Rate"),
    ("fréquence cardiaque", "Heart Rate"),
    ("frequence cardiaque", "Heart Rate"),
    ("bpm", "Heart Rate"),
    ("spo2", "SpO2"),
    ("oxygen saturation", "SpO2"),
    ("saturation", "SpO2"),
    ("respiratory rate", "Respiratory Rate"),
    ("fréquence respiratoire", "Respiratory Rate"),
    ("frequence respiratoire", "Respiratory Rate"),
    ("rr", "Respiratory Rate"),
    ("mean arterial pressure", "MAP"),
    ("pression artérielle moyenne", "MAP"),
    ("pression arterielle moyenne", "MAP"),
    ("map", "MAP"),
    ("systolic", "Systolic Blood Pressure"),
    ("sbp", "Systolic Blood Pressure"),
    ("pression systolique", "Systolic Blood Pressure"),
    ("diastolic", "Diastolic Blood Pressure"),
    ("dbp", "Diastolic Blood Pressure"),
    ("pression diastolique", "Diastolic Blood Pressure"),
    ("temperature", "Temperature"),
    ("température", "Temperature"),
]

SOURCE_TYPE_KEYWORDS = [
    ("guideline", "guideline"),
    ("guide", "guideline"),
    ("article", "article"),
    ("paper", "article"),
    ("study", "article"),
    ("report", "project_report"),
    ("readme", "project_report"),
    ("documentation", "documentation"),
]

INTENT_PATIENT_VALUE = "patient_value_question"
INTENT_VITAL_THRESHOLD = "vital_threshold_question"
INTENT_CONCEPT = "concept_question"
INTENT_DATASET = "dataset_question"
INTENT_PIPELINE = "pipeline_question"
INTENT_MISSING = "unsupported_or_missing_vital_question"

VITAL_KEYWORDS = {
    "Heart Rate": ["heart rate", "fréquence cardiaque", "frequence cardiaque", "hr", "bpm"],
    "Respiratory Rate": ["respiratory rate", "fréquence respiratoire", "frequence respiratoire", "rr"],
    "MAP": ["map", "mean arterial pressure", "pression artérielle moyenne", "pression arterielle moyenne"],
    "Systolic Blood Pressure": ["systolic", "sbp", "pression systolique"],
    "Diastolic Blood Pressure": ["diastolic", "dbp", "pression diastolique"],
    "Temperature": ["temperature", "température", "temperature celsius", "temperature fahrenheit"],
    "SpO2": ["spo2", "oxygen saturation", "o2 saturation", "saturation"],
}


def relative_source_path(path: Path) -> str:
    return str(path.relative_to(ROOT_DIR)) if path.is_relative_to(ROOT_DIR) else str(path)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _contains_any(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def detect_query_intent(query: str) -> str:
    lowered = query.lower()

    if _contains_any(lowered, ["rag_documents", "rag_chunks", "rag_index", "bigquery live", "streamlit", "pipeline", "processed", "chunk", "index"]):
        return INTENT_PIPELINE

    if _contains_any(lowered, [
        "which mimic-iv tables",
        "which tables are used",
        "where do the vital sign values come from",
        "data source",
        "table",
        "tables",
        "d_items",
        "hosp.patients",
        "icu.icustays",
        "icu.chartevents",
        "icu.d_items",
    ]):
        return INTENT_DATASET

    if _contains_any(lowered, ["difference between", "what is the difference", "why use", "why are adaptive", "adaptive percentile", "percentile-based threshold", "concept", "conceptual", "standard clinical threshold"]):
        return INTENT_CONCEPT

    if _contains_any(lowered, ["is this value", "is this high", "is this low", "is this elevated", "is this abnormal", "compare", "in the first", "aged ", "year-old", "ans"]):
        return INTENT_PATIENT_VALUE

    if _contains_any(lowered, ["above", "below", "higher than", "lower than", "threshold", "interpreted", "interpret", "bradycardia", "tachycardia", "hypotension", "hypoxemia", "elevated"]):
        return INTENT_VITAL_THRESHOLD

    return INTENT_CONCEPT


def infer_vital_sign(text: str) -> str:
    lowered = text.lower()
    for keyword, label in KEYWORD_TO_VITAL_SIGN:
        if keyword in lowered:
            return label
    if "hr" in lowered and "heart rate" in lowered:
        return "Heart Rate"
    return "General"


def infer_direction_from_query(query: str) -> str:
    lowered = query.lower()
    high_markers = ["high", "elevated", "above", "supérieur", "superieur", "haute", "élevé", "eleve", "tachycardia", "hypertension"]
    low_markers = ["low", "below", "inférieur", "inferieur", "basse", "bradycardia", "hypotension", "hypoxemia", "hypoxémie"]
    if _contains_any(lowered, high_markers):
        return "high"
    if _contains_any(lowered, low_markers):
        return "low"
    return "neutral"


def detect_threshold_condition(query: str) -> bool:
    lowered = query.lower()
    return bool(
        re.search(r"\babove\s+\d+(?:\.\d+)?", lowered)
        or re.search(r"\bbelow\s+\d+(?:\.\d+)?", lowered)
        or re.search(r"\bgreater than\s+\d+(?:\.\d+)?", lowered)
        or re.search(r"\bless than\s+\d+(?:\.\d+)?", lowered)
        or re.search(r"\bmore than\s+\d+(?:\.\d+)?", lowered)
        or re.search(r"\bless than\s+\d+(?:\.\d+)?", lowered)
        or re.search(r"\bunder\s+\d+(?:\.\d+)?", lowered)
        or re.search(r"\bover\s+\d+(?:\.\d+)?", lowered)
        or re.search(r"\bmoins de\s+\d+(?:\.\d+)?", lowered)
        or re.search(r"\bplus de\s+\d+(?:\.\d+)?", lowered)
    )


def infer_age_group_from_query(query: str) -> tuple[str | None, bool]:
    lowered = query.lower()
    patterns = [
        r"(?:aged|age|patient aged|patient age)\s*(\d{2,3})",
        r"(\d{2,3})\s*(?:-?year-old|years? old)",
        r"(?:patient de|de)\s*(\d{2,3})\s*ans",
        r"(\d{2,3})\s*ans",
    ]
    age_value: int | None = None
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            age_value = int(match.group(1))
            break
    if age_value is None:
        return None, False
    if age_value >= 85:
        return "85+", True
    if age_value >= 75:
        return "75-84", True
    if age_value >= 65:
        return "65-74", True
    return None, False


def infer_age_from_query(query: str) -> int | None:
    lowered = query.lower()
    patterns = [
        r"(?:aged|age|patient aged|patient age)\s*(\d{2,3})",
        r"(\d{2,3})\s*(?:-?year-old|years? old)",
        r"(?:patient de|de)\s*(\d{2,3})\s*ans",
        r"(\d{2,3})\s*ans",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return int(match.group(1))
    return None


def infer_time_window_from_query(query: str) -> tuple[str | None, bool]:
    lowered = query.lower()
    if re.search(r"(?:first|premi[eè]res?)\s*6\s*h(?:ours?)?", lowered):
        return "first_6h", True
    if re.search(r"(?:first|premi[eè]res?)\s*12\s*h(?:ours?)?", lowered):
        return "first_12h", True
    if re.search(r"(?:first|premi[eè]res?)\s*24\s*h(?:ours?)?", lowered):
        return "first_24h", True
    return None, False


def infer_vital_sign_from_query(query: str) -> tuple[str | None, bool]:
    lowered = query.lower()
    priority_patterns = [
        (r"(?:heart rate|fréquence cardiaque|frequence cardiaque|\bhr\b|\bbpm\b)", "Heart Rate"),
        (r"(?:respiratory rate|fréquence respiratoire|frequence respiratoire|\brr\b)", "Respiratory Rate"),
        (r"(?:mean arterial pressure|pression artérielle moyenne|pression arterielle moyenne|\bmap\b)", "MAP"),
        (r"(?:systolic|\bsbp\b|pression systolique)", "Systolic Blood Pressure"),
        (r"(?:diastolic|\bdbp\b|pression diastolique)", "Diastolic Blood Pressure"),
        (r"(?:temperature|température|temperature celsius|temperature fahrenheit)", "Temperature"),
        (r"(?:spo2|oxygen saturation|o2 saturation|saturation)", "SpO2"),
    ]
    for pattern, label in priority_patterns:
        if re.search(pattern, lowered):
            return label, True
    return None, False


def infer_temperature_itemid_from_query(query: str) -> int | None:
    lowered = query.lower()
    if re.search(r"(?:°\s*c|\bcelsius\b|\bcentigrade\b)", lowered):
        return 223762
    if re.search(r"(?:°\s*f|\bfahrenheit\b)", lowered):
        return 223761
    return None


def query_mentions_vital_sign(query: str, vital_sign: str) -> bool:
    return _contains_any(query.lower(), VITAL_KEYWORDS.get(vital_sign, []))


def query_contains_patient_context(query: str) -> bool:
    lowered = query.lower()
    return bool(
        infer_age_from_query(query) is not None
        or _contains_any(lowered, ["first 6h", "first 12h", "first 24h", "premières 6h", "premières 12h", "premières 24h"])
        or infer_vital_sign_from_query(query)[0] is not None
    )


def extract_vital_value_from_query(query: str, vital_sign: str | None) -> float | None:
    lowered = query.lower()
    patterns_by_sign = {
        "Heart Rate": [r"(?:heart rate|fréquence cardiaque|frequence cardiaque|\bhr\b|\bbpm\b)[^\d]{0,20}(\d+(?:\.\d+)?)"],
        "Respiratory Rate": [r"(?:respiratory rate|fréquence respiratoire|frequence respiratoire|\brr\b)[^\d]{0,20}(\d+(?:\.\d+)?)"],
        "MAP": [r"(?:mean arterial pressure|pression artérielle moyenne|pression arterielle moyenne|\bmap\b)[^\d]{0,20}(\d+(?:\.\d+)?)", r"(\d+(?:\.\d+)?)\s*mmhg"],
        "Systolic Blood Pressure": [r"(?:systolic|\bsbp\b|pression systolique)[^\d]{0,20}(\d+(?:\.\d+)?)"],
        "Diastolic Blood Pressure": [r"(?:diastolic|\bdbp\b|pression diastolique)[^\d]{0,20}(\d+(?:\.\d+)?)"],
        "Temperature": [r"(?:temperature|température|temperature celsius|temperature fahrenheit)[^\d]{0,20}(\d+(?:\.\d+)?)"],
        "SpO2": [r"(?:spo2|oxygen saturation|o2 saturation|saturation)[^\d]{0,20}(\d+(?:\.\d+)?)"],
    }
    if vital_sign and vital_sign in patterns_by_sign:
        for pattern in patterns_by_sign[vital_sign]:
            match = re.search(pattern, lowered)
            if match:
                return float(match.group(1))
    fallback_match = re.search(r"\b(\d+(?:\.\d+)?)\b", lowered)
    return float(fallback_match.group(1)) if fallback_match else None


def infer_source_type(path: Path, text: str | None = None) -> str:
    lowered_path = str(path).lower()
    content = (text or "").lower()
    for keyword, label in SOURCE_TYPE_KEYWORDS:
        if keyword in lowered_path or keyword in content:
            return label
    if path.name.lower() == "readme.md":
        return "project_report"
    if path.suffix.lower() in {".md", ".txt"}:
        return "documentation"
    return "documentation"


def infer_title(path: Path, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return path.stem.replace("_", " ").replace("-", " ").title()


def find_project_text_files() -> list[Path]:
    """Return markdown and text files that can be indexed by the RAG pipeline."""

    excluded_parts = {".git", ".venv", "data/raw", "data/processed", "data/rag_chunks", "data/rag_index"}
    files: list[Path] = []
    for suffix in ("*.md", "*.txt"):
        for candidate in ROOT_DIR.rglob(suffix):
            candidate_str = str(candidate)
            if any(part in candidate_str for part in excluded_parts):
                continue
            if candidate.name.lower() == "rag_documents.csv" or candidate.name.lower() == "rag_chunks.csv":
                continue
            files.append(candidate)
    return sorted(files)


def split_text_into_chunks(text: str, min_words: int = 400, max_words: int = 800, overlap: int = 80) -> list[str]:
    """Split a long text into overlapping word windows."""

    cleaned = normalize_whitespace(text)
    if not cleaned:
        return []
    words = cleaned.split()
    if len(words) <= max_words:
        return [cleaned]

    step = max(1, max_words - overlap)
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        chunk_words = words[start:end]
        if len(chunk_words) >= min_words or not chunks:
            chunks.append(" ".join(chunk_words).strip())
        if end >= len(words):
            break
        start += step
    return [normalize_whitespace(chunk) for chunk in chunks if normalize_whitespace(chunk)]


def split_paragraphs(text: str) -> list[str]:
    blocks = re.split(r"\n\s*\n", text.strip())
    return [normalize_whitespace(block) for block in blocks if normalize_whitespace(block)]


def age_group_from_age(age: float | int | None) -> str:
    if age is None:
        return "Unknown"
    if age >= 85:
        return "85+"
    if age >= 75:
        return "75-84"
    return "65-74"


def time_window_from_hours(hours_since_admission: float) -> str | None:
    if hours_since_admission <= 6:
        return "first_6h"
    if hours_since_admission <= 12:
        return "first_12h"
    if hours_since_admission <= 24:
        return "first_24h"
    return None
