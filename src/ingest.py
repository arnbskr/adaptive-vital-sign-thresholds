from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
from openai import OpenAI
from pypdf import PdfReader

try:
    from .config import ROOT_DIR, ensure_data_directories
except ImportError:  # pragma: no cover - direct script execution fallback
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.config import ROOT_DIR, ensure_data_directories

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
EMBEDDING_MODEL = "bge-m3:latest"
COLLECTION_NAME = "icu_rag"
CHROMA_DB_DIR = ROOT_DIR / "data" / "chroma_db"

FILES_TO_INGEST = [
    "README.md",
    "Rapport_Final.pdf",
    "data/processed/vital_signs_elderly_icu_summary.csv",
    "data/rag_documents/rag_documents.csv",
    "R/03_mimic_preprocessing.R",
    "R/04_exploratory_analysis.R",
    "R/05_statistical_modeling.R",
    "R/06_threshold_definition.R",
    "R/07_validation_discussion.R",
]

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


def build_client() -> OpenAI:
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def build_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(CHROMA_DB_DIR))


def build_collection(chroma_client: chromadb.PersistentClient, reset: bool = False) -> Any:
    if reset:
        try:
            chroma_client.delete_collection(name=COLLECTION_NAME)
            LOGGER.info("Removed existing ChromaDB collection: %s", COLLECTION_NAME)
        except Exception:  # noqa: BLE001
            pass

    return chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def relative_source_path(path: Path) -> str:
    return str(path.relative_to(ROOT_DIR)) if path.is_relative_to(ROOT_DIR) else str(path)


def normalize_source_file(value: Any, fallback_path: Path) -> str:
    candidate = str(value).strip() if value not in {None, ""} else ""
    if not candidate:
        return relative_source_path(fallback_path)

    candidate_path = Path(candidate)
    if candidate_path.is_absolute() and candidate_path.is_relative_to(ROOT_DIR):
        return str(candidate_path.relative_to(ROOT_DIR))
    if not candidate_path.is_absolute():
        return candidate.replace("\\", "/")
    return candidate.replace("\\", "/")


def is_forbidden_source(source_path: str) -> bool:
    lowered = source_path.replace("\\", "/").lower()
    return any(part.lower() in lowered for part in FORBIDDEN_PATH_PARTS)


def normalize_whitespace(text: str) -> str:
    return " ".join(str(text).split())


def split_text_into_chunks(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return []
    step = max(1, chunk_size - overlap)
    return [cleaned[i : i + chunk_size] for i in range(0, len(cleaned), step) if cleaned[i : i + chunk_size].strip()]


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text)
    return normalize_whitespace(" ".join(pages))


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _base_metadata(source_file: str, source_type: str, title: str | None = None) -> dict[str, Any]:
    metadata = {
        "source_file": source_file,
        "source_type": source_type,
        "vital_sign": "",
        "age_group": "",
        "time_window": "",
        "itemid": "",
        "label": "",
        "unitname": "",
        "is_demo_data": False,
        "title": title or Path(source_file).name,
    }
    return metadata


def _row_value(row: pd.Series, key: str, default: Any = "") -> Any:
    value = row.get(key, default)
    if pd.isna(value):
        return default
    return value


def _summary_row_to_document(row: pd.Series, source_file: str) -> tuple[str, dict[str, Any]]:
    vital_sign = str(_row_value(row, "vital_sign"))
    age_group = str(_row_value(row, "age_group"))
    time_window = str(_row_value(row, "time_window"))
    itemid = _row_value(row, "itemid")
    label = str(_row_value(row, "label"))
    unitname = str(_row_value(row, "unitname"))
    count = _row_value(row, "count")
    mean = _row_value(row, "mean")
    median = _row_value(row, "median")
    p5 = _row_value(row, "p5")
    p25 = _row_value(row, "p25")
    p50 = _row_value(row, "p50")
    p75 = _row_value(row, "p75")
    p90 = _row_value(row, "p90")
    standard_low = _row_value(row, "standard_low")
    standard_high = _row_value(row, "standard_high")
    percent_below = _row_value(row, "percent_below_standard_low")
    percent_above = _row_value(row, "percent_above_standard_high")

    text = "\n".join(
        [
            "Source type: MIMIC-IV statistical summary.",
            f"Vital sign: {vital_sign}.",
            f"Age group: {age_group}.",
            f"ICU time window: {time_window}.",
            f"Item ID: {itemid}.",
            f"Label: {label}.",
            f"Unit: {unitname}.",
            f"Count: {count}.",
            f"Mean: {mean}.",
            f"Median: {median}.",
            f"P5: {p5}.",
            f"P25: {p25}.",
            f"P50: {p50}.",
            f"P75: {p75}.",
            f"P90: {p90}.",
            f"Standard low threshold: {standard_low}.",
            f"Standard high threshold: {standard_high}.",
            f"Percent below standard low: {percent_below}.",
            f"Percent above standard high: {percent_above}.",
            "These statistics are descriptive summaries from MIMIC-IV ICU data. They are not clinical decision rules and must be interpreted as academic context only.",
        ]
    )

    metadata = _base_metadata(source_file=source_file, source_type="mimic_stats", title=f"{vital_sign} summary")
    metadata.update(
        {
            "vital_sign": vital_sign,
            "age_group": age_group,
            "time_window": time_window,
            "itemid": str(itemid),
            "label": label,
            "unitname": unitname,
            "is_demo_data": bool(_row_value(row, "is_demo_data", False)),
        }
    )
    return text, metadata


def _rag_documents_row_to_document(row: pd.Series, source_file: str) -> tuple[str, dict[str, Any]]:
    document_text = str(_row_value(row, "text")).strip()
    normalized_source_file = normalize_source_file(_row_value(row, "source_file", source_file), Path(source_file))
    metadata = _base_metadata(
        source_file=normalized_source_file,
        source_type=str(_row_value(row, "source_type", "documentation")),
        title=str(_row_value(row, "title", Path(source_file).stem)),
    )
    metadata.update(
        {
            "vital_sign": str(_row_value(row, "vital_sign", "")),
            "age_group": str(_row_value(row, "age_group", "")),
            "time_window": str(_row_value(row, "time_window", "")),
            "itemid": str(_row_value(row, "itemid", "")),
            "label": str(_row_value(row, "label", "")),
            "unitname": str(_row_value(row, "unitname", "")),
            "is_demo_data": bool(_row_value(row, "is_demo_data", False)),
        }
    )
    return document_text, metadata


def _generic_csv_row_to_document(row: pd.Series, source_file: str) -> tuple[str, dict[str, Any]]:
    document_text = ", ".join(f"{key}: {_row_value(row, key)}" for key in row.index)
    metadata = _base_metadata(source_file=source_file, source_type="documentation", title=Path(source_file).stem)
    return document_text, metadata


def _document_records_for_csv(path: Path) -> list[tuple[str, dict[str, Any]]]:
    dataframe = pd.read_csv(path)
    records: list[tuple[str, dict[str, Any]]] = []
    source_file = relative_source_path(path)

    # Keep table rows aligned with their statistics: one CSV row becomes one retrievable document.
    for _, row in dataframe.iterrows():
        if path.name == "vital_signs_elderly_icu_summary.csv":
            text, metadata = _summary_row_to_document(row, source_file)
        elif path.name == "rag_documents.csv":
            text, metadata = _rag_documents_row_to_document(row, source_file)
        else:
            text, metadata = _generic_csv_row_to_document(row, source_file)
        records.append((text, metadata))

    return records


def _document_records_for_text(path: Path) -> list[tuple[str, dict[str, Any]]]:
    if path.suffix.lower() == ".pdf":
        text = extract_text_from_pdf(path)
        source_type = "project_report" if path.name.lower() == "rapport_final.pdf" else "article"
        source_file = relative_source_path(path)
        metadata = _base_metadata(source_file=source_file, source_type=source_type, title=path.stem)
        return [(chunk, metadata | {"title": path.stem}) for chunk in split_text_into_chunks(text)]

    text = normalize_whitespace(read_text_file(path))
    source_type = "project_report" if path.name.lower() == "readme.md" else "documentation"
    source_file = relative_source_path(path)
    metadata = _base_metadata(source_file=source_file, source_type=source_type, title=path.stem)
    return [(chunk, metadata | {"title": path.stem}) for chunk in split_text_into_chunks(text)]


def _embed_and_store(
    client: OpenAI,
    collection: Any,
    text: str,
    metadata: dict[str, Any],
    source_file: str,
    record_index: int,
) -> bool:
    if not text or not text.strip():
        return False
    if is_forbidden_source(source_file):
        LOGGER.warning("Skipping forbidden source path: %s", source_file)
        return False

    for key in ("source_file", "title"):
        if key in metadata and is_forbidden_source(str(metadata.get(key, ""))):
            LOGGER.warning("Skipping chunk because %s is forbidden: %s", key, metadata.get(key))
            return False

    try:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        embedding = response.data[0].embedding
        chunk_id = hashlib.sha1(f"{source_file}|{record_index}|{text}".encode("utf-8")).hexdigest()

        collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )
        return True
    except Exception as exc:  # noqa: BLE001
        preview = normalize_whitespace(text)[:160]
        LOGGER.warning("Skipping chunk after embedding/indexing error for %s: %s | preview=%s", source_file, exc, preview)
        return False


def run_pipeline() -> None:
    ensure_data_directories()
    LOGGER.info("Starting whitelist-only semantic ingestion.")

    embedding_client = build_client()
    chroma_client = build_chroma_client()
    collection = build_collection(chroma_client, reset=True)

    files_ingested = 0
    chunks_created = 0
    indexed_sources: set[str] = set()

    for file_name in FILES_TO_INGEST:
        file_path = ROOT_DIR / file_name
        source_file = relative_source_path(file_path)

        if is_forbidden_source(source_file):
            LOGGER.warning("Skipping forbidden path from whitelist: %s", source_file)
            continue

        if not file_path.exists():
            LOGGER.warning("Missing whitelisted file, skipped: %s", source_file)
            continue

        LOGGER.info("Processing %s", source_file)
        files_ingested += 1

        if file_path.suffix.lower() == ".csv":
            records = _document_records_for_csv(file_path)
        else:
            records = _document_records_for_text(file_path)

        for record_index, (text, metadata) in enumerate(records):
            metadata = dict(metadata)
            metadata["source_file"] = str(metadata.get("source_file", source_file))
            if is_forbidden_source(str(metadata["source_file"])):
                LOGGER.warning("Skipping forbidden indexed source: %s", metadata["source_file"])
                continue

            stored = _embed_and_store(embedding_client, collection, text, metadata, str(metadata["source_file"]), record_index)
            if stored:
                chunks_created += 1
                indexed_sources.add(str(metadata["source_file"]))

    if any(is_forbidden_source(source) for source in indexed_sources):
        raise RuntimeError("Forbidden source detected in indexed documents.")

    LOGGER.info("Number of files ingested: %s", files_ingested)
    LOGGER.info("Number of chunks created: %s", chunks_created)
    LOGGER.info("Number of ChromaDB documents stored: %s", collection.count())
    LOGGER.info("Unique source files indexed: %s", ", ".join(sorted(indexed_sources)))


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()