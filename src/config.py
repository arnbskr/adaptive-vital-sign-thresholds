from __future__ import annotations

from pathlib import Path

PROJECT_ID = "mimic-rag-2026-vinith"
ICU_DATASET = "physionet-data.mimiciv_3_1_icu"
HOSP_DATASET = "physionet-data.mimiciv_3_1_hosp"
ALLOW_DEMO_FALLBACK = False

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVALUATION_DIR = DATA_DIR / "evaluation"
RAG_DOCUMENTS_DIR = DATA_DIR / "rag_documents"
RAG_CHUNKS_DIR = DATA_DIR / "rag_chunks"
RAG_INDEX_DIR = DATA_DIR / "rag_index"


def ensure_data_directories() -> None:
    """Create the directories used by the Phase 1 pipeline if needed."""

    for directory in [
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        EVALUATION_DIR,
        RAG_DOCUMENTS_DIR,
        RAG_CHUNKS_DIR,
        RAG_INDEX_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
