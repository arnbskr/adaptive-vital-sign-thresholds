from __future__ import annotations

import logging

import pandas as pd

from .config import RAG_CHUNKS_DIR, RAG_DOCUMENTS_DIR, ensure_data_directories
from .rag_utils import split_text_into_chunks

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def chunk_documents() -> pd.DataFrame:
    ensure_data_directories()
    source_path = RAG_DOCUMENTS_DIR / "rag_documents.csv"
    if not source_path.exists():
        raise FileNotFoundError("rag_documents.csv not found. Run python -m src.prepare_rag_documents first.")

    documents_df = pd.read_csv(source_path)
    chunk_records: list[dict[str, object]] = []
    chunk_counter = 1

    for _, row in documents_df.iterrows():
        text = str(row.get("text", "")).strip()
        if not text:
            continue

        if row.get("source_type") == "mimic_stats":
            chunks = [text]
        else:
            chunks = split_text_into_chunks(text, min_words=400, max_words=800, overlap=80)
            if not chunks:
                chunks = [text]

        for chunk_text in chunks:
            chunk_records.append(
                {
                    "chunk_id": f"chunk_{chunk_counter:06d}",
                    "doc_id": row.get("doc_id"),
                    "chunk_text": chunk_text,
                    "source_file": row.get("source_file"),
                    "source_type": row.get("source_type"),
                    "vital_sign": row.get("vital_sign"),
                    "itemid": row.get("itemid", ""),
                    "label": row.get("label", ""),
                    "unitname": row.get("unitname", ""),
                    "age_group": row.get("age_group"),
                    "time_window": row.get("time_window"),
                    "section": row.get("section"),
                    "title": row.get("title"),
                    "is_demo_data": row.get("is_demo_data", False),
                }
            )
            chunk_counter += 1

    chunks_df = pd.DataFrame(chunk_records)
    output_path = RAG_CHUNKS_DIR / "rag_chunks.csv"
    chunks_df.to_csv(output_path, index=False)
    LOGGER.info("Saved %s chunks to %s", len(chunks_df), output_path)
    return chunks_df


def main() -> None:
    chunk_documents()


if __name__ == "__main__":
    main()
