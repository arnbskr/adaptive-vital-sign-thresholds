from __future__ import annotations

import logging

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import RAG_CHUNKS_DIR, RAG_INDEX_DIR, ensure_data_directories

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def build_rag_index() -> None:
    ensure_data_directories()
    chunks_path = RAG_CHUNKS_DIR / "rag_chunks.csv"
    if not chunks_path.exists():
        raise FileNotFoundError("rag_chunks.csv not found. Run python -m src.chunk_documents first.")

    chunks_df = pd.read_csv(chunks_path)
    if chunks_df.empty:
        raise ValueError("rag_chunks.csv is empty. Nothing to index.")

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(chunks_df["chunk_text"].fillna("").astype(str))

    index_path = RAG_INDEX_DIR / "chunks_index.csv"
    vectorizer_path = RAG_INDEX_DIR / "tfidf_vectorizer.joblib"
    matrix_path = RAG_INDEX_DIR / "tfidf_matrix.joblib"

    chunks_df.to_csv(index_path, index=False)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(matrix, matrix_path)

    LOGGER.info("Saved TF-IDF index to %s, %s and %s", index_path, vectorizer_path, matrix_path)


def main() -> None:
    build_rag_index()


if __name__ == "__main__":
    main()
