# Legacy — early TF-IDF / pre-semantic RAG prototype

These files belong to the previous TF-IDF / early RAG prototype and are **not used**
by the active Phase 1 (semantic RAG) or Phase 2 (agentic RAG) pipelines.

They are kept here only for provenance — to show how the retrieval approach evolved
before `src/semantic_rag.py` (ChromaDB + `bge-m3` embeddings) replaced it.

| File | Historical role |
|------|-----------------|
| `prepare_rag_documents.py` | built `rag_documents.csv` from processed CSVs |
| `chunk_documents.py` | sliding-window chunking into `rag_chunks.csv` |
| `build_rag_index.py` | built the TF-IDF index |
| `retrieve_chunks.py` | TF-IDF retrieval |
| `generate_rag_answer.py` | standalone TF-IDF answer demo |
| `rag.py` | standalone retrieval demo |
| `rag_utils.py` | shared helpers for the above |

**Do not extend these.** They are frozen: their relative imports (`from .config`,
`from .rag_utils`) assume the old `src/` layout, so they are not runnable from this
folder. The active pipeline is documented in the top-level `README.md` / `CLAUDE.md`.
