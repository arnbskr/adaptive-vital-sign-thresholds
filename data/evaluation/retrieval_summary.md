# Retrieval Strategy Comparison

This evaluation is intentionally small and Phase 1 only. It compares semantic retrieval, semantic retrieval with strict metadata-aware reranking, a lexical baseline, and an optional metadata-filtered variant.

| Strategy | Precision proxy | Recall proxy | Latency | Cost | Complexity | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Semantic only | 25% | 100% | 173.1 ms | Medium | Low | ChromaDB similarity search with bge-m3 embeddings and no reranking. |
| Semantic + metadata reranking | 75% | 100% | 94.4 ms | Medium | Medium | Similarity search plus strict metadata-aware reranking for patient-value questions. |
| Keyword / lexical baseline | 50% | 100% | 0.0 ms | Very low | Low | Simple lexical scoring over indexed documents only; no semantic embedding at retrieval time. |
| Semantic + metadata filtering | 75% | 100% | 94.3 ms | Medium | Medium-high | Inference-guided metadata filters before reranking; high precision when metadata is clean. |

## Table Handling

The ICU vital-sign summary CSV is treated as a table-aware source: one CSV row becomes one retrievable RAG document so the age group, vital sign, time window, and threshold statistics stay aligned.

## Interpretation

- Lexical baseline: very low cost and low latency, but weak on synonyms and phrasing differences.
- Semantic retrieval: better recall because embeddings can match paraphrases and related concepts.
- Metadata-aware reranking: improves precision for patient-value questions because the age group, vital sign, and time window are explicitly prioritized.
- Metadata filtering: highest precision when metadata is clean, but it can lose recall if metadata is missing or incomplete.