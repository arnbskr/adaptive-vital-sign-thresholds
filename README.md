# ICU Trajectory RAG Assistant - Phase 1 Semantic RAG

## Project Overview

This repository implements a local Retrieval-Augmented Generation pipeline for ICU trajectory exploration. Phase 1 is now centered on a semantic architecture only: documents are ingested, chunked, embedded with `bge-m3:latest`, stored in ChromaDB, retrieved semantically, and answered by a local `qwen2.5:14b` model through Ollama.

The project is intended for academic interpretation and literature-style exploration, not for clinical diagnosis or treatment recommendation.

## What This Phase Does

- Multi-format ingestion of Markdown, PDF, CSV, and R scripts.
- Semantic embeddings with `bge-m3:latest`.
- Persistent vector storage in ChromaDB.
- Semantic retrieval with metadata-aware filtering.
- Grounded local generation with `qwen2.5:14b`.
- Streamlit interface for interactive questions and source inspection.

## What This Phase Does Not Do

- Agents.
- MCP.
- Function calling.
- Tool calling.
- Multi-agent orchestration.
- FastAPI.
- Docker.
- Clinical diagnosis.
- Treatment recommendation.

## Data Sources

The semantic pipeline ingests whitelisted project sources such as:

- `README.md`
- `Rapport_Final.pdf`
- `data/processed/vital_signs_elderly_icu_summary.csv`
- `data/rag_documents/rag_documents.csv`
- `R/03_mimic_preprocessing.R`
- `R/04_exploratory_analysis.R`
- `R/05_statistical_modeling.R`
- `R/06_threshold_definition.R`
- `R/07_validation_discussion.R`

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ollama Models

```bash
ollama pull bge-m3:latest
ollama pull qwen2.5:14b
```

## Run the Semantic Pipeline

Build the vector database:

```bash
python src/ingest.py
```

Run the Streamlit app:

```bash
streamlit run app.py
```

Recommended clean rebuild:

```bash
rm -rf data/chroma_db
python src/ingest.py
streamlit run app.py
```

## Streamlit Interface

The app exposes:

- custom question input
- top-k retrieval control
- source type filtering
- vital sign filtering
- age group filtering
- ICU time window filtering
- retrieved sources and metadata

It displays the active semantic stack as:

- Mode: Phase 1 Semantic RAG
- Index: ChromaDB
- Embedding model: `bge-m3:latest`
- LLM: `qwen2.5:14b`
- Source focus: MIMIC-IV summaries + project documents

## Example Questions

- What is the difference between a standard clinical threshold and an adaptive percentile-based threshold?
- For a patient aged 82 with mean HR 104 bpm in the first 24h ICU stay, is this value high?
- For a patient aged 78 with MAP 62 mmHg in the first 24h ICU stay, is this low?
- For a patient aged 80 with SpO2 90% in the first 24h ICU stay, is this low?

## Architecture Summary

```text
Documents / CSV / PDF / R scripts
→ whitelist-only ingestion
→ chunking and CSV row normalization
→ bge-m3 embeddings through Ollama
→ ChromaDB vector database
→ semantic retrieval with metadata filters
→ qwen2.5:14b grounded answer
```

## Archived Legacy Pipeline

The repository still contains older TF-IDF scripts for reference, but they are no longer used by the Streamlit app or the main workflow.

Archived files:

- `src/prepare_rag_documents.py`
- `src/chunk_documents.py`
- `src/build_rag_index.py`
- `src/retrieve_chunks.py`
- `src/rag_utils.py`

Legacy commands, kept only for reference:

```bash
python -m src.bigquery_extract_mimic
python -m src.prepare_rag_documents
python -m src.chunk_documents
python -m src.build_rag_index
```

## Phase 2 Direction

Phase 2 will explore agents, MCP, and function calling. Phase 1 remains a local semantic RAG system only.
