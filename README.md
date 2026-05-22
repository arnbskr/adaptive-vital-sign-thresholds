# ICU Trajectory RAG Assistant - Phase 1

## 1. Project Overview

This repository contains a local Python RAG (Retrieval-Augmented Generation) pipeline designed to explore ICU vital signs and clinical data. Originally relying on lexical search (TF-IDF), the system has been upgraded to a modern semantic architecture using dense embeddings, a vector database, and local LLM inference. 

## 2. Phase 1 Objective

The goal of Phase 1 is to implement a functional, end-to-end RAG system: user question -> semantic retrieval over indexed chunks -> grounded generation by an LLM. 
The system is intended for academic interpretation and literature-style exploration, not for clinical decision support.

## 3. What this phase does

The pipeline now features:
* **Multi-format Ingestion:** Extracts and chunks data from structured tables (CSV), text documents (Markdown), and unstructured documents (PDF).
* **Smart Chunking:** Uses a sliding window approach (800 characters with 120 overlap) to preserve semantic context.
* **Semantic Embeddings:** Uses the `bge-m3:latest` model to convert chunks into dense vectors.
* **Vector Storage:** Persistently stores embeddings and metadata using **ChromaDB**.
* **Semantic Retrieval:** Uses L2 distance to retrieve the most contextually relevant chunks.
* **Grounded Generation:** Uses `qwen2.5:14b` (or similar local LLM) acting as an academic assistant to formulate a precise answer based *strictly* on the retrieved context, preventing hallucinations.

## 4. What this phase does NOT do

This Phase 1 prototype does not implement:
- Multi-step reasoning or Agentic workflows (planned for Phase 2).
- Function calling / Tool Calling via MCP (planned for Phase 2).
- Production deployment or cloud orchestration.
- Clinical diagnosis.

## 5. Data Sources

The system ingests heterogeneous data sources:
- **Processed MIMIC-IV Data (CSV):** Vital sign summaries and patient cohorts (e.g., `elderly_icu_stays.csv`).
- **Academic/Project Documents (PDF):** Course materials or reference reports (e.g., `Rapport_Final.pdf`).
- **Text/Markdown (MD):** Project documentation (e.g., `README.md`).

## 6. Prerequisites & Installation

The project relies on a local Python environment and an active **Ollama** server for local inference.

1. **Python Environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. **Ollama Setup:**
Ensure Ollama is installed and running on your machine. Pull the required models:

```bash
ollama pull bge-m3:latest
ollama pull qwen2.5:14b
```

## 7. How to Run the Pipeline

### A. New Semantic RAG Pipeline (ChromaDB & Ollama)

**Step 1: Data Ingestion & Vectorization**
Run the ingestion script to parse the documents, generate embeddings, and populate the ChromaDB vector database.

```bash
python src/ingest.py
```

*Expected output: Confirmation of the total number of chunks successfully indexed in ChromaDB.*

**Step 2: Retrieval & Generation (RAG)**
Run the RAG script to ask a question. The system will vectorize the query, retrieve the top chunks, and generate a grounded response.

```bash
python src/rag.py
```

### B. Legacy Pipeline (TF-IDF & Streamlit App)

If you need to re-extract data from BigQuery or run the original TF-IDF Streamlit interface:

Run the legacy extraction and indexing pipeline:

```bash
python -m src.bigquery_extract_mimic
python -m src.prepare_rag_documents
python -m src.chunk_documents
python -m src.build_rag_index
```

Run the local Streamlit app:

```bash
streamlit run app.py
```

## 8. Example Questions

* "Quels sont les seuils adaptatifs pour la fréquence cardiaque des patients âgés ?"
* "Quelle est la différence entre les percentiles du groupe 65-74 ans et ceux du groupe 85 ans et plus ?"
* "For a patient aged 82 with mean HR 104 bpm in the first 24h ICU stay, is this value high?"
* "What is the difference between standard thresholds and MIMIC-IV percentile-based summaries?"

## 9. Next Step: Phase 2 (Agents & MCP)

Phase 2 will transition this "passive" RAG into an active Agentic system. The LLM will use Function Calling to autonomously select tools (including this vector search, external APIs, or arithmetic calculators) via the Model Context Protocol (MCP) to solve complex, multi-step queries.