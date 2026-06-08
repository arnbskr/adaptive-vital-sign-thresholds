from __future__ import annotations

import os
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

# Phase 2 (agentic RAG with deterministic tools) artefacts.
AGENT_TRACES_DIR = DATA_DIR / "agent_traces"
PHASE2_OUTPUTS_DIR = DATA_DIR / "phase2_outputs"

# Canonical path to the MIMIC-IV statistical summary consumed by the tools.
VITAL_SUMMARY_CSV = PROCESSED_DIR / "vital_signs_elderly_icu_summary.csv"

# Phase 2 tool backend selection.
# - "local"      : in-process MCP-compatible registry (default, no server needed).
# - "mcp_remote" : real MCP server over streamable-http at MCP_REMOTE_URL.
# Both are overridable via environment variables so nothing is hardcoded at the
# call sites (agent, evaluation, Streamlit).
DEFAULT_TOOL_BACKEND = os.environ.get("PHASE2_TOOL_BACKEND", "local")
MCP_REMOTE_URL = os.environ.get("MCP_REMOTE_URL", "http://127.0.0.1:8000/mcp")


def ensure_data_directories() -> None:
    """Create the directories used by the Phase 1 and Phase 2 pipelines if needed."""

    for directory in [
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        EVALUATION_DIR,
        RAG_DOCUMENTS_DIR,
        RAG_CHUNKS_DIR,
        RAG_INDEX_DIR,
        AGENT_TRACES_DIR,
        PHASE2_OUTPUTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
