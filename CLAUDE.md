# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Local Retrieval-Augmented Generation pipeline ("ICU Trajectory RAG Assistant", Phase 1) for exploring adaptive vital-sign thresholds in elderly ICU patients, built on MIMIC-IV data. Everything runs locally through **Ollama** — no cloud LLM calls. The system is an academic interpretation aid, **not** a clinical decision tool.

The repo now has two layers. **Phase 1** is the semantic RAG. **Phase 2** adds a single LLM agent that orchestrates deterministic tools over an MCP boundary, reusing Phase 1 for retrieval. Phase 2 is deliberately constrained: **single agent only** — no multi-agent, no supervisor, no long-term memory, no FastAPI, no Docker, no real-time monitoring, no clinical diagnosis/treatment. Keep new work inside that boundary.

## Environment & Commands

This repo mixes a Python RAG app and an R analysis pipeline. They share the `data/` directory but run independently.

```bash
# Python setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Ollama models (must be pulled and `ollama serve` running on 127.0.0.1:11434)
ollama pull bge-m3:latest      # embeddings
ollama pull qwen2.5:14b        # generation

# Build the vector DB (wipes & rebuilds the ChromaDB collection each run)
python src/ingest.py

# Run the app
streamlit run app.py

# Clean rebuild
rm -rf data/chroma_db && python src/ingest.py && streamlit run app.py

# Retrieval strategy benchmark → writes data/evaluation/*.{csv,md}
python -m src.evaluate_retrieval

# Phase 2: agent scenario test (local backend) → data/evaluation/agent_{evaluation.csv,summary.md}
python -m src.evaluate_agent --backend local
python -m src.evaluate_agent --backend mcp_remote   # needs the MCP server up; skips cleanly otherwise

# Phase 2: in-process MCP tool catalogue / stdio server (real MCP if `mcp` installed, else local mode)
python -m src.mcp_server

# Phase 2: real MCP network backend — unified streamable-http server exposing the SAME specialized tools
python src/server_mcp.py                              # FastMCP streamable-http server on MCP_REMOTE_URL
PHASE2_TOOL_BACKEND=mcp_remote streamlit run app.py   # app drives tools over the network (falls back to local if down)

# Standalone ReAct client demo against that server (optional, separate from app.py)
python src/agent2.py

# Restore the gitignored summary CSV from the ChromaDB index if lost
python -m src.restore_summary_from_index
```

**Important data note:** `data/processed/vital_signs_elderly_icu_summary.csv` is gitignored and may be absent on a fresh checkout, but its `mimic_stats` rows live inside the committed `data/chroma_db/` index. A blind `python src/ingest.py` rebuild when the CSV is missing would **destroy** the vital-sign data. Run `python -m src.restore_summary_from_index` first to rebuild the CSV from the index (it also dedups the historical precise/rounded row pairs). `src/ingest.py` now runs a strict post-ingestion whitelist audit and raises if any non-whitelisted source (e.g. a stale `requirements.txt`) is in the index.

There is no test runner or linter configured. `test_mimic_bigquery.py` is a standalone connectivity check for BigQuery, not a unit test (run with `python test_mimic_bigquery.py`; requires GCP auth).

## Architecture

**Active pipeline** (everything else under `src/` is legacy — see below):

- `src/config.py` — paths, BigQuery project/dataset IDs, `ensure_data_directories()`.
- `src/ingest.py` — whitelist-only ingestion. Reads a hardcoded `FILES_TO_INGEST` list (README, the final report PDF, two CSVs, five R scripts), chunks/normalizes them, embeds via Ollama `bge-m3`, and upserts into the ChromaDB collection `icu_rag` at `data/chroma_db`. `reset=True` deletes the collection first, so **ingest is destructive and idempotent**.
- `src/semantic_rag.py` — the retrieval + generation core, imported by the app. Contains the query-time intelligence; this is the main file to edit for RAG behavior.
- `app.py` — Streamlit UI. Thin layer over `semantic_rag`; sidebar filters map directly to `retrieve_semantic_chunks` kwargs.
- `src/evaluate_retrieval.py` — offline comparison of retrieval strategies (semantic-only vs. metadata reranking vs. filtering) over a fixed question set.

**Phase 2 layer (agentic RAG):**

- `src/tools/` — deterministic, pure-function tools: the seven specialized ones (data availability, vital summary, threshold comparison + `explain_threshold_type`, percentile comparison, project-context RAG wrapper, report generator) **plus `calculator.py`'s `calculatrice_medicale`** (demonstration-only safe arithmetic via a strict AST, never `eval()`). They do all calculations; the LLM never computes. `vital_summary.py` is CSV-first with a ChromaDB `mimic_stats` fallback and selects the highest-`count` itemid when a vital has several.
- `src/mcp_server.py` — the in-process tool registry. `TOOL_SPECS`/`call_tool(name, arguments)`/`list_tools()` (eight tools) back the **local** backend and are also what the remote server re-exports; `main()` runs a real MCP stdio server only if the `mcp` SDK is installed.
- `src/tool_client.py` — **backend-agnostic** `ToolClient` interface (`list_tools` / `call_tool` / `is_available` / `close`) + `get_tool_client(backend)` factory. The agent depends on this, **not** on `mcp_server` or `server_mcp` directly.
- `src/tool_backends/local_backend.py` — `LocalToolBackend`: thin adapter over `mcp_server.call_tool` (default; no network, no `mcp` SDK).
- `src/tool_backends/mcp_remote_backend.py` — `MCPRemoteBackend`: a **real MCP streamable-http** client wrapped as a synchronous `ToolClient`. The async session lives on a dedicated event loop in a background thread (entered once via `AsyncExitStack`, reused per call) — a clean sync/async bridge, not a fragile hack. Connects, handshakes, discovers tools dynamically, and normalizes MCP results back into the same plain dicts the local backend returns. Imports `mcp` lazily; a connection failure raises `ToolBackendError`.
- `src/agent.py` — `run_agent(question, tool_backend=None, tool_client=None, allow_fallback=True)`: the single orchestrator. Classifies into `patient_value_question` / `concept_question` / `dataset_question` / `pipeline_question` / `calculator_question` / `unsupported_or_missing_data`, calls tools **through a `ToolClient`** with a recorded trace, then asks `qwen2.5:14b` to phrase the answer. **Tool routing by intent:** patient questions → data/threshold/percentile tools; concept/dataset → `retrieve_project_context` (+ `explain_threshold_type`); a **purely arithmetic** question (detected by `_extract_calculation`, only on non-patient questions, guarded by trigger words so ranges like `65-74` aren't read as subtraction) → `calculatrice_medicale` only, with **no RAG/MIMIC call and no heavy clinical warning**. Backend defaults to `DEFAULT_TOOL_BACKEND` (env `PHASE2_TOOL_BACKEND`); if a remote backend is unreachable it **falls back to local with a warning**. **Degrades gracefully** everywhere — never crashes. Result includes `tool_backend`.
- `src/tool_trace.py` — `ToolTrace(backend=...).record(name, inputs, callable)` times/captures each call and stamps the `backend`; `save_trace` persists each run to `data/agent_traces/agent_trace_*.json`.
- `src/evaluate_agent.py` — 5-scenario agent benchmark (patient / concept / **calculator** / unsupported-age / dataset) with `--backend local|mcp_remote` → `data/evaluation/agent_{evaluation,summary}{,_mcp_remote}.{csv,md}`. Metrics include `expected_tool_called`, `forbidden_tools_not_called`, `calculator_result_correct`, `backend_used`, `fallback_used`, `non_clinical_warning_when_needed`. Remote run pre-checks reachability and skips cleanly (no crash) if the server is down.

**One Phase 2 architecture, two execution backends.** The agent always runs the same flow — `question → src/agent.py → src/tool_client.py → tool backend → src/tools/ → src/tool_trace.py → answer`. Only the backend differs: **local in-process** (default; `LocalToolBackend → src/mcp_server.py`) or **real MCP network** (optional; `MCPRemoteBackend → src/server_mcp.py` over streamable-http). Both expose the **same eight tools** (seven specialized + `calculatrice_medicale`) and identical outputs, so the medical logic, the trace, and the answers match. This is one system with two modes — not two architectures. See **[Real MCP network backend](#real-mcp-network-backend)** for the network mode and the names you must not confuse.

**Key design decisions to preserve when editing:**

1. **Table-aware ingestion.** `vital_signs_elderly_icu_summary.csv` and `rag_documents.csv` are ingested **one row = one document** (see `_summary_row_to_document` / `_rag_documents_row_to_document`), keeping each patient stat-block atomic with its metadata. Only PDFs/free text use sliding-window chunking (`split_text_into_chunks`, 800 chars / 120 overlap). Don't collapse CSVs into text windows.

2. **Metadata-aware reranking for patient-value questions.** `infer_patient_context()` parses age, vital sign, value, and time window from the question via regex. When all are present (`is_patient_value_question`), `retrieve_semantic_chunks` reranks candidates with large additive bonuses/penalties (`_match_bonus_and_penalty`) and a `priority_bucket`, so the exact-matching `mimic_stats` row outranks semantically-similar-but-wrong rows. For general questions it falls back to pure similarity (`1/(1+distance)`). The grounded prompt then feeds **only exact-match chunks** to the LLM for direct comparisons (`build_grounded_prompt`) to avoid cross-age-group/time-window contamination.

3. **Source allowlist + forbidden-path guards** appear in **both** `ingest.py` and `semantic_rag.py` (`PROJECT_SOURCE_ALLOWLIST`, `FORBIDDEN_PATH_PARTS`, `is_allowed_source_file`). These prevent `.venv`/site-packages/chroma internals from ever being indexed or displayed. If you change the set of indexed sources, update both files.

4. **Canonical vocab is fixed.** Vital signs, age groups (`65-74`, `75-84`, `85+`), and time windows (`first_6h/12h/24h`) are enumerated in `semantic_rag.py` regex tables, the BigQuery `ITEM_SPECS`/`VITAL_GROUP_SPECS`, the app's sidebar selectboxes, and the CSV metadata. They must stay consistent across all four or filtering/reranking silently breaks.

Ollama is accessed through the **OpenAI client** pointed at `http://127.0.0.1:11434/v1` (`build_ollama_client`), not a dedicated Ollama SDK.

## Real MCP network backend

The **MCP remote backend** is the optional network execution mode of the *same* Phase 2 system: the same agent reaches the same tools over the official `mcp` SDK and the `streamable-http` transport instead of in-process. It is **opt-in** (`PHASE2_TOOL_BACKEND=mcp_remote`); the default Streamlit flow stays local. This is a genuine MCP network boundary (decoupled client/server, dynamic tool discovery), useful for **Devoir 2**, but it does **not** replace the local backend.

**`src/server_mcp.py` — MCP HTTP server.** Builds a `FastMCP("icu-trajectory-mcp")` server configured from `MCP_REMOTE_URL` (host/port/path parsed from the single config value, default `http://127.0.0.1:8000/mcp`) and exposes tools over the **official MCP protocol** via `streamable-http`. It simply iterates the shared `TOOL_SPECS` and registers **every tool** (the seven specialized ones **and** `calculatrice_medicale`) via `mcp.add_tool(spec.func, ...)` — **no tool logic is defined or duplicated here**, and it is import-safe (no module-level ChromaDB/Ollama clients). `calculatrice_medicale` lives in `src/tools/calculator.py` (strict AST, never `eval()`) and is therefore exposed identically by both backends. (The old `rechercher_informations_cliniques` was removed; `retrieve_project_context` is the single RAG tool.)

**`src/tool_backends/mcp_remote_backend.py` — the remote client backend.** This is how the agent talks to the server (it supersedes calling `agent2.py` from the app). It performs the MCP handshake, **dynamically discovers** the tools (`session.list_tools()`), calls them (`session.call_tool`), and normalizes results back into the same plain dicts the local backend returns — wrapped as a synchronous `ToolClient` via a background event-loop thread.

**`src/agent2.py` — standalone ReAct client demo.** An asynchronous MCP client kept as a self-contained demonstration: connect → handshake → discover tools → convert MCP schemas to the OpenAI/Ollama `tools` format → drive `qwen2.5:14b` in a ReAct loop where the LLM chooses tools and results are injected back with the `tool` role. It is **not** used by `app.py`; the app uses `MCPRemoteBackend`.

### Commands

```bash
# Run the app against the real MCP network backend
python src/server_mcp.py                              # terminal 1: unified MCP HTTP server
PHASE2_TOOL_BACKEND=mcp_remote streamlit run app.py   # terminal 2: app uses the remote backend

# Evaluate each backend
python -m src.evaluate_agent --backend local
python -m src.evaluate_agent --backend mcp_remote     # skips cleanly if the server is down

# Standalone ReAct client demo (optional)
python src/agent2.py
```

Prerequisites: Ollama running with `bge-m3:latest` + `qwen2.5:14b`; the ChromaDB index at `data/chroma_db`; and the `mcp` SDK installed (`pip install mcp`, already in `requirements.txt`). The local backend needs none of the MCP SDK or server.

### Presentation wording

> The Streamlit Phase 2 uses a local in-process MCP-compatible tool backend by default. The same agent can also run against a separate real MCP network backend — a streamable-http server and a discovering client — using the *same* specialized tools and medical logic. This demonstrates how the RAG/tool pipeline decouples into a client-server MCP architecture where tools are discovered dynamically and called over the network, with a clean fallback to local.

### Important distinction

Do not confuse these pieces, and do not edit one expecting the others to change:

- `src/tool_client.py` — the backend-agnostic `ToolClient` interface + factory the agent depends on.
- `src/mcp_server.py` — the **local in-process** registry (`TOOL_SPECS` / `call_tool`), behind `LocalToolBackend`.
- `src/server_mcp.py` — the **real MCP HTTP server** (FastMCP, `streamable-http`), reached via `MCPRemoteBackend`.
- `src/agent.py` — the **main agent** used by `app.py` (backend-agnostic, local by default).
- `src/agent2.py` — a **standalone ReAct client demo**, not used by `app.py`.

Therefore:

- The default Streamlit flow remains `app.py + src/agent.py` on the **local** backend.
- The remote backend is opt-in; if the server is down the agent **falls back to local with a warning** (never crashes).
- Both backends expose the **same eight tools** (seven specialized + `calculatrice_medicale`) and identical outputs — do not give one backend a different tool set.
- Do **not** describe the project as fully migrated to a network MCP architecture; the network mode is an optional second backend.
- Do **not** remove the local in-process architecture.

### Limitations of the real MCP network backend

- Demonstration/optional: the **local** backend is the default, validated path.
- Depends on the `mcp` SDK and the HTTP server being up and reachable (otherwise: clean fallback / skip).
- Adds per-call network latency vs. in-process calls.
- The standalone `agent2.py` ReAct loop may need multiple turns or fail to plan (capped at 5 iterations).
- `calculatrice_medicale` is demonstration-only (AST-restricted arithmetic), never a general evaluator.
- Non-clinical by construction: no clinical diagnosis or treatment recommendation.

## Phase 3 layer (ICU Multi-Data Explorer)

Phase 3 generalizes the **same single Phase 2 agent** from 7 vital signs to **25
MIMIC-IV ICU variables** (13 labs + 12 charted). It is additive: it reuses
`src/agent.py`, `src/tool_client.py`, `src/tool_trace.py`, `src/mcp_server.py`,
`src/server_mcp.py` and the Streamlit app. Still single agent, descriptive,
**non-clinical**. No new BigQuery run is needed to demo — the aggregated table is
committed.

**Active Phase 3 files (edit these for Phase 3 work):**

- `src/icu_variables.py` — canonical variable registry (single source of truth): 34 `VariableSpec`s with category, source_table, candidate itemids, unit, safe bounds, inclusion/cleaning rules. `python -m src.icu_variables` writes the dictionary CSV.
- `src/validate_icu_itemids.py` — light BigQuery validation of itemids against `d_items`/`d_labitems` (dimension tables only; degrades cleanly without auth).
- `src/extract_icu_features.py` — batched, cost-guarded BigQuery extractor. **One query per family** (labs/charted/outputs scan their event table once via `itemid IN (...)` + `QUALIFY` per-itemid cap), outcomes per-variable. Flags: `--family`, `--dry-run`, `--estimate-cost` (free BigQuery dry-run → `total_bytes_processed`), `--max-bytes-billed` (default ~5 GB, per-query cap; over-large queries fail cleanly), `--sample`/`--limit`. Writes the two CSVs below; never touches Phase 1/2 files.
- `src/tools/icu_feature_tools.py` — the **8 Phase 3 tools**: `list_available_variables`, `get_variable_summary`, `query_cohort_statistics`, `compare_age_groups`, `compare_time_windows`, `generate_evidence_card`, `plot_variable_distribution`, `detect_clinical_advice_request`. Pure functions over `icu_feature_summary.csv`, JSON-serializable.
- `src/evaluate_phase3.py` — 10-scenario Phase 3 benchmark (routing, tool, variable/age/window recognition, evidence card, non-clinical refusal, Phase 1/2 tools avoided).
- `data/processed/icu_variable_dictionary.csv` and `data/processed/icu_feature_summary.csv` — **committable** aggregated artifacts.

**Phase 2 files extended for Phase 3 (additive — keep Phase 2 behavior intact, verified by `evaluate_agent` = 5/5):** `src/tools/__init__.py` (exports), `src/mcp_server.py` (8 `ToolSpec`s; registry now 16 tools; `server_mcp.py` auto-exposes them), `src/agent.py` (safety gate first + Phase 3 intents `phase3_available_variables/variable_summary/compare_age_groups/compare_time_windows` + `clinical_advice_refused`; a "strong" Phase 3 signal overrides patient/concept routing; `_extract_calculation` and patient routing untouched), `app.py` (third mode "Phase 3 — ICU Multi-Data Explorer").

**Do NOT touch / do NOT commit:**

- `data/processed/icu_patient_features.csv` — patient-level, **gitignored, never commit**.
- `data/processed/vital_signs_elderly_icu_summary.csv` and the Phase 1/2 sources (`semantic_rag.py`, `bigquery_extract_mimic.py`, `tools/vital_summary.py`, `calculator.py`, `tool_client.py`, `server_mcp.py`) — leave unchanged.
- `data/chroma_db/` shows as modified after running the app/agent (benign SQLite/Chroma open side-effect, not a content change); restore it with `git checkout -- data/chroma_db/` before committing.
- Ignored generated outputs (do not commit): `data/evaluation/` (incl. `phase3_*`), `data/agent_traces/`, `data/processed/icu_itemid_validation.csv`, `data/phase3_outputs/`.

**Run / evaluate Phase 3:**

```bash
streamlit run app.py                              # pick "Phase 3 — ICU Multi-Data Explorer"
python -m src.evaluate_phase3 --backend local     # 10 scenarios → data/evaluation/phase3_*.{csv,md}
python -m src.evaluate_agent  --backend local     # Phase 2 regression check (must stay 5/5)
# Optional re-extraction (BigQuery, capped): see src/extract_icu_features.py --dry-run / --estimate-cost first.
```

**Non-clinical safety (must preserve):** the `detect_clinical_advice_request`
gate runs first on every question and refuses diagnosis/treatment requests; tools
are descriptive only (percentiles/medians by age group & window), never alarming
language, never fabricated numbers; `missing_rate` is **sample coverage**, not true
population missingness. Keep every answer ending with the non-clinical warning.

**Git discipline:** never `git add .`; stage only the intended Phase 3 files. Never
commit `icu_patient_features.csv` or gitignored evaluation/trace outputs.

## R analysis pipeline (`R/`)

Numbered phases `01`–`07` run in order in RStudio (`adaptive-vital-sign-thresholds.Rproj`), producing the processed CSVs the RAG layer later consumes. `01`/`02` are learning/exploration; `03`–`06` do HR cleaning, percentile-based adaptive-threshold modeling, and interpretable scoring; `07` is validation. They read/write `data/processed/*.csv` (gitignored as sensitive MIMIC data). Tidyverse/data.table based, comments in French.

## Data flow & sensitivity

```
MIMIC-IV (BigQuery)                R/ phases                 src/ingest.py
  src/bigquery_extract_mimic.py  →  clean & model        →   chunk + embed   →  ChromaDB
  → data/processed/*.csv            adaptive thresholds       (bge-m3)            (icu_rag)
                                                                                     ↓
                                                          semantic_rag.py → qwen2.5:14b answer (app.py)
```

`data/raw/`, `data/processed/`, and generated RAG artifacts (`data/rag_documents/`, `rag_chunks/`, `rag_index/`) are **gitignored** because they contain MIMIC-IV data, which is access-controlled — never commit derived patient data. `data/chroma_db/` is committed (it's the prebuilt index). Generated outputs are **not** committed either: `data/evaluation/` (retrieval + agent benchmark CSV/MD), `data/agent_traces/`, and `R/results/` figures are gitignored — regenerate them, don't track them. The repo is bilingual: code/docs in English, R comments and some commit messages in French.

## Reference docs (`docs/`)

Reference papers that are **not** ingested into the RAG index live in `docs/` (`2024.knowllm-1.6.pdf`, `s41597-022-01899-x.pdf`). Only `Rapport_Final.pdf` (at the repo root) is in `FILES_TO_INGEST` and must stay at the root for `src/ingest.py`.

## Legacy (archived — do not extend)

The TF-IDF / pre-semantic prototype has been **moved out of `src/`** into `archive/legacy_phase1_tfidf/` (with its own README): `prepare_rag_documents.py`, `chunk_documents.py`, `build_rag_index.py`, `retrieve_chunks.py`, `generate_rag_answer.py`, `rag.py`, `rag_utils.py`. They are unused by `app.py` and the active Phase 1 / Phase 2 pipelines, and are frozen (their relative imports assume the old `src/` layout — not runnable in place). Do not import or extend them.

`src/bigquery_extract_mimic.py` is **kept in `src/`** as the upstream MIMIC-IV → CSV extractor (data provenance). It is now self-contained — the `age_group_from_age` / `time_window_from_hours` helpers were inlined when `rag_utils` was archived. It still needs GCP + MIMIC-IV access and is **not** part of the runtime app. Prefer the active pipeline above.
