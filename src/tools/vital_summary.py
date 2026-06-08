"""Tool: get_vital_summary.

Retrieves the exact MIMIC-IV statistical summary for a (vital_sign, age_group,
time_window) triple. The processed CSV is the primary source; if it is missing
(it is gitignored), the loader falls back to the ``mimic_stats`` documents that
already live in the ChromaDB index, parsing them back into the same structure.

When several itemids exist for one vital sign (e.g. MAP via arterial line and
via non-invasive cuff), the representative row is the one with the largest
``count`` (most observations); the alternatives are reported alongside.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import ROOT_DIR, VITAL_SUMMARY_CSV

LOGGER = logging.getLogger(__name__)

NUMERIC_COLUMNS = [
    "count",
    "mean",
    "median",
    "p5",
    "p25",
    "p50",
    "p75",
    "p90",
    "standard_low",
    "standard_high",
    "percent_below_standard_low",
    "percent_above_standard_high",
]

SUMMARY_REL_PATH = str(VITAL_SUMMARY_CSV.relative_to(ROOT_DIR)).replace("\\", "/")

# Maps the human-readable ChromaDB document line label to the CSV column name,
# mirroring src.ingest._summary_row_to_document so the fallback stays faithful.
_DOC_LINE_TO_COLUMN = {
    "Count": "count",
    "Mean": "mean",
    "Median": "median",
    "P5": "p5",
    "P25": "p25",
    "P50": "p50",
    "P75": "p75",
    "P90": "p90",
    "Standard low threshold": "standard_low",
    "Standard high threshold": "standard_high",
    "Percent below standard low": "percent_below_standard_low",
    "Percent above standard high": "percent_above_standard_high",
}


def _clean_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _parse_chroma_document(document: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for line in str(document).splitlines():
        if ":" not in line:
            continue
        key, _, raw_value = line.partition(":")
        column = _DOC_LINE_TO_COLUMN.get(key.strip())
        if column is None:
            continue
        cleaned = raw_value.strip().rstrip(".").strip()
        if cleaned in {"", "None", "nan", "NaN"}:
            values[column] = None
        else:
            try:
                values[column] = float(cleaned)
            except ValueError:
                values[column] = cleaned
    return values


def _load_from_chroma() -> pd.DataFrame:
    """Rebuild the summary table from indexed mimic_stats documents."""

    import chromadb  # local import: only needed when the CSV is absent

    chroma_dir = ROOT_DIR / "data" / "chroma_db"
    if not chroma_dir.exists():
        return pd.DataFrame()

    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        collection = client.get_collection(name="icu_rag")
    except Exception:  # noqa: BLE001 - collection may not exist yet
        return pd.DataFrame()

    payload = collection.get(include=["metadatas", "documents"])
    rows: list[dict[str, Any]] = []
    for metadata, document in zip(payload.get("metadatas", []), payload.get("documents", [])):
        if str(metadata.get("source_type")) != "mimic_stats":
            continue
        row: dict[str, Any] = {
            "vital_sign": metadata.get("vital_sign", ""),
            "age_group": metadata.get("age_group", ""),
            "time_window": metadata.get("time_window", ""),
            "itemid": str(metadata.get("itemid", "")).replace(".0", ""),
            "label": metadata.get("label", ""),
            "unitname": metadata.get("unitname", ""),
        }
        row.update(_parse_chroma_document(document))
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).drop_duplicates(
        subset=["vital_sign", "age_group", "time_window", "itemid"], keep="first"
    )
    return frame


def load_summary_dataframe() -> tuple[pd.DataFrame, str]:
    """Return (dataframe, source_label). Prefers the CSV, falls back to ChromaDB."""

    if Path(VITAL_SUMMARY_CSV).exists():
        frame = pd.read_csv(VITAL_SUMMARY_CSV)
        frame["itemid"] = frame["itemid"].astype(str).str.replace(r"\.0$", "", regex=True)
        return frame, SUMMARY_REL_PATH

    LOGGER.warning("Summary CSV missing at %s; falling back to ChromaDB mimic_stats docs.", SUMMARY_REL_PATH)
    frame = _load_from_chroma()
    return frame, "chromadb:icu_rag/mimic_stats"


def find_summary_rows(
    vital_sign: str, age_group: str, time_window: str
) -> tuple[pd.DataFrame, str]:
    """Return all matching rows (possibly several itemids) for the exact triple."""

    frame, source_label = load_summary_dataframe()
    if frame.empty:
        return frame, source_label
    mask = (
        (frame["vital_sign"].astype(str) == str(vital_sign))
        & (frame["age_group"].astype(str) == str(age_group))
        & (frame["time_window"].astype(str) == str(time_window))
    )
    return frame[mask].copy(), source_label


def get_vital_summary(vital_sign: str, age_group: str, time_window: str) -> dict[str, Any]:
    """Return the exact statistical summary for the requested triple.

    Returns ``{"error": "No summary found."}`` if no exact match exists, so the
    agent can refuse to invent a comparison.
    """

    matches, source_label = find_summary_rows(vital_sign, age_group, time_window)
    if matches.empty:
        return {"error": "No summary found.", "requested": {
            "vital_sign": vital_sign, "age_group": age_group, "time_window": time_window,
        }}

    # Deterministic representative: the itemid with the most observations.
    matches = matches.sort_values("count", ascending=False, na_position="last")
    row = matches.iloc[0]

    alternative_itemids = [str(item) for item in matches["itemid"].tolist()[1:]]

    summary: dict[str, Any] = {
        "vital_sign": str(row.get("vital_sign", vital_sign)),
        "age_group": str(row.get("age_group", age_group)),
        "time_window": str(row.get("time_window", time_window)),
        "itemid": str(row.get("itemid", "")),
        "label": str(row.get("label", "")),
        "unitname": str(row.get("unitname", "")),
        "source_file": source_label,
    }
    for column in NUMERIC_COLUMNS:
        summary[column] = _clean_number(row.get(column))

    if alternative_itemids:
        summary["alternative_itemids"] = alternative_itemids
        summary["selection_note"] = (
            "Multiple itemids exist for this vital sign; the row with the largest count was selected."
        )
    return summary
