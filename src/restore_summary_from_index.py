"""Rebuild ``data/processed/vital_signs_elderly_icu_summary.csv`` from ChromaDB.

The MIMIC-IV summary CSV is gitignored (it is a derived aggregate of
access-controlled data) and can be lost from a fresh checkout while the
embedded ``mimic_stats`` documents still live in the committed ChromaDB index.

This utility reads those indexed ``mimic_stats`` documents back, parses the
fixed ``Key: value.`` lines produced by ``src.ingest._summary_row_to_document``,
and writes the aggregate summary CSV back to disk. It restores ONLY aggregate
statistics (counts, means, percentiles) -- never raw patient rows.

Run once before a clean rebuild when the processed CSV is missing:

    python -m src.restore_summary_from_index
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import chromadb
import pandas as pd

try:
    from .config import PROCESSED_DIR, ROOT_DIR, ensure_data_directories
except ImportError:  # pragma: no cover - direct script execution fallback
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.config import PROCESSED_DIR, ROOT_DIR, ensure_data_directories

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

COLLECTION_NAME = "icu_rag"
CHROMA_DB_DIR = ROOT_DIR / "data" / "chroma_db"
SUMMARY_CSV = PROCESSED_DIR / "vital_signs_elderly_icu_summary.csv"

# Maps the human-readable document line label to the CSV column name.
DOC_LINE_TO_COLUMN = {
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

COLUMN_ORDER = [
    "vital_sign",
    "age_group",
    "time_window",
    "itemid",
    "label",
    "unitname",
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
    "is_demo_data",
]


def _parse_numeric(raw: str) -> float | str | None:
    cleaned = raw.strip().rstrip(".").strip()
    if cleaned in {"", "None", "nan", "NaN"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return cleaned


def _parse_document(document: str) -> dict[str, object]:
    values: dict[str, object] = {}
    for line in document.splitlines():
        if ":" not in line:
            continue
        key, _, raw_value = line.partition(":")
        column = DOC_LINE_TO_COLUMN.get(key.strip())
        if column is not None:
            values[column] = _parse_numeric(raw_value)
    return values


def restore_summary_csv() -> Path:
    ensure_data_directories()
    if not CHROMA_DB_DIR.exists():
        raise FileNotFoundError(f"ChromaDB not found at {CHROMA_DB_DIR}; nothing to restore from.")

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    payload = collection.get(include=["metadatas", "documents"])

    rows: list[dict[str, object]] = []
    for metadata, document in zip(payload["metadatas"], payload["documents"]):
        if str(metadata.get("source_type")) != "mimic_stats":
            continue
        row: dict[str, object] = {
            "vital_sign": metadata.get("vital_sign", ""),
            "age_group": metadata.get("age_group", ""),
            "time_window": metadata.get("time_window", ""),
            "itemid": metadata.get("itemid", ""),
            "label": metadata.get("label", ""),
            "unitname": metadata.get("unitname", ""),
            "is_demo_data": bool(metadata.get("is_demo_data", False)),
        }
        row.update(_parse_document(document))
        rows.append(row)

    if not rows:
        raise RuntimeError("No mimic_stats documents found in the index; cannot restore the summary CSV.")

    dataframe = pd.DataFrame(rows)
    for column in COLUMN_ORDER:
        if column not in dataframe.columns:
            dataframe[column] = None
    dataframe = dataframe[COLUMN_ORDER]

    # The historical index holds a precise and a rounded copy of every row
    # (e.g. mean 77.755 alongside 77.8). Normalise the itemid and keep, per
    # (vital_sign, age_group, time_window, itemid), the most precise copy so
    # downstream tools see exactly one deterministic row per item.
    dataframe["itemid"] = (
        dataframe["itemid"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    )

    def _mean_precision(value: object) -> int:
        text = str(value)
        return len(text.split(".", 1)[1]) if "." in text else 0

    dataframe["_precision"] = dataframe["mean"].map(_mean_precision)
    dataframe = (
        dataframe.sort_values("_precision", ascending=False)
        .drop_duplicates(subset=["vital_sign", "age_group", "time_window", "itemid"], keep="first")
        .drop(columns="_precision")
        .sort_values(["vital_sign", "age_group", "time_window", "itemid"])
        .reset_index(drop=True)
    )

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(SUMMARY_CSV, index=False)
    LOGGER.info("Restored %s rows to %s", len(dataframe), SUMMARY_CSV)
    return SUMMARY_CSV


def main() -> None:
    restore_summary_csv()


if __name__ == "__main__":
    main()
