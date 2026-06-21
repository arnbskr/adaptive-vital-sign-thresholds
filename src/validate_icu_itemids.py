"""Light-weight itemid validation against MIMIC-IV dimension tables.

Phase 3 reads candidate ``itemid``s from ``src/icu_variables.py``. Before any
(potentially costly) feature extraction, this script confirms that those ids
really exist and what their official ``label`` / ``unitname`` are, by querying
ONLY the small dimension tables ``d_items`` (ICU) and ``d_labitems`` (hosp).

These are dimension lookups (a few thousand rows), so the queries are cheap --
unlike the event tables. The script:

* groups candidate itemids by their dimension table,
* runs one ``SELECT ... WHERE itemid IN (...)`` per table,
* prints a found/missing report comparing official labels to our registry,
* writes ``data/processed/icu_itemid_validation.csv`` (gitignored) for review.

It degrades gracefully: if BigQuery / credentials are unavailable it explains
how to authenticate and exits without crashing. It NEVER touches event tables
and writes nothing to the index.
"""

from __future__ import annotations

import logging

import pandas as pd

from .config import HOSP_DATASET, ICU_DATASET, PROCESSED_DIR, PROJECT_ID, ensure_data_directories
from .icu_variables import ICU_VARIABLES, VariableSpec

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

VALIDATION_CSV = PROCESSED_DIR / "icu_itemid_validation.csv"

# Map a spec's source_table prefix to the dimension table + dataset used to
# resolve its itemids.
_DIM_FOR_SOURCE = {
    "icu.chartevents": (f"{ICU_DATASET}.d_items", "icu.d_items"),
    "icu.outputevents": (f"{ICU_DATASET}.d_items", "icu.d_items"),
    "icu.inputevents": (f"{ICU_DATASET}.d_items", "icu.d_items"),
    "icu.procedureevents": (f"{ICU_DATASET}.d_items", "icu.d_items"),
    "hosp.labevents": (f"{HOSP_DATASET}.d_labitems", "hosp.d_labitems"),
}


def _specs_with_itemids() -> list[VariableSpec]:
    return [spec for spec in ICU_VARIABLES if spec.itemids]


def _candidate_index() -> dict[str, dict[int, str]]:
    """Return {dim_table_fqn: {itemid: variable_name}} for all itemid-based specs."""

    index: dict[str, dict[int, str]] = {}
    for spec in _specs_with_itemids():
        mapping = _DIM_FOR_SOURCE.get(spec.source_table)
        if mapping is None:
            LOGGER.warning("No dimension table mapping for source %s (%s); skipping.",
                           spec.source_table, spec.variable_name)
            continue
        dim_fqn, _ = mapping
        for itemid in spec.itemids:
            index.setdefault(dim_fqn, {})[itemid] = spec.variable_name
    return index


def _build_client():
    """Build a BigQuery client lazily so the module imports without the SDK/auth."""

    from google.cloud import bigquery  # local import: only needed when querying

    return bigquery.Client(project=PROJECT_ID)


def _query_dimension(client, dim_fqn: str, itemids: list[int]) -> pd.DataFrame:
    is_labitems = dim_fqn.endswith("d_labitems")
    unit_col = "NULL AS unitname" if is_labitems else "unitname"
    extra = "fluid, category" if is_labitems else "abbreviation, category, unitname AS unit2"
    # d_labitems has (label, fluid, category); d_items has (label, abbreviation,
    # category, unitname). Select a common, robust subset.
    if is_labitems:
        select = "itemid, label, fluid, category, NULL AS unitname"
    else:
        select = "itemid, label, abbreviation, category, unitname"
    id_list = ", ".join(str(int(i)) for i in sorted(set(itemids)))
    query = f"SELECT {select} FROM `{dim_fqn}` WHERE itemid IN ({id_list}) ORDER BY itemid"
    return client.query(query).to_dataframe()


def validate() -> pd.DataFrame | None:
    """Validate candidate itemids. Returns the report frame, or None if unavailable."""

    ensure_data_directories()
    index = _candidate_index()
    expected = {itemid: var for mapping in index.values() for itemid, var in mapping.items()}

    try:
        client = _build_client()
        client.query("SELECT 1 AS ok").to_dataframe()
    except Exception as exc:  # noqa: BLE001 - degrade gracefully if no auth/SDK
        LOGGER.error(
            "BigQuery is unavailable (%s). Authenticate with `gcloud auth application-default login` "
            "and ensure access to %s / %s, then re-run `python -m src.validate_icu_itemids`.",
            exc, ICU_DATASET, HOSP_DATASET,
        )
        return None

    found_rows: list[dict] = []
    found_ids: set[int] = set()
    for dim_fqn, mapping in index.items():
        LOGGER.info("Validating %s candidate itemids against %s", len(mapping), dim_fqn)
        try:
            frame = _query_dimension(client, dim_fqn, list(mapping))
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Query against %s failed: %s", dim_fqn, exc)
            continue
        for _, row in frame.iterrows():
            itemid = int(row["itemid"])
            found_ids.add(itemid)
            found_rows.append({
                "itemid": itemid,
                "variable_name": mapping.get(itemid, ""),
                "official_label": row.get("label"),
                "official_unit": row.get("unitname"),
                "category": row.get("category"),
                "dimension_table": dim_fqn.split(".")[-1],
                "status": "FOUND",
            })

    missing_rows = [
        {"itemid": itemid, "variable_name": var, "official_label": None,
         "official_unit": None, "category": None, "dimension_table": None, "status": "MISSING"}
        for itemid, var in expected.items() if itemid not in found_ids
    ]

    report = pd.DataFrame(found_rows + missing_rows).sort_values(
        ["status", "variable_name", "itemid"], na_position="last"
    ).reset_index(drop=True)

    report.to_csv(VALIDATION_CSV, index=False)
    n_found = len(found_ids)
    n_total = len(expected)
    LOGGER.info("Validated %s/%s candidate itemids. Report: %s", n_found, n_total, VALIDATION_CSV)
    if missing_rows:
        LOGGER.warning(
            "%s itemids were NOT found and must be corrected in src/icu_variables.py: %s",
            len(missing_rows), ", ".join(f"{r['itemid']}({r['variable_name']})" for r in missing_rows),
        )
    return report


def main() -> None:
    report = validate()
    if report is None:
        return
    print("\n=== ITEMID VALIDATION REPORT ===")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(report.to_string(index=False))


if __name__ == "__main__":
    main()
