"""Phase 3 ICU feature extraction (multi-family, descriptive only).

Extends the project beyond the seven vital signs to several MIMIC-IV ICU
variable families (labs, additional charted variables, simple outputs, and
simple aggregated outcomes), driven entirely by the canonical registry in
``src/icu_variables.py`` (single source of truth).

Design goals (and the constraints they honor):

* **Additive, never destructive.** Writes only NEW files
  (``icu_feature_summary.csv`` aggregated/committable,
  ``icu_patient_features.csv`` patient-level/gitignored). The Phase 1/2 vital
  files are never touched. Re-running one family merges into, rather than
  clobbers, the others.
* **Family-scoped + cost-aware.** ``--family labs|charted|outputs|outcomes|all``
  and explicit ``--elderly-limit/--event-limit/--lab-limit`` (or ``--sample`` /
  ``--limit`` for a tiny test). ``--dry-run`` prints the generated SQL and the
  tables touched and writes nothing -- use it before any real run.
* **Shared cohort.** Elderly ICU stays (``anchor_age >= 65`` joined to
  ``icustays``), the SAME age groups (65-74/75-84/85+) and time windows
  (first_6h/12h/24h) as the existing pipeline.
* **Safety / quality.** Per-variable physiologic "safe" bounds and unit
  handling (Temperature F->C, FiO2 fraction->percent, GCS component sum) come
  from the registry; out-of-range values are excluded and counted. Descriptive
  statistics only -- no diagnosis, no clinical recommendation.

This module imports BigQuery lazily, so it byte-compiles and ``--dry-run`` works
without credentials or the SDK.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .config import HOSP_DATASET, ICU_DATASET, PROCESSED_DIR, PROJECT_ID, ensure_data_directories
from .icu_variables import ICU_VARIABLES, VariableSpec, variables_by_category

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# ---- Limits (overridable via CLI) ----------------------------------------- #
ELDERLY_LIMIT_DEFAULT = 1000   # elderly ICU stays in the shared cohort
EVENT_LIMIT_DEFAULT = 10000    # chartevents/outputevents rows per variable
LAB_LIMIT_DEFAULT = 10000      # labevents rows per lab variable

# ---- Outputs (new files only; Phase 1/2 files are never written) ---------- #
FEATURE_SUMMARY_CSV = PROCESSED_DIR / "icu_feature_summary.csv"     # aggregated, committable
PATIENT_FEATURES_CSV = PROCESSED_DIR / "icu_patient_features.csv"   # patient-level, gitignored

FAMILIES = ("labs", "charted", "outputs", "outcomes")

SUMMARY_COLUMNS = [
    "variable_name", "variable_category", "source_table", "itemids", "unit",
    "age_group", "time_window", "n_patients", "n_measurements",
    "mean", "std", "min", "max", "median",
    "p05", "p25", "p50", "p75", "p90", "p95", "missing_rate",
]

PATIENT_KEY_COLS = ["subject_id", "hadm_id", "stay_id", "age_group", "gender", "time_window"]

# Special time window used for stay-level outcomes (no intra-stay window).
OUTCOME_WINDOW = "full_stay"

# Per-query safety cap: a real query that would scan more than this fails cleanly
# instead of running up a bill. ~5 GB; override with --max-bytes-billed.
MAX_BYTES_BILLED_DEFAULT = 5 * 1024 ** 3

# Every fully-qualified table the builders can reference, with cost class.
_ALL_TABLES = [
    f"{HOSP_DATASET}.patients", f"{ICU_DATASET}.icustays",
    f"{ICU_DATASET}.chartevents", f"{ICU_DATASET}.outputevents",
    f"{HOSP_DATASET}.labevents", f"{HOSP_DATASET}.admissions",
]
_LARGE_TABLES = {"chartevents", "outputevents", "labevents"}


def _tables_in_sql(sql: str) -> set[str]:
    return {t for t in _ALL_TABLES if t.split(".")[-1] in sql}


def _table_cost_label(table_fqn: str) -> str:
    return "LARGE (event table -- main cost)" if table_fqn.split(".")[-1] in _LARGE_TABLES \
        else "small (dimension/stay table)"


def _fmt_bytes(n: int | None) -> str:
    if n is None:
        return "unknown"
    mb = n / 1024 ** 2
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.3f} GB ({mb:,.1f} MB)"
    return f"{mb:.1f} MB"


# --------------------------------------------------------------------------- #
# Canonical bucketing (mirrors the existing pipeline; disjoint windows).
# --------------------------------------------------------------------------- #
def age_group_from_age(age: float | int | None) -> str:
    if age is None or pd.isna(age):
        return "Unknown"
    if age >= 85:
        return "85+"
    if age >= 75:
        return "75-84"
    return "65-74"


def time_window_from_hours(hours: float) -> str | None:
    if hours < 0:
        return None
    if hours <= 6:
        return "first_6h"
    if hours <= 12:
        return "first_12h"
    if hours <= 24:
        return "first_24h"
    return None


# --------------------------------------------------------------------------- #
# SQL builders (pure strings; printed verbatim in --dry-run).
# --------------------------------------------------------------------------- #
def _cohort_cte(elderly_limit: int) -> str:
    return f"""elderly_icu AS (
  SELECT p.subject_id, p.gender, p.anchor_age, i.hadm_id, i.stay_id, i.intime
  FROM `{HOSP_DATASET}.patients` p
  JOIN `{ICU_DATASET}.icustays` i ON p.subject_id = i.subject_id
  WHERE p.anchor_age >= 65
  LIMIT {int(elderly_limit)}
)"""


def cohort_size_query(elderly_limit: int) -> str:
    return f"WITH {_cohort_cte(elderly_limit)}\nSELECT subject_id, anchor_age FROM elderly_icu"


def build_family_query(
    family: str, specs: list[VariableSpec], elderly_limit: int, event_limit: int, lab_limit: int
) -> str:
    """One batched query per family: scan the event table ONCE for ALL itemids.

    Instead of one query per variable (N scans of the same big table), this
    unions every itemid of the family into a single ``itemid IN (...)`` scan and
    dispatches back to variables in pandas. A per-itemid row cap
    (``QUALIFY ROW_NUMBER``) bounds memory; results match the per-variable form
    whenever no variable hit its cap (and stay deterministic by charttime if it
    does). LIMIT does not change bytes scanned, so this is purely a cost win.
    """

    cohort = _cohort_cte(elderly_limit)
    itemids = sorted({int(i) for spec in specs for i in spec.itemids})
    ids = ", ".join(str(i) for i in itemids)

    if family == "labs":
        cap = int(lab_limit)
        # labevents has no stay_id; dedup hadm -> earliest ICU stay to avoid
        # double-counting labs when an admission has multiple ICU stays.
        return f"""WITH {cohort},
elderly_adm AS (
  SELECT subject_id, gender, anchor_age, hadm_id, stay_id, intime FROM (
    SELECT subject_id, gender, anchor_age, hadm_id, stay_id, intime,
           ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY intime) AS rn
    FROM elderly_icu
  ) WHERE rn = 1
)
SELECT e.subject_id, e.hadm_id, e.stay_id, e.anchor_age, e.gender, e.intime,
       l.charttime, l.itemid, l.valuenum AS value
FROM elderly_adm e
JOIN `{HOSP_DATASET}.labevents` l ON e.hadm_id = l.hadm_id
WHERE l.itemid IN ({ids})
  AND l.valuenum IS NOT NULL
  AND l.charttime >= e.intime
  AND l.charttime < TIMESTAMP_ADD(e.intime, INTERVAL 24 HOUR)
QUALIFY ROW_NUMBER() OVER (PARTITION BY l.itemid ORDER BY l.charttime) <= {cap}"""

    if family == "outputs":
        cap = int(event_limit)
        return f"""WITH {cohort}
SELECT e.subject_id, e.hadm_id, e.stay_id, e.anchor_age, e.gender, e.intime,
       o.charttime, o.itemid, o.value AS value
FROM elderly_icu e
JOIN `{ICU_DATASET}.outputevents` o ON e.stay_id = o.stay_id
WHERE o.itemid IN ({ids})
  AND o.value IS NOT NULL
  AND o.charttime >= e.intime
  AND o.charttime < TIMESTAMP_ADD(e.intime, INTERVAL 24 HOUR)
QUALIFY ROW_NUMBER() OVER (PARTITION BY o.itemid ORDER BY o.charttime) <= {cap}"""

    # charted: chartevents (charted vitals + GCS components, both via valuenum).
    cap = int(event_limit)
    return f"""WITH {cohort}
SELECT e.subject_id, e.hadm_id, e.stay_id, e.anchor_age, e.gender, e.intime,
       c.charttime, c.itemid, c.valuenum AS value
FROM elderly_icu e
JOIN `{ICU_DATASET}.chartevents` c ON e.stay_id = c.stay_id
WHERE c.itemid IN ({ids})
  AND c.valuenum IS NOT NULL
  AND c.charttime >= e.intime
  AND c.charttime < TIMESTAMP_ADD(e.intime, INTERVAL 24 HOUR)
QUALIFY ROW_NUMBER() OVER (PARTITION BY c.itemid ORDER BY c.charttime) <= {cap}"""


def build_los_query(spec: VariableSpec, elderly_limit: int) -> str:
    """Stay/admission-level numeric outcome (icu_los, hospital_los)."""

    cohort = _cohort_cte(elderly_limit)
    if spec.variable_name == "icu_los":
        return f"""WITH {cohort}
SELECT e.subject_id, e.hadm_id, e.stay_id, e.anchor_age, e.gender, i.los AS value
FROM elderly_icu e
JOIN `{ICU_DATASET}.icustays` i ON e.stay_id = i.stay_id
WHERE i.los IS NOT NULL"""
    # hospital_los derived from admissions admit/disch timestamps.
    return f"""WITH {cohort}
SELECT e.subject_id, e.hadm_id, e.stay_id, e.anchor_age, e.gender,
       TIMESTAMP_DIFF(a.dischtime, a.admittime, HOUR) / 24.0 AS value
FROM elderly_icu e
JOIN `{HOSP_DATASET}.admissions` a ON e.hadm_id = a.hadm_id
WHERE a.dischtime IS NOT NULL AND a.admittime IS NOT NULL"""


# --------------------------------------------------------------------------- #
# Cleaning / transforms (registry-driven; exclusions are counted).
# --------------------------------------------------------------------------- #
def _apply_unit_transforms(spec: VariableSpec, frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    if spec.variable_name == "temperature":
        # itemid 223761 is Fahrenheit -> convert to Celsius.
        mask_f = frame["itemid"].astype("Int64") == 223761
        frame.loc[mask_f, "value"] = (frame.loc[mask_f, "value"] - 32.0) * 5.0 / 9.0
    if spec.variable_name == "fio2":
        # Fractions in [0, 1] are recorded instead of percent; normalize.
        mask_frac = frame["value"] <= 1.0
        frame.loc[mask_frac, "value"] = frame.loc[mask_frac, "value"] * 100.0
    return frame


def _collapse_gcs(frame: pd.DataFrame) -> pd.DataFrame:
    """Sum GCS eye/verbal/motor components at matching (stay, charttime)."""

    if frame.empty:
        return frame
    keys = ["subject_id", "hadm_id", "stay_id", "anchor_age", "gender", "intime", "charttime"]
    wide = frame.pivot_table(index=keys, columns="itemid", values="value", aggfunc="max")
    needed = [c for c in (220739, 223900, 223901) if c in wide.columns]
    if len(needed) < 3:
        LOGGER.warning("GCS: missing component itemids %s; cannot sum a total.", needed)
        return pd.DataFrame(columns=keys + ["itemid", "value"])
    wide = wide.dropna(subset=needed)
    total = wide[needed].sum(axis=1).reset_index()
    total = total.rename(columns={0: "value"})
    total["itemid"] = -1  # synthetic composite itemid
    return total


def clean_variable_frame(spec: VariableSpec, raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Transform, window-bucket, safe-bound-filter and dedup one variable.

    Returns (clean_frame, exclusion_stats). The clean frame has standard columns
    [subject_id, hadm_id, stay_id, age_group, gender, time_window, variable_name,
    variable_category, value].
    """

    stats = {"raw_rows": int(len(raw)), "after_bounds": 0, "after_window": 0, "final": 0}
    if raw.empty:
        empty = pd.DataFrame(columns=PATIENT_KEY_COLS + ["variable_name", "variable_category", "value"])
        return empty, stats

    if spec.composite:
        frame = _collapse_gcs(raw)
    else:
        frame = _apply_unit_transforms(spec, raw)

    frame = frame.dropna(subset=["value"])

    # Safe physiologic bounds (inclusive) drop obvious outliers / errors.
    if spec.safe_low is not None:
        frame = frame[frame["value"] >= spec.safe_low]
    if spec.safe_high is not None:
        frame = frame[frame["value"] <= spec.safe_high]
    stats["after_bounds"] = int(len(frame))

    # Time window relative to ICU intime (disjoint 6h/12h/24h buckets).
    intime = pd.to_datetime(frame["intime"], errors="coerce")
    charttime = pd.to_datetime(frame["charttime"], errors="coerce")
    hours = (charttime - intime).dt.total_seconds() / 3600.0
    frame = frame.assign(time_window=hours.apply(
        lambda h: time_window_from_hours(h) if pd.notna(h) else None))
    frame = frame[frame["time_window"].notna()]
    stats["after_window"] = int(len(frame))

    frame = frame.assign(age_group=frame["anchor_age"].apply(age_group_from_age))
    # Deduplicate identical measurements.
    frame = frame.drop_duplicates(subset=["subject_id", "stay_id", "charttime", "itemid", "value"])
    frame = frame.assign(variable_name=spec.variable_name, variable_category=spec.variable_category)
    stats["final"] = int(len(frame))

    cols = PATIENT_KEY_COLS + ["variable_name", "variable_category", "value"]
    return frame[cols].copy(), stats


def clean_outcome_frame(spec: VariableSpec, raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {"raw_rows": int(len(raw)), "after_bounds": int(len(raw)), "after_window": int(len(raw)), "final": 0}
    if raw.empty:
        empty = pd.DataFrame(columns=PATIENT_KEY_COLS + ["variable_name", "variable_category", "value"])
        return empty, stats
    frame = raw.copy()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame[frame["value"] >= 0].dropna(subset=["value"])  # drop implausible LOS
    frame = frame.assign(
        age_group=frame["anchor_age"].apply(age_group_from_age),
        time_window=OUTCOME_WINDOW,
        variable_name=spec.variable_name,
        variable_category=spec.variable_category,
    )
    stats["final"] = int(len(frame))
    cols = PATIENT_KEY_COLS + ["variable_name", "variable_category", "value"]
    return frame[cols].copy(), stats


# --------------------------------------------------------------------------- #
# Aggregation.
# --------------------------------------------------------------------------- #
def summarize_variable(spec: VariableSpec, clean: pd.DataFrame, cohort_sizes: dict[str, int]) -> list[dict[str, Any]]:
    if clean.empty:
        return []
    rows: list[dict[str, Any]] = []
    for (age_group, time_window), group in clean.groupby(["age_group", "time_window"], dropna=False):
        values = pd.to_numeric(group["value"], errors="coerce").dropna()
        if values.empty:
            continue
        n_patients = int(group["subject_id"].nunique())
        denom = cohort_sizes.get(age_group)
        missing_rate = round(1.0 - n_patients / denom, 4) if denom else None
        if missing_rate is not None:
            missing_rate = max(0.0, min(1.0, missing_rate))
        rows.append({
            "variable_name": spec.variable_name,
            "variable_category": spec.variable_category,
            "source_table": spec.source_table,
            "itemids": spec.itemid_str(),
            "unit": spec.unit,
            "age_group": age_group,
            "time_window": time_window,
            "n_patients": n_patients,
            "n_measurements": int(values.count()),
            "mean": round(float(values.mean()), 4),
            "std": round(float(values.std(ddof=1)), 4) if values.count() > 1 else 0.0,
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
            "median": round(float(values.median()), 4),
            "p05": round(float(values.quantile(0.05)), 4),
            "p25": round(float(values.quantile(0.25)), 4),
            "p50": round(float(values.quantile(0.50)), 4),
            "p75": round(float(values.quantile(0.75)), 4),
            "p90": round(float(values.quantile(0.90)), 4),
            "p95": round(float(values.quantile(0.95)), 4),
            "missing_rate": missing_rate,
        })
    return rows


def build_patient_features(clean_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Wide patient-level matrix: one row per (stay, window), one col per variable.

    Outputs are summed per stay/window; other variables use the per-stay/window
    mean. Outcome variables (full_stay window) are intentionally excluded here.
    """

    parts: list[pd.DataFrame] = []
    for frame in clean_frames:
        if frame.empty:
            continue
        spec_name = str(frame["variable_name"].iloc[0])
        category = str(frame["variable_category"].iloc[0])
        if category == "outcome":
            continue
        aggfunc = "sum" if category == "output" else "mean"
        feature = (
            frame.groupby(PATIENT_KEY_COLS, dropna=False)["value"].agg(aggfunc)
            .round(4).rename(spec_name).reset_index()
        )
        parts.append(feature)
    if not parts:
        return pd.DataFrame(columns=PATIENT_KEY_COLS)
    wide = parts[0]
    for part in parts[1:]:
        wide = wide.merge(part, on=PATIENT_KEY_COLS, how="outer")
    return wide


# --------------------------------------------------------------------------- #
# Merge + write (additive; never clobbers other families).
# --------------------------------------------------------------------------- #
def _merge_summary(new_rows: list[dict[str, Any]], extracted_vars: set[str]) -> pd.DataFrame:
    new = pd.DataFrame(new_rows, columns=SUMMARY_COLUMNS)
    if FEATURE_SUMMARY_CSV.exists():
        existing = pd.read_csv(FEATURE_SUMMARY_CSV)
        existing = existing[~existing["variable_name"].isin(extracted_vars)]
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new
    return combined.sort_values(
        ["variable_category", "variable_name", "age_group", "time_window"]
    ).reset_index(drop=True)


def _merge_patient_features(new_wide: pd.DataFrame, extracted_vars: set[str]) -> pd.DataFrame:
    if new_wide.empty:
        if PATIENT_FEATURES_CSV.exists():
            return pd.read_csv(PATIENT_FEATURES_CSV)
        return new_wide
    if not PATIENT_FEATURES_CSV.exists():
        return new_wide
    existing = pd.read_csv(PATIENT_FEATURES_CSV)
    # Drop feature columns being refreshed, then outer-merge on the keys.
    drop_cols = [c for c in existing.columns if c in extracted_vars]
    existing = existing.drop(columns=drop_cols)
    return existing.merge(new_wide, on=PATIENT_KEY_COLS, how="outer")


# --------------------------------------------------------------------------- #
# BigQuery access (lazy) and cohort sizing.
# --------------------------------------------------------------------------- #
def _build_client():
    from google.cloud import bigquery  # local import: only needed for real runs

    return bigquery.Client(project=PROJECT_ID)


def _billed_job_config(max_bytes_billed: int):
    """Real-run job config: cap bytes scanned so an over-large query fails cleanly."""

    from google.cloud import bigquery  # local import

    return bigquery.QueryJobConfig(maximum_bytes_billed=int(max_bytes_billed), use_query_cache=False)


def fetch_cohort_sizes(client, elderly_limit: int, job_config=None) -> dict[str, int]:
    frame = client.query(cohort_size_query(elderly_limit), job_config=job_config).to_dataframe()
    if frame.empty:
        return {}
    frame = frame.assign(age_group=frame["anchor_age"].apply(age_group_from_age))
    sizes = frame.groupby("age_group")["subject_id"].nunique().to_dict()
    return {str(k): int(v) for k, v in sizes.items()}


def _specs_for_families(families: list[str]) -> list[VariableSpec]:
    selected: list[VariableSpec] = []
    if "charted" in families:
        selected += variables_by_category("vital_sign")
    if "labs" in families:
        selected += variables_by_category("lab")
    if "outputs" in families:
        selected += variables_by_category("output")
    if "outcomes" in families:
        # MVP: only numeric LOS outcomes fit the descriptive-distribution schema.
        selected += [s for s in variables_by_category("outcome") if s.variable_name in ("icu_los", "hospital_los")]
        skipped = [s.variable_name for s in variables_by_category("outcome")
                   if s.variable_name not in ("icu_los", "hospital_los")]
        if skipped:
            LOGGER.info("Outcomes (categorical/mortality) deferred post-MVP: %s", ", ".join(skipped))
    return selected


# --------------------------------------------------------------------------- #
# Query plan: event families -> ONE batched query; outcomes -> per-variable.
# --------------------------------------------------------------------------- #
@dataclass
class QueryUnit:
    """One query that will actually run, plus the variables it feeds."""

    label: str
    sql: str
    specs: list[VariableSpec]


_CATEGORY_TO_FAMILY = {
    "vital_sign": "charted", "lab": "labs", "output": "outputs",
    "outcome": "outcomes", "input": "inputs", "procedure": "procedures",
}


def plan_queries(specs: list[VariableSpec], args: argparse.Namespace) -> list[QueryUnit]:
    """Group selected specs into the minimal set of queries.

    Event families (labs/charted/outputs) collapse to ONE batched query each
    (single table scan); numeric outcomes stay per-variable (small stay-level
    tables, negligible cost).
    """

    by_family: dict[str, list[VariableSpec]] = {}
    outcomes: list[VariableSpec] = []
    for spec in specs:
        if spec.variable_category == "outcome":
            outcomes.append(spec)
            continue
        by_family.setdefault(_CATEGORY_TO_FAMILY[spec.variable_category], []).append(spec)

    units: list[QueryUnit] = []
    for family in ("labs", "charted", "outputs"):
        fam_specs = by_family.get(family)
        if not fam_specs:
            continue
        sql = build_family_query(family, fam_specs, args.elderly_limit, args.event_limit, args.lab_limit)
        units.append(QueryUnit(
            label=f"{family} (batched: {len(fam_specs)} vars, 1 scan)", sql=sql, specs=fam_specs))
    for spec in outcomes:
        units.append(QueryUnit(label=spec.variable_name,
                               sql=build_los_query(spec, args.elderly_limit), specs=[spec]))
    return units


# --------------------------------------------------------------------------- #
# Dry-run reporting.
# --------------------------------------------------------------------------- #
def dry_run(specs: list[VariableSpec], args: argparse.Namespace) -> None:
    units = plan_queries(specs, args)
    tables: set[str] = {f"{HOSP_DATASET}.patients", f"{ICU_DATASET}.icustays"}
    print("=" * 78)
    print("DRY RUN -- no query is executed, no file is written.")
    print(f"Families: {args.family} | elderly_limit={args.elderly_limit} "
          f"event_limit={args.event_limit} lab_limit={args.lab_limit}")
    print(f"Variables in scope: {len(specs)} -> {', '.join(s.variable_name for s in specs)}")
    print(f"Queries to run: {len(units)} (event families are batched into ONE scan each)")
    print("=" * 78)
    for unit in units:
        tables |= _tables_in_sql(unit.sql)
        print(f"\n----- {unit.label} | vars: {', '.join(s.variable_name for s in unit.specs)} -----")
        print(unit.sql)
    print("\n" + "=" * 78)
    print("Tables that would be scanned:")
    for tbl in sorted(tables):
        print(f"  - {tbl}  [{_table_cost_label(tbl)}]")
    print("\nFor a REAL byte estimate (free BigQuery dry-run), run:")
    print("  python -m src.extract_icu_features --family labs --estimate-cost")
    print("Suggested very limited test run:")
    print("  python -m src.extract_icu_features --family labs --sample 200")
    print("(Note: --sample limits rows RETURNED, not bytes SCANNED; see --estimate-cost.)")
    print("=" * 78)


# --------------------------------------------------------------------------- #
# Cost estimation via a FREE BigQuery dry-run (dry_run=True scans nothing).
# --------------------------------------------------------------------------- #
def estimate_cost(specs: list[VariableSpec], args: argparse.Namespace) -> None:
    from google.cloud import bigquery  # local import

    try:
        client = _build_client()
    except Exception as exc:  # noqa: BLE001 - degrade gracefully without auth/SDK
        LOGGER.error("BigQuery unavailable for cost estimation (%s). Authenticate first.", exc)
        return

    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    units = plan_queries(specs, args)
    print("=" * 78)
    print("COST ESTIMATE -- BigQuery dry-run (free; scans no data, writes nothing).")
    print(f"Families: {args.family} | per-query cap (--max-bytes-billed) = {_fmt_bytes(args.max_bytes_billed)}")
    print(f"Queries: {len(units)} (event families batched into ONE scan each)")
    print("=" * 78)

    tables: set[str] = set()
    per_unit: list[tuple[str, int]] = []
    for unit in units:
        tables |= _tables_in_sql(unit.sql)
        try:
            job = client.query(unit.sql, job_config=job_config)
            processed = int(job.total_bytes_processed or 0)
        except Exception as exc:  # noqa: BLE001 - one estimate failing is non-fatal
            LOGGER.error("Dry-run estimate for %s failed: %s", unit.label, exc)
            continue
        per_unit.append((unit.label, processed))
        flag = "  >>> EXCEEDS --max-bytes-billed" if processed > args.max_bytes_billed else ""
        print(f"  {unit.label:<34} {_fmt_bytes(processed)}{flag}")

    if not per_unit:
        print("\nNo estimates could be produced.")
        return

    max_single = max(b for _, b in per_unit)
    total = sum(b for _, b in per_unit)
    print("-" * 78)
    print(f"  {'MAX single query':<34} {_fmt_bytes(max_single)}  (this is what the cap applies to)")
    print(f"  {'SUM over all queries':<34} {_fmt_bytes(total)}  ({len(per_unit)} queries total)")
    print("\nTables scanned:")
    for tbl in sorted(tables):
        print(f"  - {tbl}  [{_table_cost_label(tbl)}]")
    if max_single > args.max_bytes_billed:
        print(f"\nWARNING: the largest query ({_fmt_bytes(max_single)}) exceeds the cap "
              f"({_fmt_bytes(args.max_bytes_billed)}); a real run would FAIL cleanly.")
        print("Raise it explicitly if intended, e.g. --max-bytes-billed " f"{max_single + 1024**3}")
    else:
        print(f"\nOK: every query is under the cap ({_fmt_bytes(args.max_bytes_billed)}); a real "
              "labs run would proceed.")
    print("=" * 78)


# --------------------------------------------------------------------------- #
# Real extraction.
# --------------------------------------------------------------------------- #
def run_extraction(specs: list[VariableSpec], args: argparse.Namespace) -> None:
    ensure_data_directories()
    client = _build_client()
    job_config = _billed_job_config(args.max_bytes_billed)
    LOGGER.info("Per-query cap (maximum_bytes_billed) = %s", _fmt_bytes(args.max_bytes_billed))
    LOGGER.info("Fetching shared elderly cohort sizes (elderly_limit=%s)...", args.elderly_limit)
    cohort_sizes = fetch_cohort_sizes(client, args.elderly_limit, job_config=job_config)
    LOGGER.info("Cohort patients by age_group: %s", cohort_sizes)

    summary_rows: list[dict[str, Any]] = []
    clean_frames: list[pd.DataFrame] = []
    extracted_vars: set[str] = set()
    exclusions: list[dict[str, Any]] = []

    for unit in plan_queries(specs, args):
        LOGGER.info("Running query: %s [%s]", unit.label,
                    ", ".join(s.variable_name for s in unit.specs))
        try:
            raw = client.query(unit.sql, job_config=job_config).to_dataframe()
        except Exception as exc:  # noqa: BLE001 - one query failing must not abort the rest
            LOGGER.error(
                "Query '%s' failed: %s. If this is a bytes-billed cap, raise --max-bytes-billed.",
                unit.label, exc,
            )
            continue
        # Dispatch the single batched scan back to each variable by itemid.
        for spec in unit.specs:
            if spec.variable_category == "outcome":
                clean, stats = clean_outcome_frame(spec, raw)
            else:
                sub = raw[raw["itemid"].isin(list(spec.itemids))].copy() if "itemid" in raw.columns else raw
                clean, stats = clean_variable_frame(spec, sub)
            exclusions.append({"variable_name": spec.variable_name, **stats})
            extracted_vars.add(spec.variable_name)
            if clean.empty:
                LOGGER.warning("%s: no usable rows after cleaning (%s).", spec.variable_name, stats)
                continue
            summary_rows.extend(summarize_variable(spec, clean, cohort_sizes))
            clean_frames.append(clean)

    if not extracted_vars:
        LOGGER.error("No variables were extracted; nothing written.")
        return

    summary = _merge_summary(summary_rows, extracted_vars)
    summary.to_csv(FEATURE_SUMMARY_CSV, index=False)
    LOGGER.info("Wrote %s summary rows to %s", len(summary), FEATURE_SUMMARY_CSV)

    new_wide = build_patient_features(clean_frames)
    patient_features = _merge_patient_features(new_wide, extracted_vars)
    if not patient_features.empty:
        patient_features.to_csv(PATIENT_FEATURES_CSV, index=False)
        LOGGER.info("Wrote %s patient-level rows to %s (gitignored)",
                    len(patient_features), PATIENT_FEATURES_CSV)

    print("\n=== EXCLUSION REPORT (rows kept at each stage) ===")
    print(pd.DataFrame(exclusions).to_string(index=False))
    print("\nDescriptive, non-clinical summary only. Out-of-(safe-bound) and "
          "out-of-window rows were excluded as shown above.")


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 3 ICU multi-family feature extraction (descriptive).")
    parser.add_argument("--family", default="all", choices=("all",) + FAMILIES,
                        help="Variable family to extract (default: all).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the generated SQL and tables touched; execute nothing, write nothing.")
    parser.add_argument("--estimate-cost", action="store_true",
                        help="Run a FREE BigQuery dry-run per variable and report total_bytes_processed "
                             "(MB/GB) and tables scanned. Executes no real query, writes nothing.")
    parser.add_argument("--max-bytes-billed", type=int, default=MAX_BYTES_BILLED_DEFAULT,
                        help="Per-query safety cap in bytes for REAL runs; a query that would scan more "
                             f"fails cleanly. Default ~{MAX_BYTES_BILLED_DEFAULT // 1024 ** 3} GB.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Shortcut: set event_limit AND lab_limit to this small value for a cheap test "
                             "(limits rows RETURNED, not bytes SCANNED).")
    parser.add_argument("--limit", type=int, default=None, help="Alias for --sample.")
    parser.add_argument("--elderly-limit", type=int, default=ELDERLY_LIMIT_DEFAULT)
    parser.add_argument("--event-limit", type=int, default=EVENT_LIMIT_DEFAULT)
    parser.add_argument("--lab-limit", type=int, default=LAB_LIMIT_DEFAULT)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    sample = args.sample if args.sample is not None else args.limit
    if sample is not None:
        args.event_limit = sample
        args.lab_limit = sample

    families = list(FAMILIES) if args.family == "all" else [args.family]
    specs = _specs_for_families(families)
    if not specs:
        LOGGER.error("No variables selected for families %s.", families)
        return

    if args.dry_run:
        dry_run(specs, args)
        return

    if args.estimate_cost:
        estimate_cost(specs, args)
        return

    LOGGER.warning(
        "Real extraction will scan MIMIC-IV event tables (chartevents/labevents). "
        "Limits: elderly=%s event=%s lab=%s. Use --dry-run first to review the SQL.",
        args.elderly_limit, args.event_limit, args.lab_limit,
    )
    run_extraction(specs, args)


if __name__ == "__main__":
    main()
