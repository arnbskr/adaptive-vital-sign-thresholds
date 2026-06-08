from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import db_dtypes  # noqa: F401
import pandas as pd
from google.cloud import bigquery

from .config import ALLOW_DEMO_FALLBACK, HOSP_DATASET, ICU_DATASET, PROJECT_ID, PROCESSED_DIR, ensure_data_directories

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ELDERLY_LIMIT = 1000
EVENT_LIMIT = 10000


# Canonical age-group / time-window bucketing. Inlined here (previously imported
# from the now-archived legacy ``rag_utils``) so this upstream extractor is
# self-contained. Must stay consistent with the vocab in src/semantic_rag.py.
def age_group_from_age(age: float | int | None) -> str:
    if age is None:
        return "Unknown"
    if age >= 85:
        return "85+"
    if age >= 75:
        return "75-84"
    return "65-74"


def time_window_from_hours(hours_since_admission: float) -> str | None:
    if hours_since_admission <= 6:
        return "first_6h"
    if hours_since_admission <= 12:
        return "first_12h"
    if hours_since_admission <= 24:
        return "first_24h"
    return None

VITAL_GROUP_SPECS: list[dict[str, Any]] = [
    {"vital_sign": "Heart Rate", "itemids": [220045]},
    {"vital_sign": "Respiratory Rate", "itemids": [220210]},
    {"vital_sign": "MAP", "itemids": [220052, 220181]},
    {"vital_sign": "Systolic Blood Pressure", "itemids": [220050, 220179]},
    {"vital_sign": "Diastolic Blood Pressure", "itemids": [220051, 220180]},
    {"vital_sign": "Temperature", "itemids": [223762, 223761]},
    {"vital_sign": "SpO2", "itemids": [220277]},
]

ITEM_SPECS: dict[int, dict[str, Any]] = {
    220045: {"vital_sign": "Heart Rate", "standard_low": 60.0, "standard_high": 100.0, "safe_low": 20.0, "safe_high": 250.0},
    220210: {"vital_sign": "Respiratory Rate", "standard_low": 12.0, "standard_high": 20.0, "safe_low": 1.0, "safe_high": 80.0},
    220052: {"vital_sign": "MAP", "standard_low": 65.0, "standard_high": None, "safe_low": 20.0, "safe_high": 200.0},
    220181: {"vital_sign": "MAP", "standard_low": 65.0, "standard_high": None, "safe_low": 20.0, "safe_high": 200.0},
    220050: {"vital_sign": "Systolic Blood Pressure", "standard_low": 90.0, "standard_high": 140.0, "safe_low": 40.0, "safe_high": 300.0},
    220179: {"vital_sign": "Systolic Blood Pressure", "standard_low": 90.0, "standard_high": 140.0, "safe_low": 40.0, "safe_high": 300.0},
    220051: {"vital_sign": "Diastolic Blood Pressure", "standard_low": 60.0, "standard_high": 90.0, "safe_low": 20.0, "safe_high": 200.0},
    220180: {"vital_sign": "Diastolic Blood Pressure", "standard_low": 60.0, "standard_high": 90.0, "safe_low": 20.0, "safe_high": 200.0},
    223762: {"vital_sign": "Temperature", "standard_low": 36.0, "standard_high": 38.0, "safe_low": 25.0, "safe_high": 45.0},
    223761: {"vital_sign": "Temperature", "standard_low": 96.8, "standard_high": 100.4, "safe_low": 77.0, "safe_high": 113.0},
    220277: {"vital_sign": "SpO2", "standard_low": 92.0, "standard_high": None, "safe_low": 0.0, "safe_high": 100.0, "strict_safe_low": True},
}

ALL_VITAL_ITEMIDS = sorted(ITEM_SPECS)


def build_client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT_ID)


def check_bigquery_connection(client: bigquery.Client) -> None:
    client.query("SELECT 1 AS ok LIMIT 1").to_dataframe()


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    LOGGER.info("Saved %s rows to %s", len(df), path)


def _demo_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "vital_sign": "Heart Rate",
                "itemid": 220045,
                "label": "Heart Rate",
                "unitname": "bpm",
                "age_group": "75-84",
                "time_window": "first_24h",
                "count": 12,
                "mean": 104.0,
                "median": 101.0,
                "min": 72.0,
                "max": 138.0,
                "std": 18.4,
                "p5": 74.0,
                "p25": 88.0,
                "p50": 101.0,
                "p75": 114.0,
                "p90": 122.0,
                "percent_above_standard_high": 58.3,
                "percent_below_standard_low": 0.0,
                "standard_low": 60.0,
                "standard_high": 100.0,
                "is_demo_data": True,
            }
        ]
    )


def _demo_sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": 1,
                "hadm_id": 1,
                "stay_id": 1,
                "anchor_age": 82,
                "age_group": "75-84",
                "intime": "2026-01-01 00:00:00",
                "charttime": "2026-01-01 02:00:00",
                "hours_since_icu_admission": 2.0,
                "time_window": "first_6h",
                "vital_sign": "Heart Rate",
                "itemid": 220045,
                "label": "Heart Rate",
                "unitname": "bpm",
                "value": 104.0,
                "is_demo_data": True,
            }
        ]
    )


def extract_elderly_icu_stays(client: bigquery.Client) -> pd.DataFrame:
    query = f"""
    SELECT
      p.subject_id,
      p.gender,
      p.anchor_age,
      i.hadm_id,
      i.stay_id,
      i.intime,
      i.outtime,
      i.los
    FROM `{HOSP_DATASET}.patients` p
    JOIN `{ICU_DATASET}.icustays` i
      ON p.subject_id = i.subject_id
    WHERE p.anchor_age >= 65
    LIMIT {ELDERLY_LIMIT}
    """
    return client.query(query).to_dataframe()


def extract_icu_vital_items(client: bigquery.Client) -> pd.DataFrame:
    itemid_list = ", ".join(str(int(itemid)) for itemid in ALL_VITAL_ITEMIDS)
    query = f"""
    SELECT
      itemid,
      label,
      abbreviation,
      category,
      unitname
    FROM `{ICU_DATASET}.d_items`
    WHERE itemid IN ({itemid_list})
    ORDER BY itemid
    """
    return client.query(query).to_dataframe()


def _item_spec(itemid: int | float | None) -> dict[str, Any] | None:
    if itemid is None or pd.isna(itemid):
        return None
    return ITEM_SPECS.get(int(itemid))


def _safe_filter_clause(itemid: int) -> str:
    spec = _item_spec(itemid)
    if spec is None:
        raise KeyError(f"No safe filter specification found for itemid {itemid}")
    lower_op = ">" if spec.get("strict_safe_low") else ">="
    lower_bound = float(spec["safe_low"])
    upper_bound = float(spec["safe_high"])
    return f"(c.itemid = {int(itemid)} AND c.valuenum {lower_op} {lower_bound} AND c.valuenum <= {upper_bound})"


def _query_vital_rows(client: bigquery.Client, itemids: list[int], limit: int = EVENT_LIMIT) -> pd.DataFrame:
    itemid_list = ", ".join(str(int(itemid)) for itemid in sorted(set(itemids)))
    safe_clause = " OR ".join(_safe_filter_clause(itemid) for itemid in sorted(set(itemids)))
    query = f"""
    WITH elderly_icu AS (
      SELECT
        p.subject_id,
        p.anchor_age,
        i.hadm_id,
        i.stay_id,
        i.intime
      FROM `{HOSP_DATASET}.patients` p
      JOIN `{ICU_DATASET}.icustays` i
        ON p.subject_id = i.subject_id
      WHERE p.anchor_age >= 65
      LIMIT {ELDERLY_LIMIT}
    )
    SELECT
      e.subject_id,
      e.hadm_id,
      e.stay_id,
      e.anchor_age,
      e.intime,
      c.charttime,
      c.itemid,
      c.valuenum AS value,
      d.label,
      d.unitname
    FROM elderly_icu e
    JOIN `{ICU_DATASET}.chartevents` c
      ON e.stay_id = c.stay_id
    LEFT JOIN `{ICU_DATASET}.d_items` d
      ON c.itemid = d.itemid
    WHERE c.itemid IN ({itemid_list})
      AND c.valuenum IS NOT NULL
      AND c.charttime >= e.intime
      AND c.charttime < TIMESTAMP_ADD(e.intime, INTERVAL 24 HOUR)
            AND ({safe_clause})
    LIMIT {limit}
    """
    rows = client.query(query).to_dataframe()
    if rows.empty:
        return rows
    rows = rows[pd.notna(rows["itemid"])].copy()
    rows["itemid"] = rows["itemid"].astype(int)
    return rows


def extract_vital_sign_sample(client: bigquery.Client) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_frames: list[pd.DataFrame] = []

    for itemid in ALL_VITAL_ITEMIDS:
        spec = _item_spec(itemid)
        if spec is None:
            continue

        vital_sign = str(spec["vital_sign"])
        rows = _query_vital_rows(client, [itemid], limit=EVENT_LIMIT)
        if rows.empty:
            LOGGER.warning("No event rows found for %s; skipping.", vital_sign)
            continue

        rows["label"] = rows["label"].fillna(vital_sign)
        rows["unitname"] = rows["unitname"].fillna("")
        rows["vital_sign"] = vital_sign
        rows["standard_low"] = float(spec["standard_low"]) if spec.get("standard_low") is not None else None
        rows["standard_high"] = float(spec["standard_high"]) if spec.get("standard_high") is not None else None

        rows["age_group"] = rows["anchor_age"].apply(age_group_from_age)
        rows["hours_since_icu_admission"] = (
            pd.to_datetime(rows["charttime"], errors="coerce") - pd.to_datetime(rows["intime"], errors="coerce")
        ).dt.total_seconds() / 3600.0
        rows["time_window"] = rows["hours_since_icu_admission"].apply(
            lambda value: time_window_from_hours(value) if pd.notna(value) else None
        )
        rows = rows[
            [
                "subject_id",
                "hadm_id",
                "stay_id",
                "anchor_age",
                "age_group",
                "intime",
                "charttime",
                "hours_since_icu_admission",
                "time_window",
                "vital_sign",
                "itemid",
                "label",
                "unitname",
                "value",
                "standard_low",
                "standard_high",
            ]
        ].copy()
        sample_frames.append(rows)

    if sample_frames:
        sample_df = pd.concat(sample_frames, ignore_index=True)
    else:
        sample_df = pd.DataFrame(
            columns=[
                "subject_id",
                "hadm_id",
                "stay_id",
                "anchor_age",
                "age_group",
                "intime",
                "charttime",
                "hours_since_icu_admission",
                "time_window",
                "vital_sign",
                "itemid",
                "label",
                "unitname",
                "value",
                "standard_low",
                "standard_high",
            ]
        )

    return sample_df, extract_icu_vital_items(client)


def build_vital_signs_summary(sample_df: pd.DataFrame) -> pd.DataFrame:
    if sample_df.empty:
        return pd.DataFrame(
            columns=[
                "vital_sign",
                "itemid",
                "label",
                "unitname",
                "age_group",
                "time_window",
                "count",
                "mean",
                "median",
                "min",
                "max",
                "std",
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
        )

    summary_rows: list[dict[str, Any]] = []
    group_cols = ["vital_sign", "itemid", "label", "unitname", "age_group", "time_window"]
    for keys, group in sample_df.groupby(group_cols, dropna=False):
        values = pd.to_numeric(group["value"], errors="coerce").dropna()
        if values.empty:
            continue
        vital_sign, itemid, label, unitname, age_group, time_window = keys
        standard_low = group["standard_low"].dropna().iloc[0] if group["standard_low"].notna().any() else None
        standard_high = group["standard_high"].dropna().iloc[0] if group["standard_high"].notna().any() else None
        percent_above = None
        percent_below = None
        if standard_high is not None:
            percent_above = float((values > float(standard_high)).mean() * 100.0)
        if standard_low is not None:
            percent_below = float((values < float(standard_low)).mean() * 100.0)

        summary_rows.append(
            {
                "vital_sign": vital_sign,
                "itemid": int(itemid) if pd.notna(itemid) else None,
                "label": label,
                "unitname": unitname,
                "age_group": age_group,
                "time_window": time_window,
                "count": int(values.count()),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "min": float(values.min()),
                "max": float(values.max()),
                "std": float(values.std(ddof=1)) if values.count() > 1 else 0.0,
                "p5": float(values.quantile(0.05)),
                "p25": float(values.quantile(0.25)),
                "p50": float(values.quantile(0.50)),
                "p75": float(values.quantile(0.75)),
                "p90": float(values.quantile(0.90)),
                "standard_low": float(standard_low) if standard_low is not None else None,
                "standard_high": float(standard_high) if standard_high is not None else None,
                "percent_below_standard_low": percent_below,
                "percent_above_standard_high": percent_above,
                "is_demo_data": False,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["vital_sign", "age_group", "time_window", "itemid"]).reset_index(drop=True)
    return summary_df


def _write_legacy_heart_rate_outputs(sample_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    heart_rate_sample = sample_df[sample_df["vital_sign"] == "Heart Rate"].copy() if not sample_df.empty else pd.DataFrame()
    heart_rate_summary = summary_df[summary_df["vital_sign"] == "Heart Rate"].copy() if not summary_df.empty else pd.DataFrame()

    legacy_sample = heart_rate_sample[[
        "subject_id",
        "hadm_id",
        "stay_id",
        "anchor_age",
        "age_group",
        "intime",
        "charttime",
        "hours_since_icu_admission",
        "time_window",
        "vital_sign",
        "itemid",
        "label",
        "unitname",
        "value",
        "is_demo_data",
    ]].copy() if not heart_rate_sample.empty else pd.DataFrame()
    legacy_summary = heart_rate_summary[[
        "vital_sign",
        "itemid",
        "label",
        "unitname",
        "age_group",
        "time_window",
        "count",
        "mean",
        "median",
        "min",
        "max",
        "std",
        "p5",
        "p25",
        "p50",
        "p75",
        "p90",
        "percent_above_standard_high",
        "percent_below_standard_low",
        "standard_low",
        "standard_high",
        "is_demo_data",
    ]].copy() if not heart_rate_summary.empty else pd.DataFrame()

    if not legacy_sample.empty:
        _save_dataframe(legacy_sample, PROCESSED_DIR / "heart_rate_elderly_icu_sample.csv")
    if not legacy_summary.empty:
        _save_dataframe(legacy_summary, PROCESSED_DIR / "heart_rate_elderly_icu_summary.csv")


def run_pipeline() -> None:
    ensure_data_directories()
    client = build_client()

    try:
        LOGGER.info("Checking BigQuery connectivity for project %s", PROJECT_ID)
        check_bigquery_connection(client)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "BigQuery connection failed. Check credentials, ADC login, and dataset access. Error: %s",
            exc,
        )
        if ALLOW_DEMO_FALLBACK:
            LOGGER.warning("ALLOW_DEMO_FALLBACK=True, writing demo outputs only.")
            demo_sample = _demo_sample_frame()
            demo_summary = _demo_summary_frame()
            _save_dataframe(demo_sample, PROCESSED_DIR / "vital_signs_elderly_icu_sample_demo.csv")
            _save_dataframe(demo_summary, PROCESSED_DIR / "vital_signs_elderly_icu_summary_demo.csv")
            _write_legacy_heart_rate_outputs(demo_sample, demo_summary)
        raise SystemExit(1) from exc

    elderly_icu = extract_elderly_icu_stays(client)
    sample_df, vital_items = extract_vital_sign_sample(client)

    if sample_df.empty:
        raise SystemExit("No vital sign rows were extracted from BigQuery.")

    summary_df = build_vital_signs_summary(sample_df)

    elderly_icu = elderly_icu.assign(is_demo_data=False)
    vital_items = vital_items.assign(is_demo_data=False)
    sample_df = sample_df.assign(is_demo_data=False)
    summary_df = summary_df.assign(is_demo_data=False)

    _save_dataframe(elderly_icu, PROCESSED_DIR / "elderly_icu_stays.csv")
    _save_dataframe(vital_items, PROCESSED_DIR / "icu_vital_items.csv")
    _save_dataframe(sample_df[[
        "subject_id",
        "hadm_id",
        "stay_id",
        "anchor_age",
        "age_group",
        "intime",
        "charttime",
        "hours_since_icu_admission",
        "time_window",
        "vital_sign",
        "itemid",
        "label",
        "unitname",
        "value",
        "is_demo_data",
    ]], PROCESSED_DIR / "vital_signs_elderly_icu_sample.csv")
    _save_dataframe(summary_df[[
        "vital_sign",
        "itemid",
        "label",
        "unitname",
        "age_group",
        "time_window",
        "count",
        "mean",
        "median",
        "min",
        "max",
        "std",
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
    ]], PROCESSED_DIR / "vital_signs_elderly_icu_summary.csv")
    _write_legacy_heart_rate_outputs(sample_df, summary_df)


def main() -> None:
    try:
        run_pipeline()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Phase 1 BigQuery extraction stopped: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
