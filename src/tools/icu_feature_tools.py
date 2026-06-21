"""Phase 3 deterministic tools over the multi-variable ICU feature summary.

These tools read the aggregated, committable table
``data/processed/icu_feature_summary.csv`` (one row per
variable x age_group x time_window, produced by
``src/extract_icu_features.py``). They turn it into an auditable, NON-CLINICAL
"ICU Multi-Data Agent": listing variables, summarizing one, querying a cohort,
comparing age groups / time windows, building an evidence card, and preparing a
simple bar-chart payload.

Every tool is a pure function returning JSON-serializable dicts (so both the
local and the real MCP backend can carry them). They never diagnose, never
recommend treatment, and never invent numbers -- they only read the summary.
The aggregates are descriptive statistics of a (capped) sample; ``missing_rate``
reflects sample coverage, not true population missingness.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import PROCESSED_DIR, ROOT_DIR

LOGGER = logging.getLogger(__name__)

FEATURE_SUMMARY_CSV = PROCESSED_DIR / "icu_feature_summary.csv"
_SUMMARY_REL = str(FEATURE_SUMMARY_CSV.relative_to(ROOT_DIR)).replace("\\", "/")

# Metric columns a caller may ask for, mapped to the CSV column.
METRIC_COLUMNS = {
    "mean": "mean", "std": "std", "min": "min", "max": "max", "median": "median",
    "p05": "p05", "p25": "p25", "p50": "p50", "p75": "p75", "p90": "p90", "p95": "p95",
    "average": "mean",  # convenience alias
}
DEFAULT_METRIC = "median"

# A high sample missing_rate is flagged so answers never over-claim coverage.
MISSING_RATE_WARN = 0.5

NON_CLINICAL_NOTE = (
    "Descriptive academic statistics only; not a clinical diagnosis or treatment recommendation."
)


# --------------------------------------------------------------------------- #
# Loading / small helpers.
# --------------------------------------------------------------------------- #
def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def load_feature_summary() -> tuple[pd.DataFrame, str]:
    """Return (dataframe, source_label). Empty frame if the CSV is absent."""

    if Path(FEATURE_SUMMARY_CSV).exists():
        frame = pd.read_csv(FEATURE_SUMMARY_CSV)
        frame["itemids"] = frame["itemids"].astype(str)
        return frame, _SUMMARY_REL
    LOGGER.warning("Feature summary CSV missing at %s.", _SUMMARY_REL)
    return pd.DataFrame(), _SUMMARY_REL


def _resolve_metric(metric: str | None) -> str:
    if not metric:
        return DEFAULT_METRIC
    return METRIC_COLUMNS.get(str(metric).lower().strip(), DEFAULT_METRIC)


def _row_metric(row: pd.Series, metric_col: str) -> float | None:
    return _num(row.get(metric_col))


def _known_variables(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["variable_name"].unique().tolist()) if not frame.empty else []


# --------------------------------------------------------------------------- #
# 1. list_available_variables
# --------------------------------------------------------------------------- #
def list_available_variables(variable_category: str | None = None) -> dict[str, Any]:
    """List variables present in the feature summary, optionally by category."""

    frame, source = load_feature_summary()
    if frame.empty:
        return {"error": "Feature summary not available.", "source": source, "variables": [], "count": 0}

    categories = sorted(frame["variable_category"].unique().tolist())
    if variable_category and str(variable_category).lower() not in ("all", ""):
        wanted = str(variable_category).lower()
        frame = frame[frame["variable_category"].str.lower() == wanted]
        if frame.empty:
            return {"error": f"No variables in category '{variable_category}'.",
                    "categories_available": categories, "variables": [], "count": 0, "source": source}

    variables: list[dict[str, Any]] = []
    for name, group in frame.groupby("variable_name"):
        head = group.iloc[0]
        variables.append({
            "variable_name": str(name),
            "variable_category": str(head["variable_category"]),
            "unit": str(head["unit"]),
            "source_table": str(head["source_table"]),
            "itemids": str(head["itemids"]),
            "n_buckets": int(len(group)),
            "age_groups": sorted(group["age_group"].unique().tolist()),
            "time_windows": sorted(group["time_window"].unique().tolist()),
        })
    variables.sort(key=lambda d: (d["variable_category"], d["variable_name"]))
    return {
        "count": len(variables),
        "categories_available": categories,
        "variables": variables,
        "source": source,
        "non_clinical_note": NON_CLINICAL_NOTE,
    }


# --------------------------------------------------------------------------- #
# 2. get_variable_summary
# --------------------------------------------------------------------------- #
def get_variable_summary(variable_name: str, age_group: str, time_window: str) -> dict[str, Any]:
    """Exact statistical summary for one (variable, age_group, time_window)."""

    frame, source = load_feature_summary()
    if frame.empty:
        return {"error": "Feature summary not available.", "source": source}

    mask = (
        (frame["variable_name"].astype(str) == str(variable_name))
        & (frame["age_group"].astype(str) == str(age_group))
        & (frame["time_window"].astype(str) == str(time_window))
    )
    sub = frame[mask]
    if sub.empty:
        return {
            "error": "No summary found for that variable/age_group/time_window.",
            "requested": {"variable_name": variable_name, "age_group": age_group, "time_window": time_window},
            "known_variables": _known_variables(frame),
            "source": source,
        }

    row = sub.iloc[0]
    missing_rate = _num(row.get("missing_rate"))
    result: dict[str, Any] = {
        "variable_name": str(row["variable_name"]),
        "variable_category": str(row["variable_category"]),
        "source_table": str(row["source_table"]),
        "itemids": str(row["itemids"]),
        "unit": str(row["unit"]),
        "age_group": str(row["age_group"]),
        "time_window": str(row["time_window"]),
        "n_patients": int(_num(row.get("n_patients")) or 0),
        "n_measurements": int(_num(row.get("n_measurements")) or 0),
        "missing_rate": missing_rate,
        "missing_rate_warning": bool(missing_rate is not None and missing_rate > MISSING_RATE_WARN),
        "source": source,
        "non_clinical_note": NON_CLINICAL_NOTE,
    }
    for col in ("mean", "std", "min", "max", "median", "p05", "p25", "p50", "p75", "p90", "p95"):
        result[col] = _num(row.get(col))
    return result


# --------------------------------------------------------------------------- #
# 3. query_cohort_statistics
# --------------------------------------------------------------------------- #
def query_cohort_statistics(
    variable_name: str,
    age_group: str | None = None,
    time_window: str | None = None,
    metric: str | None = None,
) -> dict[str, Any]:
    """Descriptive query on a variable; age_group/time_window optional."""

    frame, source = load_feature_summary()
    if frame.empty:
        return {"error": "Feature summary not available.", "source": source}

    sub = frame[frame["variable_name"].astype(str) == str(variable_name)]
    if sub.empty:
        return {"error": f"Unknown variable '{variable_name}'.",
                "known_variables": _known_variables(frame), "source": source}
    if age_group:
        sub = sub[sub["age_group"].astype(str) == str(age_group)]
    if time_window:
        sub = sub[sub["time_window"].astype(str) == str(time_window)]
    if sub.empty:
        return {"error": "No rows match the requested filters.",
                "requested": {"variable_name": variable_name, "age_group": age_group,
                              "time_window": time_window}, "source": source}

    metric_col = _resolve_metric(metric)
    sub = sub.sort_values(["age_group", "time_window"])
    rows = [{
        "age_group": str(r["age_group"]),
        "time_window": str(r["time_window"]),
        "n_patients": int(_num(r.get("n_patients")) or 0),
        "n_measurements": int(_num(r.get("n_measurements")) or 0),
        metric_col: _row_metric(r, metric_col),
        "missing_rate": _num(r.get("missing_rate")),
    } for _, r in sub.iterrows()]

    unit = str(sub.iloc[0]["unit"])
    multiple = len(rows) > 1
    parts = [f"{r['age_group']}/{r['time_window']}: {metric_col}={r[metric_col]}{unit}" for r in rows]
    readable = f"{variable_name} ({metric_col}) -> " + "; ".join(parts)
    return {
        "variable_name": str(variable_name),
        "variable_category": str(sub.iloc[0]["variable_category"]),
        "unit": unit,
        "metric": metric_col,
        "n_rows": len(rows),
        "multiple_groups_or_windows": multiple,
        "rows": rows,
        "readable_summary": readable,
        "source": source,
        "non_clinical_note": NON_CLINICAL_NOTE,
    }


# --------------------------------------------------------------------------- #
# 4. compare_age_groups
# --------------------------------------------------------------------------- #
def compare_age_groups(variable_name: str, time_window: str, metric: str | None = None) -> dict[str, Any]:
    """Compare one variable across age groups for a fixed time window."""

    frame, source = load_feature_summary()
    if frame.empty:
        return {"error": "Feature summary not available.", "source": source}
    metric_col = _resolve_metric(metric)

    sub = frame[
        (frame["variable_name"].astype(str) == str(variable_name))
        & (frame["time_window"].astype(str) == str(time_window))
    ]
    if sub.empty:
        return {"error": f"No data for {variable_name} in {time_window}.",
                "known_variables": _known_variables(frame), "source": source}

    values: dict[str, float | None] = {}
    for _, r in sub.iterrows():
        values[str(r["age_group"])] = _row_metric(r, metric_col)
    present = {k: v for k, v in values.items() if v is not None}

    unit = str(sub.iloc[0]["unit"])
    result: dict[str, Any] = {
        "variable_name": str(variable_name),
        "variable_category": str(sub.iloc[0]["variable_category"]),
        "time_window": str(time_window),
        "metric": metric_col,
        "unit": unit,
        "values_by_age_group": values,
        "source": source,
        "non_clinical_note": NON_CLINICAL_NOTE,
    }
    if len(present) >= 2:
        highest = max(present, key=present.get)
        lowest = min(present, key=present.get)
        spread = round(present[highest] - present[lowest], 4)
        result.update({
            "highest_age_group": highest,
            "lowest_age_group": lowest,
            "spread": spread,
            "descriptive": (
                f"For {variable_name} ({metric_col}) in {time_window}, the highest value is in "
                f"age group {highest} ({present[highest]}{unit}) and the lowest in {lowest} "
                f"({present[lowest]}{unit}); spread {spread}{unit}. Descriptive only."
            ),
        })
    else:
        result["descriptive"] = (
            f"Only {len(present)} age group(s) have data for {variable_name} in {time_window}; "
            "no comparison can be made."
        )
    return result


# --------------------------------------------------------------------------- #
# 5. compare_time_windows
# --------------------------------------------------------------------------- #
_WINDOW_ORDER = ["first_6h", "first_12h", "first_24h"]


def compare_time_windows(variable_name: str, age_group: str, metric: str | None = None) -> dict[str, Any]:
    """Compare one variable across time windows for a fixed age group."""

    frame, source = load_feature_summary()
    if frame.empty:
        return {"error": "Feature summary not available.", "source": source}
    metric_col = _resolve_metric(metric)

    sub = frame[
        (frame["variable_name"].astype(str) == str(variable_name))
        & (frame["age_group"].astype(str) == str(age_group))
    ]
    if sub.empty:
        return {"error": f"No data for {variable_name} in age group {age_group}.",
                "known_variables": _known_variables(frame), "source": source}

    by_window = {str(r["time_window"]): _row_metric(r, metric_col) for _, r in sub.iterrows()}
    ordered = {w: by_window.get(w) for w in _WINDOW_ORDER if w in by_window}
    series = [v for v in ordered.values() if v is not None]
    unit = str(sub.iloc[0]["unit"])

    if len(series) >= 2:
        if all(series[i] < series[i + 1] for i in range(len(series) - 1)):
            trend = "increasing"
        elif all(series[i] > series[i + 1] for i in range(len(series) - 1)):
            trend = "decreasing"
        elif abs(series[-1] - series[0]) < 1e-9:
            trend = "stable"
        else:
            trend = "non-monotonic"
        descriptive = (
            f"For {variable_name} ({metric_col}) in age group {age_group}, the value is {trend} "
            f"across {', '.join(ordered)}: {series}. Descriptive only, not a temporal causal claim."
        )
    else:
        trend = "insufficient_data"
        descriptive = f"Insufficient time windows with data for {variable_name} in {age_group}."

    return {
        "variable_name": str(variable_name),
        "variable_category": str(sub.iloc[0]["variable_category"]),
        "age_group": str(age_group),
        "metric": metric_col,
        "unit": unit,
        "values_by_time_window": ordered,
        "trend": trend,
        "descriptive": descriptive,
        "source": source,
        "non_clinical_note": NON_CLINICAL_NOTE,
    }


# --------------------------------------------------------------------------- #
# 6. generate_evidence_card
# --------------------------------------------------------------------------- #
def generate_evidence_card(
    variable_name: str,
    age_group: str | None = None,
    time_window: str | None = None,
    tool_name: str | None = None,
    main_metric: str | None = None,
) -> dict[str, Any]:
    """Assemble a structured, sourced evidence card for a Phase 3 answer."""

    frame, source = load_feature_summary()
    if frame.empty:
        return {"error": "Feature summary not available.", "source": source}

    sub = frame[frame["variable_name"].astype(str) == str(variable_name)]
    if sub.empty:
        return {"error": f"Unknown variable '{variable_name}'.",
                "known_variables": _known_variables(frame), "source": source}
    if age_group:
        sub = sub[sub["age_group"].astype(str) == str(age_group)]
    if time_window:
        sub = sub[sub["time_window"].astype(str) == str(time_window)]
    if sub.empty:
        sub = frame[frame["variable_name"].astype(str) == str(variable_name)]  # fall back to all buckets

    head = sub.iloc[0]
    metric_col = _resolve_metric(main_metric)
    n_patients = int(pd.to_numeric(sub["n_patients"], errors="coerce").fillna(0).sum())
    n_measurements = int(pd.to_numeric(sub["n_measurements"], errors="coerce").fillna(0).sum())
    missing_rates = pd.to_numeric(sub["missing_rate"], errors="coerce").dropna()
    max_missing = float(missing_rates.max()) if not missing_rates.empty else None
    metric_values = pd.to_numeric(sub[metric_col], errors="coerce").dropna().tolist()
    main_metric_repr = (
        round(float(metric_values[0]), 4) if len(metric_values) == 1
        else [round(float(v), 4) for v in metric_values]
    )

    warn = (
        "High sample missing_rate; figures reflect a capped sample, not full population coverage."
        if (max_missing is not None and max_missing > MISSING_RATE_WARN) else "None."
    )

    card = {
        "variable": str(variable_name),
        "category": str(head["variable_category"]),
        "source_table": str(head["source_table"]),
        "itemids": str(head["itemids"]),
        "unit": str(head["unit"]),
        "age_group": age_group or sorted(sub["age_group"].unique().tolist()),
        "time_window": time_window or sorted(sub["time_window"].unique().tolist()),
        "n_patients": n_patients,
        "n_measurements": n_measurements,
        "main_metric": {metric_col: main_metric_repr},
        "missing_rate_warning": warn,
        "tool_used": tool_name,
        "non_clinical_note": NON_CLINICAL_NOTE,
        "evidence_source": source,
    }
    text = (
        "Evidence card:\n"
        f"- Variable: {card['variable']}\n"
        f"- Category: {card['category']}\n"
        f"- Source table: {card['source_table']}\n"
        f"- Itemids: {card['itemids']}\n"
        f"- Unit: {card['unit']}\n"
        f"- Age group: {card['age_group']}\n"
        f"- Time window: {card['time_window']}\n"
        f"- N patients: {card['n_patients']}\n"
        f"- N measurements: {card['n_measurements']}\n"
        f"- Main metric ({metric_col}): {main_metric_repr}\n"
        f"- Missing rate warning: {warn}\n"
        f"- Non-clinical note: {NON_CLINICAL_NOTE}"
    )
    card["text"] = text
    return card


# --------------------------------------------------------------------------- #
# 7. plot_variable_distribution (bar chart of an aggregate; no fake histogram)
# --------------------------------------------------------------------------- #
def plot_variable_distribution(
    variable_name: str,
    metric: str | None = None,
    by: str = "age_group",
    time_window: str | None = None,
    age_group: str | None = None,
    save_png: bool = False,
) -> dict[str, Any]:
    """Bar-chart payload of an aggregate metric, grouped by age_group OR time_window.

    The summary holds aggregates, not raw values, so this returns bar-chart data
    of median/p90/etc. -- it does NOT fabricate a per-value histogram.
    """

    frame, source = load_feature_summary()
    if frame.empty:
        return {"error": "Feature summary not available.", "source": source}
    metric_col = _resolve_metric(metric)
    group_col = "age_group" if by != "time_window" else "time_window"

    sub = frame[frame["variable_name"].astype(str) == str(variable_name)]
    if sub.empty:
        return {"error": f"Unknown variable '{variable_name}'.",
                "known_variables": _known_variables(frame), "source": source}
    if group_col == "age_group" and time_window:
        sub = sub[sub["time_window"].astype(str) == str(time_window)]
    if group_col == "time_window" and age_group:
        sub = sub[sub["age_group"].astype(str) == str(age_group)]
    if sub.empty:
        return {"error": "No rows match the requested filters.", "source": source}

    order = ["65-74", "75-84", "85+"] if group_col == "age_group" else _WINDOW_ORDER
    agg = sub.groupby(group_col)[metric_col].mean()
    x = [g for g in order if g in agg.index]
    y = [round(float(agg[g]), 4) for g in x]

    payload = {
        "chart_type": "bar",
        "variable_name": str(variable_name),
        "metric": metric_col,
        "group_by": group_col,
        "x": x,
        "y": y,
        "unit": str(sub.iloc[0]["unit"]),
        "title": f"{variable_name} {metric_col} by {group_col}",
        "source": source,
        "non_clinical_note": NON_CLINICAL_NOTE,
    }

    if save_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            out_dir = ROOT_DIR / "data" / "phase3_outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(x, y, color="#4C78A8")
            ax.set_title(payload["title"])
            ax.set_ylabel(f"{metric_col} ({payload['unit']})")
            fig.tight_layout()
            png_path = out_dir / f"{variable_name}_{metric_col}_by_{group_col}.png"
            fig.savefig(png_path, dpi=110)
            plt.close(fig)
            payload["png_path"] = str(png_path.relative_to(ROOT_DIR))
        except Exception as exc:  # noqa: BLE001 - plotting is best-effort
            payload["png_error"] = f"Could not render PNG ({exc}); use x/y for Streamlit charting."
    return payload


# --------------------------------------------------------------------------- #
# Safety: detect clinical-advice requests (deterministic gate, no LLM).
# --------------------------------------------------------------------------- #
_CLINICAL_ADVICE_PATTERNS = (
    "what treatment", "which treatment", "how to treat", "treat this", "treat the patient",
    "should we give", "should i give", "should we administer", "should we start",
    "should this patient receive", "should the patient receive", "what should we do",
    "diagnose", "diagnosis", "what is wrong with", "prescribe", "prescription",
    "what medication", "what drug", "what dose", "dosage", "how much should",
    "is this patient septic", "does this patient have", "recommend treatment",
    "manage this patient", "intervention should", "what therapy",
)


def detect_clinical_advice_request(question: str) -> dict[str, Any]:
    """Flag treatment/diagnosis requests so the agent can refuse them, non-clinically."""

    lowered = str(question).lower()
    matched = [p for p in _CLINICAL_ADVICE_PATTERNS if p in lowered]
    is_clinical = bool(matched)
    return {
        "is_clinical_advice": is_clinical,
        "matched_patterns": matched,
        "reason": (
            "The question asks for diagnosis or treatment guidance, which this descriptive, "
            "non-clinical academic tool does not provide."
            if is_clinical else "No clinical-advice intent detected."
        ),
        "refusal_message": (
            "This is a descriptive, non-clinical academic tool. It cannot provide a diagnosis or "
            "recommend treatment. It can only summarize MIMIC-IV ICU variable distributions "
            "(e.g. percentiles by age group and time window)."
        ),
    }
