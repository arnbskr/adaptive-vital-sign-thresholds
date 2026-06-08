"""Tools: compare_to_standard_threshold and explain_threshold_type.

Both are deterministic and descriptive. The comparison never uses alarming
language ("critical", "dangerous", "emergency"); it only reports position
relative to the predefined reference thresholds.
"""

from __future__ import annotations

from typing import Any

# Canonical reference thresholds and default display units per vital sign,
# mirroring the ITEM_SPECS used by src/bigquery_extract_mimic.py. The agent can
# use these as a fallback when a retrieved summary does not carry thresholds.
STANDARD_THRESHOLDS: dict[str, dict[str, Any]] = {
    "Heart Rate": {"standard_low": 60.0, "standard_high": 100.0, "unitname": "bpm"},
    "Respiratory Rate": {"standard_low": 12.0, "standard_high": 20.0, "unitname": "breaths/min"},
    "MAP": {"standard_low": 65.0, "standard_high": None, "unitname": "mmHg"},
    "Systolic Blood Pressure": {"standard_low": 90.0, "standard_high": 140.0, "unitname": "mmHg"},
    "Diastolic Blood Pressure": {"standard_low": 60.0, "standard_high": 90.0, "unitname": "mmHg"},
    "Temperature": {"standard_low": 36.0, "standard_high": 38.0, "unitname": "°C"},
    "SpO2": {"standard_low": 90.0, "standard_high": None, "unitname": "%"},
}


def _format_value(value: float, unit: str) -> str:
    text = f"{value:g}"
    return f"{text} {unit}".strip()


def compare_to_standard_threshold(
    vital_sign: str,
    value: float,
    standard_low: float | None,
    standard_high: float | None,
    unitname: str = "",
) -> dict[str, Any]:
    """Compare a value to predefined standard reference thresholds.

    Status is one of: below_standard_low, above_standard_high, borderline,
    within_reference_range, not_applicable. The output is descriptive only and
    ``is_diagnostic`` is always False.
    """

    unit = unitname or STANDARD_THRESHOLDS.get(vital_sign, {}).get("unitname", "")
    value_text = _format_value(value, unit)

    if standard_low is None and standard_high is None:
        return {
            "status": "not_applicable",
            "explanation": (
                f"No predefined standard threshold is available for {vital_sign}, "
                "so a standard-threshold comparison cannot be made."
            ),
            "is_diagnostic": False,
            "standard_low": standard_low,
            "standard_high": standard_high,
        }

    if (standard_low is not None and value == standard_low) or (
        standard_high is not None and value == standard_high
    ):
        status = "borderline"
        explanation = f"{value_text} sits exactly on a predefined reference threshold ({value_text})."
    elif standard_low is not None and value < standard_low:
        status = "below_standard_low"
        explanation = (
            f"{value_text} is below the predefined low reference threshold of "
            f"{_format_value(standard_low, unit)}."
        )
    elif standard_high is not None and value > standard_high:
        status = "above_standard_high"
        explanation = (
            f"{value_text} is above the predefined high reference threshold of "
            f"{_format_value(standard_high, unit)}."
        )
    else:
        status = "within_reference_range"
        bounds = []
        if standard_low is not None:
            bounds.append(f"low {_format_value(standard_low, unit)}")
        if standard_high is not None:
            bounds.append(f"high {_format_value(standard_high, unit)}")
        explanation = f"{value_text} is within the predefined reference range ({', '.join(bounds)})."

    return {
        "status": status,
        "explanation": explanation,
        "is_diagnostic": False,
        "standard_low": standard_low,
        "standard_high": standard_high,
    }


def explain_threshold_type() -> dict[str, Any]:
    """Explain the difference between standard and adaptive percentile thresholds."""

    return {
        "standard_threshold": (
            "A standard (reference) threshold is a single fixed cut-off applied uniformly to "
            "all patients, for example a heart rate of 60-100 bpm. It is simple and widely "
            "recognised but ignores age, time in the ICU, and population context."
        ),
        "adaptive_percentile_threshold": (
            "An adaptive, percentile-based threshold is derived from the observed distribution "
            "of a vital sign within a specific subgroup (here: elderly ICU patients by age group "
            "and early time window). A value is interpreted relative to percentiles such as P5, "
            "P25, P50, P75, and P90 of that subgroup rather than a single fixed number."
        ),
        "project_interpretation": (
            "This project compares both views: it reports where a value falls against the standard "
            "reference threshold AND where it falls within the matching MIMIC-IV percentile "
            "distribution for the same vital sign, age group, and ICU time window."
        ),
        "limitations": (
            "Percentile-based positions are descriptive summaries of historical MIMIC-IV data and "
            "are not clinical decision rules. They depend on the quality and completeness of the "
            "underlying data and must not be used for diagnosis or treatment."
        ),
    }
