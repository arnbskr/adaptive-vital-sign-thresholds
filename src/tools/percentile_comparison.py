"""Tool: compare_to_percentiles.

Compares a value to the MIMIC-IV percentile distribution carried by an EXACT
summary dict. It must only ever be called with the summary for the SAME vital
sign as the value -- never with another vital sign's percentiles.
"""

from __future__ import annotations

from typing import Any


def _percentiles(summary: dict[str, Any]) -> dict[str, float] | None:
    keys = ["p5", "p25", "p50", "p75", "p90"]
    values: dict[str, float] = {}
    for key in keys:
        raw = summary.get(key)
        if raw is None:
            return None
        try:
            values[key] = float(raw)
        except (TypeError, ValueError):
            return None
    return values


def _unit(summary: dict[str, Any]) -> str:
    return str(summary.get("unitname", "") or "")


def _fmt(value: float, unit: str) -> str:
    return f"{value:g} {unit}".strip()


def compare_to_percentiles(
    value: float,
    summary: dict[str, Any],
    direction: str | None = None,
) -> dict[str, Any]:
    """Locate ``value`` within the summary's percentile distribution.

    ``direction`` is "high", "low", or None/neutral. Returns the percentile
    position, a descriptive explanation, and the percentiles actually used.
    """

    percentiles = _percentiles(summary)
    if percentiles is None:
        return {
            "percentile_position": "unavailable",
            "explanation": "The exact summary does not contain a complete percentile distribution.",
            "used_percentiles": {},
        }

    unit = _unit(summary)
    value_text = _fmt(value, unit)
    p5, p25, p50, p75, p90 = (
        percentiles["p5"],
        percentiles["p25"],
        percentiles["p50"],
        percentiles["p75"],
        percentiles["p90"],
    )
    normalized_direction = (direction or "neutral").strip().lower()

    if normalized_direction == "high":
        if value > p90:
            position = "above_p90"
            explanation = f"{value_text} is above the P90 ({_fmt(p90, unit)}) of the retrieved MIMIC-IV distribution."
        elif value > p75:
            position = "above_p75"
            explanation = f"{value_text} is above the P75 ({_fmt(p75, unit)}) but not above the P90 of the distribution."
        else:
            position = "not_high_relative_to_distribution"
            explanation = f"{value_text} is at or below the P75 ({_fmt(p75, unit)}); it is not high relative to the distribution."
    elif normalized_direction == "low":
        if value <= p5:
            position = "below_p5"
            explanation = f"{value_text} is at or below the P5 ({_fmt(p5, unit)}) of the retrieved MIMIC-IV distribution."
        elif value <= p25:
            position = "below_p25"
            explanation = f"{value_text} is at or below the P25 ({_fmt(p25, unit)}) but above the P5 of the distribution."
        elif value <= p50:
            position = "below_median"
            explanation = f"{value_text} is at or below the median ({_fmt(p50, unit)}) but above the P25 of the distribution."
        else:
            position = "not_low_relative_to_distribution"
            explanation = f"{value_text} is above the median ({_fmt(p50, unit)}); it is not low relative to the distribution."
    else:
        if value <= p5:
            position = "at_or_below_p5"
        elif value <= p25:
            position = "between_p5_and_p25"
        elif value <= p50:
            position = "between_p25_and_median"
        elif value <= p75:
            position = "between_median_and_p75"
        elif value <= p90:
            position = "between_p75_and_p90"
        else:
            position = "above_p90"
        explanation = (
            f"{value_text} falls in the '{position}' band of the retrieved MIMIC-IV distribution "
            f"(P5={_fmt(p5, unit)}, P25={_fmt(p25, unit)}, median={_fmt(p50, unit)}, "
            f"P75={_fmt(p75, unit)}, P90={_fmt(p90, unit)})."
        )

    return {
        "percentile_position": position,
        "explanation": explanation,
        "used_percentiles": {"p5": p5, "p25": p25, "p50": p50, "p75": p75, "p90": p90},
    }
