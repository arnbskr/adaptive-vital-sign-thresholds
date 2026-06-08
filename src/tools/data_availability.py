"""Tool: check_data_availability.

Verifies whether an EXACT MIMIC-IV statistical summary exists for a
(vital_sign, age_group, time_window) triple, before any comparison is made.
If the data is absent the agent must say so explicitly and must not invent a
comparison or substitute a different vital sign.
"""

from __future__ import annotations

from typing import Any

from .vital_summary import find_summary_rows


def check_data_availability(vital_sign: str, age_group: str, time_window: str) -> dict[str, Any]:
    """Return whether an exact summary exists for the requested triple."""

    matches, source_label = find_summary_rows(vital_sign, age_group, time_window)
    if not matches.empty:
        return {
            "available": True,
            "message": "Exact MIMIC-IV summary found.",
            "matched_source": source_label,
            "matched_itemids": [str(item) for item in matches["itemid"].tolist()],
            "requested": {
                "vital_sign": vital_sign,
                "age_group": age_group,
                "time_window": time_window,
            },
        }

    return {
        "available": False,
        "message": f"No exact MIMIC-IV summary found for {vital_sign} / {age_group} / {time_window}.",
        "matched_source": None,
        "requested": {
            "vital_sign": vital_sign,
            "age_group": age_group,
            "time_window": time_window,
        },
    }
