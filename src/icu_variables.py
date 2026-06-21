"""Canonical ICU variable registry (Phase 3, single source of truth).

This module is the ONE place that enumerates every ICU variable the Phase 3
"ICU Multi-Data Agent" can explore: its category, source MIMIC-IV table,
candidate ``itemid``(s), unit, physiologic "safe" bounds for outlier filtering,
and the human-readable inclusion / cleaning rules.

Two consumers read from here so they never drift apart (Cours 14 -- single
source of truth, reproducibility):

* ``write_dictionary_csv`` emits ``data/processed/icu_variable_dictionary.csv``
  (the committable, human-facing data dictionary).
* the Phase 3 BigQuery extractor (``src/extract_icu_features.py``) reads the same
  specs to build ``icu_feature_summary.csv``.

IMPORTANT -- itemids are *candidates*. They are drawn from the public MIMIC-IV
documentation but MUST be confirmed against ``d_items`` / ``d_labitems`` with
``python -m src.validate_icu_itemids`` before any extraction. ``verified`` stays
False until that check passes. Nothing here is clinical: the registry only
supports descriptive, academic statistics.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from .config import PROCESSED_DIR

# Canonical vocabulary, kept consistent with src/semantic_rag.py and the existing
# vital-sign pipeline. Do not introduce new spellings without updating both.
AGE_GROUPS = ("65-74", "75-84", "85+")
TIME_WINDOWS = ("first_6h", "first_12h", "first_24h")

# Columns of the committable data dictionary, in the order requested for Phase 3.
DICTIONARY_COLUMNS = [
    "variable_name",
    "variable_category",
    "source_table",
    "itemid",
    "label",
    "unit",
    "description",
    "inclusion_rule",
    "cleaning_rule",
]

VARIABLE_CATEGORIES = ("vital_sign", "lab", "output", "input", "procedure", "outcome")

DICTIONARY_CSV = PROCESSED_DIR / "icu_variable_dictionary.csv"


@dataclass(frozen=True)
class VariableSpec:
    """One ICU variable. ``itemids`` empty => column-derived (outcomes)."""

    variable_name: str
    variable_category: str
    source_table: str
    itemids: tuple[int, ...]
    label: str
    unit: str
    description: str
    inclusion_rule: str
    cleaning_rule: str
    # Extraction-only metadata (not written to the dictionary CSV).
    safe_low: float | None = None
    safe_high: float | None = None
    composite: bool = False            # e.g. GCS total = eye + verbal + motor
    aggregated_only: bool = False      # cohort-level only, never patient-level output
    verified: bool = False             # flipped to True once d_items check passes
    notes: str = ""

    def itemid_str(self) -> str:
        return ";".join(str(i) for i in self.itemids)

    def to_dictionary_row(self) -> dict[str, str]:
        return {
            "variable_name": self.variable_name,
            "variable_category": self.variable_category,
            "source_table": self.source_table,
            "itemid": self.itemid_str(),
            "label": self.label,
            "unit": self.unit,
            "description": self.description,
            "inclusion_rule": self.inclusion_rule,
            "cleaning_rule": self.cleaning_rule,
        }


# Shared rule snippets to keep the registry terse and consistent.
_R_WINDOW = "charttime within the chosen window (first_6h/12h/24h) after ICU intime; valuenum not null"
_R_LAB_WINDOW = "labevent within the chosen window after ICU intime; valuenum not null; cohort = elderly ICU stays (age>=65)"
_R_DEDUP = "drop values outside safe bounds; deduplicate identical (stay_id, charttime) rows; keep numeric valuenum"


# --------------------------------------------------------------------------- #
# The registry. Vital signs mirror the existing ITEM_SPECS; the rest is new.
# --------------------------------------------------------------------------- #
ICU_VARIABLES: list[VariableSpec] = [
    # ---- Vital signs (already extracted by the Phase 1 pipeline) ---------- #
    VariableSpec("heart_rate", "vital_sign", "icu.chartevents", (220045,),
                 "Heart Rate", "bpm",
                 "Charted heart rate.", _R_WINDOW, _R_DEDUP,
                 safe_low=20, safe_high=250, verified=True),
    VariableSpec("respiratory_rate", "vital_sign", "icu.chartevents", (220210,),
                 "Respiratory Rate", "insp/min",
                 "Charted respiratory rate.", _R_WINDOW, _R_DEDUP,
                 safe_low=1, safe_high=80, verified=True),
    VariableSpec("map", "vital_sign", "icu.chartevents", (220052, 220181),
                 "Arterial/NIBP Mean", "mmHg",
                 "Mean arterial pressure (arterial line or non-invasive cuff).", _R_WINDOW, _R_DEDUP,
                 safe_low=20, safe_high=200, verified=True),
    VariableSpec("sbp", "vital_sign", "icu.chartevents", (220050, 220179),
                 "Arterial/NIBP Systolic", "mmHg",
                 "Systolic blood pressure.", _R_WINDOW, _R_DEDUP,
                 safe_low=40, safe_high=300, verified=True),
    VariableSpec("dbp", "vital_sign", "icu.chartevents", (220051, 220180),
                 "Arterial/NIBP Diastolic", "mmHg",
                 "Diastolic blood pressure.", _R_WINDOW, _R_DEDUP,
                 safe_low=20, safe_high=200, verified=True),
    VariableSpec("temperature", "vital_sign", "icu.chartevents", (223762, 223761),
                 "Temperature C/F", "degC",
                 "Body temperature; Fahrenheit (223761) is converted to Celsius.", _R_WINDOW,
                 "convert F->C for itemid 223761; " + _R_DEDUP,
                 safe_low=25, safe_high=45, verified=True),
    VariableSpec("spo2", "vital_sign", "icu.chartevents", (220277,),
                 "O2 saturation pulseoxymetry", "%",
                 "Peripheral oxygen saturation (pulse oximetry).", _R_WINDOW, _R_DEDUP,
                 safe_low=0, safe_high=100, verified=True),

    # ---- Additional charted variables (Phase 3 MVP, >=5 new) ------------- #
    VariableSpec("glucose", "vital_sign", "icu.chartevents", (220621, 225664, 226537),
                 "Glucose (serum/fingerstick/whole blood)", "mg/dL",
                 "Charted glucose; representative itemid chosen by largest count.", _R_WINDOW, _R_DEDUP,
                 safe_low=10, safe_high=2000),
    VariableSpec("fio2", "vital_sign", "icu.chartevents", (223835,),
                 "Inspired O2 Fraction", "%",
                 "Fraction of inspired oxygen; fractions in [0.21,1.0] normalized to percent.", _R_WINDOW,
                 "normalize values <=1.0 to percent (x100); " + _R_DEDUP,
                 safe_low=21, safe_high=100),
    VariableSpec("o2_flow", "vital_sign", "icu.chartevents", (223834,),
                 "O2 Flow", "L/min",
                 "Supplemental oxygen flow rate.", _R_WINDOW, _R_DEDUP,
                 safe_low=0, safe_high=100),
    VariableSpec("gcs_total", "vital_sign", "icu.chartevents", (220739, 223900, 223901),
                 "Glasgow Coma Scale (total)", "points",
                 "GCS total = eye (220739) + verbal (223900) + motor (223901), summed per charttime.", _R_WINDOW,
                 "sum the three components at matching charttime; range 3-15; " + _R_DEDUP,
                 safe_low=3, safe_high=15, composite=True),
    VariableSpec("cvp", "vital_sign", "icu.chartevents", (220074,),
                 "Central Venous Pressure", "mmHg",
                 "Central venous pressure.", _R_WINDOW, _R_DEDUP,
                 safe_low=0, safe_high=50),

    # ---- Labs (Phase 3 MVP, labevents) ----------------------------------- #
    VariableSpec("lactate", "lab", "hosp.labevents", (50813,),
                 "Lactate", "mmol/L", "Blood lactate.", _R_LAB_WINDOW, _R_DEDUP,
                 safe_low=0, safe_high=30),
    VariableSpec("creatinine", "lab", "hosp.labevents", (50912,),
                 "Creatinine", "mg/dL", "Serum creatinine.", _R_LAB_WINDOW, _R_DEDUP,
                 safe_low=0, safe_high=25),
    VariableSpec("bilirubin_total", "lab", "hosp.labevents", (50885,),
                 "Bilirubin, Total", "mg/dL", "Total bilirubin.", _R_LAB_WINDOW, _R_DEDUP,
                 safe_low=0, safe_high=60),
    VariableSpec("platelets", "lab", "hosp.labevents", (51265,),
                 "Platelet Count", "K/uL", "Platelet count.", _R_LAB_WINDOW, _R_DEDUP,
                 safe_low=0, safe_high=2000),
    VariableSpec("wbc", "lab", "hosp.labevents", (51301,),
                 "White Blood Cells", "K/uL", "White blood cell count.", _R_LAB_WINDOW, _R_DEDUP,
                 safe_low=0, safe_high=200),
    VariableSpec("hemoglobin", "lab", "hosp.labevents", (51222,),
                 "Hemoglobin", "g/dL", "Blood hemoglobin.", _R_LAB_WINDOW, _R_DEDUP,
                 safe_low=0, safe_high=25),
    VariableSpec("sodium", "lab", "hosp.labevents", (50983,),
                 "Sodium", "mEq/L", "Serum sodium.", _R_LAB_WINDOW, _R_DEDUP,
                 safe_low=80, safe_high=200),
    VariableSpec("potassium", "lab", "hosp.labevents", (50971,),
                 "Potassium", "mEq/L", "Serum potassium.", _R_LAB_WINDOW, _R_DEDUP,
                 safe_low=1, safe_high=12),
    VariableSpec("bicarbonate", "lab", "hosp.labevents", (50882,),
                 "Bicarbonate", "mEq/L", "Serum bicarbonate.", _R_LAB_WINDOW, _R_DEDUP,
                 safe_low=1, safe_high=60),
    VariableSpec("ph_blood", "lab", "hosp.labevents", (50820,),
                 "pH (blood gas)", "units", "Blood gas pH.", _R_LAB_WINDOW, _R_DEDUP,
                 safe_low=6.5, safe_high=8.0),
    VariableSpec("pao2", "lab", "hosp.labevents", (50821,),
                 "pO2 (arterial)", "mmHg", "Arterial partial pressure of oxygen.", _R_LAB_WINDOW, _R_DEDUP,
                 safe_low=10, safe_high=700),
    VariableSpec("paco2", "lab", "hosp.labevents", (50818,),
                 "pCO2 (arterial)", "mmHg", "Arterial partial pressure of CO2.", _R_LAB_WINDOW, _R_DEDUP,
                 safe_low=5, safe_high=200),
    VariableSpec("inr", "lab", "hosp.labevents", (51237,),
                 "INR(PT)", "ratio", "International normalized ratio.", _R_LAB_WINDOW, _R_DEDUP,
                 safe_low=0.5, safe_high=20),

    # ---- Outputs (Phase 3 MVP) ------------------------------------------- #
    VariableSpec("urine_output", "output", "icu.outputevents", (226559, 226560, 226561),
                 "Urine output (Foley/void)", "mL",
                 "Charted urine output volume per event.", _R_WINDOW,
                 "value (mL) within [0,5000] per event; " + _R_DEDUP,
                 safe_low=0, safe_high=5000),

    # ---- Inputs / interventions (optional, lower priority) --------------- #
    VariableSpec("norepinephrine", "input", "icu.inputevents", (221906,),
                 "Norepinephrine", "mcg/kg/min",
                 "Norepinephrine infusion rate (vasopressor marker).", _R_WINDOW,
                 "use rate where available; " + _R_DEDUP,
                 safe_low=0, safe_high=5, notes="optional; verify rate units before extraction"),
    VariableSpec("iv_fluids_nacl", "input", "icu.inputevents", (225158,),
                 "NaCl 0.9%", "mL",
                 "0.9% saline volume administered.", _R_WINDOW,
                 "sum amount (mL) per window; " + _R_DEDUP,
                 safe_low=0, safe_high=20000, notes="optional"),

    # ---- Procedures (optional) ------------------------------------------- #
    VariableSpec("invasive_ventilation", "procedure", "icu.procedureevents", (225792,),
                 "Invasive Ventilation", "presence",
                 "Marker of invasive mechanical ventilation during the stay.", _R_WINDOW,
                 "presence/duration flag; " + _R_DEDUP,
                 notes="optional; presence flag only"),

    # ---- Outcomes (cohort-aggregated ONLY) ------------------------------- #
    VariableSpec("icu_los", "outcome", "icu.icustays", (),
                 "ICU length of stay", "days",
                 "ICU length of stay (icustays.los).", "one row per ICU stay",
                 "drop negative/implausible los", aggregated_only=True),
    VariableSpec("hospital_los", "outcome", "hosp.admissions", (),
                 "Hospital length of stay", "days",
                 "Hospital LOS derived from admittime/dischtime.", "one row per admission",
                 "drop negative/implausible los", aggregated_only=True),
    VariableSpec("admission_type", "outcome", "hosp.admissions", (),
                 "Admission type", "category",
                 "Admission type (e.g. EW EMER., ELECTIVE).", "one row per admission",
                 "categorical; no cleaning", aggregated_only=True),
    VariableSpec("first_careunit", "outcome", "icu.icustays", (),
                 "First care unit", "category",
                 "First ICU care unit of the stay.", "one row per ICU stay",
                 "categorical; no cleaning", aggregated_only=True),
    VariableSpec("hospital_mortality", "outcome", "hosp.admissions", (),
                 "Hospital mortality flag", "flag",
                 "In-hospital mortality (admissions.hospital_expire_flag). "
                 "Reported ONLY as a cohort-aggregated rate, never per patient.",
                 "one row per admission",
                 "0/1 flag; aggregate to a rate per cohort", aggregated_only=True),
]


REGISTRY: dict[str, VariableSpec] = {spec.variable_name: spec for spec in ICU_VARIABLES}


def variables_by_category(category: str | None = None) -> list[VariableSpec]:
    if category is None:
        return list(ICU_VARIABLES)
    return [spec for spec in ICU_VARIABLES if spec.variable_category == category]


def to_dictionary_rows() -> list[dict[str, str]]:
    return [spec.to_dictionary_row() for spec in ICU_VARIABLES]


def write_dictionary_csv(path: Path | None = None) -> Path:
    """Write the committable data dictionary CSV from the registry."""

    target = Path(path) if path is not None else DICTIONARY_CSV
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DICTIONARY_COLUMNS)
        writer.writeheader()
        for row in to_dictionary_rows():
            writer.writerow(row)
    return target


def main() -> None:
    path = write_dictionary_csv()
    n_by_cat = {cat: len(variables_by_category(cat)) for cat in VARIABLE_CATEGORIES}
    print(f"Wrote {len(ICU_VARIABLES)} variables to {path}")
    print("By category:", ", ".join(f"{k}={v}" for k, v in n_by_cat.items()))
    unverified = [s.variable_name for s in ICU_VARIABLES if not s.verified and s.itemids]
    if unverified:
        print(
            f"\n{len(unverified)} itemid-based variables are UNVERIFIED. "
            "Run `python -m src.validate_icu_itemids` before extraction:"
        )
        print("  " + ", ".join(unverified))


if __name__ == "__main__":
    main()
