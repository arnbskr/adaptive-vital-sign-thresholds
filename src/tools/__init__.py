"""Phase 2 deterministic tools for the ICU Trajectory RAG Assistant.

Each tool is a pure, auditable function that performs ONE deterministic job
(data lookup, threshold comparison, percentile comparison, RAG retrieval, or
report assembly). The single LLM agent (``src.agent``) and the MCP server
(``src.mcp_server``) call these tools; the LLM never performs the calculations
itself. Tools are descriptive and academic only -- never clinical.
"""

from .calculator import calculatrice_medicale
from .data_availability import check_data_availability
from .icu_feature_tools import (
    compare_age_groups,
    compare_time_windows,
    detect_clinical_advice_request,
    generate_evidence_card,
    get_variable_summary,
    list_available_variables,
    plot_variable_distribution,
    query_cohort_statistics,
)
from .percentile_comparison import compare_to_percentiles
from .project_context import retrieve_project_context
from .report_generator import generate_patient_interpretation_report
from .threshold_comparison import compare_to_standard_threshold, explain_threshold_type
from .vital_summary import get_vital_summary

__all__ = [
    # Phase 2 (vital-sign) tools.
    "check_data_availability",
    "get_vital_summary",
    "compare_to_standard_threshold",
    "explain_threshold_type",
    "compare_to_percentiles",
    "retrieve_project_context",
    "generate_patient_interpretation_report",
    "calculatrice_medicale",
    # Phase 3 (multi-variable ICU) tools.
    "list_available_variables",
    "get_variable_summary",
    "query_cohort_statistics",
    "compare_age_groups",
    "compare_time_windows",
    "generate_evidence_card",
    "plot_variable_distribution",
    "detect_clinical_advice_request",
]
