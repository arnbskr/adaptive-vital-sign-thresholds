"""Phase 2 deterministic tools for the ICU Trajectory RAG Assistant.

Each tool is a pure, auditable function that performs ONE deterministic job
(data lookup, threshold comparison, percentile comparison, RAG retrieval, or
report assembly). The single LLM agent (``src.agent``) and the MCP server
(``src.mcp_server``) call these tools; the LLM never performs the calculations
itself. Tools are descriptive and academic only -- never clinical.
"""

from .data_availability import check_data_availability
from .percentile_comparison import compare_to_percentiles
from .project_context import retrieve_project_context
from .report_generator import generate_patient_interpretation_report
from .threshold_comparison import compare_to_standard_threshold, explain_threshold_type
from .vital_summary import get_vital_summary

__all__ = [
    "check_data_availability",
    "get_vital_summary",
    "compare_to_standard_threshold",
    "explain_threshold_type",
    "compare_to_percentiles",
    "retrieve_project_context",
    "generate_patient_interpretation_report",
]
