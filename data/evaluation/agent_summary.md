# Phase 2 Agent Evaluation

Single-agent orchestration over deterministic MCP tools. Descriptive, non-clinical.

## Aggregate metrics

| Metric | Value |
| --- | --- |
| Scenarios | 5 |
| Tool-call success rate | 1.0 |
| Exact context-extraction rate | 1.0 |
| Data-availability checked (patient questions) | 1.0 |
| Grounded answer present | 1.0 |
| Non-clinical warning present | 1.0 |
| Expected tools called | 1.0 |
| Average tool latency (ms) | 100.32 |

## Per-scenario

| Question | Type (got/expected) | Tools called | Success | Context OK | Warning |
| --- | --- | --- | --- | --- | --- |
| For a patient aged 82 with mean HR 104 bpm in the first 24h … | patient_value_question/patient_value_question | check_data_availability|get_vital_summary|compare_to_standard_threshold|compare_to_percentiles|retrieve_project_context|generate_patient_interpretation_report | 1.0 | True | True |
| For a patient aged 78 with MAP 62 mmHg in the first 24h ICU … | patient_value_question/patient_value_question | check_data_availability|get_vital_summary|compare_to_standard_threshold|compare_to_percentiles|retrieve_project_context|generate_patient_interpretation_report | 1.0 | True | True |
| For a patient aged 80 with SpO2 90% in the first 24h ICU sta… | patient_value_question/patient_value_question | check_data_availability|get_vital_summary|compare_to_standard_threshold|compare_to_percentiles|retrieve_project_context|generate_patient_interpretation_report | 1.0 | True | True |
| What is the difference between a standard clinical threshold… | concept_question/concept_question | retrieve_project_context|explain_threshold_type | 1.0 | True | True |
| Which MIMIC-IV tables are used to derive ICU vital-sign summ… | dataset_question/dataset_question | retrieve_project_context | 1.0 | True | True |
