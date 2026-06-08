from __future__ import annotations

import logging
import os
import re
from typing import Any

import pandas as pd

from .rag_utils import (
    INTENT_CONCEPT,
    INTENT_DATASET,
    INTENT_PATIENT_VALUE,
    INTENT_PIPELINE,
    INTENT_VITAL_THRESHOLD,
    detect_query_intent,
    detect_threshold_condition,
    extract_vital_value_from_query,
    infer_age_group_from_query,
    infer_direction_from_query,
    infer_time_window_from_query,
    infer_temperature_itemid_from_query,
    infer_vital_sign_from_query,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MANDATORY_DISCLAIMER = "Cette réponse est une aide académique à l’interprétation et ne constitue pas une décision clinique."
INSUFFICIENT_MESSAGE = "Les sources récupérées ne permettent pas de répondre de manière fiable à cette question."


def _format_chunk_reference(chunk: dict[str, Any]) -> str:
    title = chunk.get("title") or chunk.get("source_file") or chunk.get("doc_id")
    score = float(chunk.get("final_score", 0.0))
    return f"- {title} ({chunk.get('source_type')}, score={score:.3f})"


def _normalize_itemid(value: object) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _sources_used(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    sources: list[dict[str, Any]] = []
    for chunk in chunks:
        key = (str(chunk.get("source_file")), str(chunk.get("title")))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "source_file": chunk.get("source_file"),
                "source_type": chunk.get("source_type"),
                "title": chunk.get("title"),
                "score": float(chunk.get("final_score", 0.0)),
            }
        )
    return sources


def _first_matching_chunk(
    retrieved_chunks: list[dict[str, Any]],
    vital_sign: str | None,
    preferred_itemid: int | None = None,
) -> dict[str, Any] | None:
    matching = [
        chunk
        for chunk in retrieved_chunks
        if str(chunk.get("source_type", "")).lower() == "mimic_stats"
        and (not vital_sign or str(chunk.get("vital_sign", "")).lower() == vital_sign.lower())
        and (preferred_itemid is None or _normalize_itemid(chunk.get("itemid")) == preferred_itemid)
    ]
    if matching:
        return matching[0]
    if preferred_itemid is not None:
        matching = [
            chunk
            for chunk in retrieved_chunks
            if str(chunk.get("source_type", "")).lower() == "mimic_stats"
            and (not vital_sign or str(chunk.get("vital_sign", "")).lower() == vital_sign.lower())
        ]
    return matching[0] if matching else None


def _parse_number(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _extract_summary_details(chunk: dict[str, Any] | None) -> dict[str, float]:
    if chunk is None:
        return {}
    text = str(chunk.get("chunk_text", ""))
    patterns = {
        "mean": r"Mean(?: HR)?:\s*([0-9]+(?:\.[0-9]+)?)",
        "median": r"Median(?: HR)?:\s*([0-9]+(?:\.[0-9]+)?)",
        "p5": r"P5(?: HR)?:\s*([0-9]+(?:\.[0-9]+)?)",
        "p25": r"P25(?: HR)?:\s*([0-9]+(?:\.[0-9]+)?)",
        "p50": r"P50(?: HR)?:\s*([0-9]+(?:\.[0-9]+)?)",
        "p75": r"P75(?: HR)?:\s*([0-9]+(?:\.[0-9]+)?)",
        "p90": r"P90(?: HR)?:\s*([0-9]+(?:\.[0-9]+)?)",
        "standard_low": r"standard low threshold:\s*([0-9]+(?:\.[0-9]+)?)",
        "standard_high": r"standard high threshold:\s*([0-9]+(?:\.[0-9]+)?)",
        "percent_above_standard_high": r"percent_above_standard_high:\s*([0-9]+(?:\.[0-9]+)?)",
        "percent_below_standard_low": r"percent_below_standard_low:\s*([0-9]+(?:\.[0-9]+)?)",
    }
    details: dict[str, float] = {}
    for key, pattern in patterns.items():
        value = _parse_number(pattern, text)
        if value is not None:
            details[key] = value
    # Fall back to metadata when available.
    for key in ["standard_low", "standard_high"]:
        if key not in details and chunk.get(key) not in {None, ""}:
            try:
                details[key] = float(chunk.get(key))
            except (TypeError, ValueError):
                pass
    return details


def _age_context_lines(query: str) -> tuple[list[str], str | None]:
    age_group, precise = infer_age_group_from_query(query)
    age = infer_age_group_from_query(query)[0]
    lines: list[str] = []
    if age_group:
        lines.append(f"Age inféré: {age_group}" + (" (détection précise à partir de l’âge mentionné)" if precise else ""))
    return lines, age_group


def _patient_context_lines(query: str, numeric_value: float | None, intent: str) -> tuple[list[str], dict[str, Any]]:
    age_group, age_precise = infer_age_group_from_query(query)
    time_window, time_precise = infer_time_window_from_query(query)
    vital_sign, vital_precise = infer_vital_sign_from_query(query)
    direction = infer_direction_from_query(query)
    threshold_condition = detect_threshold_condition(query)
    context_lines: list[str] = []
    if age_group:
        context_lines.append(f"Age inféré: {age_group}" + (" (détection précise à partir de l’âge mentionné)" if age_precise else ""))
    if time_window:
        context_lines.append(f"Fenêtre temporelle inférée: {time_window}" + (" (détection précise)" if time_precise else ""))
    if vital_sign:
        context_lines.append(f"Signe vital inféré: {vital_sign}" + (" (détection précise)" if vital_precise else ""))
    context_lines.append(f"Direction inférée: {direction}")
    context_lines.append(f"Forme de question: {'threshold condition' if threshold_condition else 'exact value / contextual value'}")
    if numeric_value is not None:
        context_lines.append(f"Valeur numérique extraite: {numeric_value:.1f}")
    return context_lines, {
        "age_group": age_group,
        "time_window": time_window,
        "vital_sign": vital_sign,
        "direction": direction,
        "threshold_condition": threshold_condition,
    }


def _threshold_line(vital_sign: str | None, value: float | None, direction: str, threshold_condition: bool) -> str:
    if value is None:
        return "Aucune valeur numérique exploitable n’a été extraite de la question."
    if threshold_condition:
        if direction == "high":
            return f"La question décrit un seuil de type > {value:.1f} ou équivalent, donc la valeur {value:.1f} est la frontière, pas une mesure au-dessus de la frontière."
        if direction == "low":
            return f"La question décrit un seuil de type < {value:.1f} ou équivalent, donc la valeur {value:.1f} est la frontière, pas une mesure en dessous de la frontière."
    if vital_sign == "Heart Rate":
        if value < 60:
            return f"HR = {value:.1f} bpm est sous le seuil standard de bradycardie de 60 bpm."
        if value > 100:
            return f"HR = {value:.1f} bpm est au-dessus du seuil standard de tachycardie de 100 bpm."
        return f"HR = {value:.1f} bpm se situe entre les seuils standards de 60 et 100 bpm."
    if vital_sign == "MAP":
        return f"MAP = {value:.1f} mmHg est à comparer au seuil standard bas de 65 mmHg."
    if vital_sign == "Respiratory Rate":
        if value < 12:
            return f"Respiratory rate = {value:.1f} est sous le seuil standard bas de 12/min."
        if value > 20:
            return f"Respiratory rate = {value:.1f} est au-dessus de la plage standard 12-20/min."
        return f"Respiratory rate = {value:.1f} est dans la plage standard 12-20/min."
    if vital_sign == "SpO2":
        return f"SpO2 = {value:.1f}% est à comparer au seuil standard bas de 92%."
    if vital_sign == "Systolic Blood Pressure":
        if value < 90:
            return f"Systolic blood pressure = {value:.1f} mmHg est sous le seuil standard bas de 90 mmHg."
        if value > 140:
            return f"Systolic blood pressure = {value:.1f} mmHg est au-dessus du seuil standard haut de 140 mmHg."
        return f"Systolic blood pressure = {value:.1f} mmHg se situe dans la plage standard 90-140 mmHg."
    if vital_sign == "Diastolic Blood Pressure":
        if value < 60:
            return f"Diastolic blood pressure = {value:.1f} mmHg est sous le seuil standard bas de 60 mmHg."
        if value > 90:
            return f"Diastolic blood pressure = {value:.1f} mmHg est au-dessus du seuil standard haut de 90 mmHg."
        return f"Diastolic blood pressure = {value:.1f} mmHg se situe dans la plage standard 60-90 mmHg."
    if vital_sign == "Temperature":
        if value < 36:
            return f"Temperature = {value:.1f} est sous le seuil standard bas de 36°C/96.8°F selon l’unité du résumé récupéré."
        if value > 38:
            return f"Temperature = {value:.1f} est au-dessus du seuil standard haut de 38°C/100.4°F selon l’unité du résumé récupéré."
        return f"Temperature = {value:.1f} est dans une plage standard de référence selon l’unité du résumé récupéré."
    return "Le seuil standard doit être interprété en fonction du signe vital visé et de la valeur numérique extraite."


def _percentile_line(direction: str, value: float | None, details: dict[str, float]) -> str:
    if value is None:
        return "Aucune comparaison percentile n’est possible sans valeur numérique."
    if direction == "high":
        if "p90" in details and value > details["p90"]:
            return f"La valeur de {value:.1f} est au-dessus de la queue supérieure du résumé MIMIC-IV récupéré (au-dessus de P90={details['p90']:.1f})."
        if "p75" in details and value > details["p75"]:
            upper = f"P90={details['p90']:.1f}" if "p90" in details else "la borne supérieure du sous-groupe"
            return f"La valeur de {value:.1f} est élevée par rapport à la distribution récupérée et se situe entre P75={details['p75']:.1f} et {upper}."
        if "p75" in details:
            return f"La valeur de {value:.1f} n’est pas particulièrement élevée par rapport à la distribution récupérée et reste à ou sous P75={details['p75']:.1f}."
        return f"La valeur de {value:.1f} ne peut pas être positionnée précisément dans les percentiles sans P75/P90."
    if direction == "low":
        if "p5" in details and value <= details["p5"]:
            return f"La valeur de {value:.1f} est très basse par rapport à la distribution récupérée (à ou sous P5={details['p5']:.1f})."
        if "p25" in details and value <= details["p25"]:
            return f"La valeur de {value:.1f} est basse par rapport à la distribution récupérée (à ou sous P25={details['p25']:.1f})."
        if "p50" in details and value <= details["p50"]:
            return f"La valeur de {value:.1f} est sous la médiane du résumé récupéré (P50={details['p50']:.1f})."
        if "p50" in details:
            return f"La valeur de {value:.1f} n’est pas basse par rapport à la distribution récupérée et se situe au-dessus de P50={details['p50']:.1f}."
        return f"La valeur de {value:.1f} ne peut pas être positionnée précisément dans les percentiles sans P5/P25/P50."
    return "La direction n’est pas suffisamment claire pour une lecture percentile ciblée."


def _summary_context_lines(summary_chunk: dict[str, Any] | None, details: dict[str, float]) -> list[str]:
    if summary_chunk is None:
        return []
    item_label = summary_chunk.get("label") or summary_chunk.get("vital_sign") or "the retrieved vital sign"
    item_window = summary_chunk.get("time_window") or "the retrieved window"
    item_age = summary_chunk.get("age_group") or "the retrieved age group"
    lines = [
        f"Le résumé MIMIC-IV récupéré pour {item_label} dans la fenêtre {item_window} et le groupe d’âge {item_age} décrit la distribution observée.",
    ]
    if "standard_low" in details or "standard_high" in details:
        threshold_bits = []
        if "standard_low" in details:
            threshold_bits.append(f"standard low={details['standard_low']:.1f}")
        if "standard_high" in details:
            threshold_bits.append(f"standard high={details['standard_high']:.1f}")
        if threshold_bits:
            lines.append(f"Seuils standards du résumé récupéré: {', '.join(threshold_bits)}.")
    if "mean" in details:
        lines.append(
            f"Résumé statistique: mean={details.get('mean', float('nan')):.1f}, median={details.get('median', float('nan')):.1f}, P75={details.get('p75', float('nan')):.1f}, P90={details.get('p90', float('nan')):.1f}."
        )
    return lines


def _missing_vital_response(vital_sign: str | None, source_chunks: list[dict[str, Any]]) -> str:
    source_lines = [
        "1. Réponse courte",
        INSUFFICIENT_MESSAGE,
        "",
        "2. Patient context inferred from the question",
        f"Signe vital inféré: {vital_sign or 'unknown'}",
        "",
        "3. Interprétation selon les seuils standards",
        _threshold_line(vital_sign, None, "neutral", False),
        "",
        "4. Interprétation selon les statistiques MIMIC-IV récupérées",
        f"No reliable {vital_sign or 'requested vital sign'} statistical summary was found in the current index.",
        "Les sources récupérées ne permettent pas de répondre de manière fiable à cette question.",
        "",
        "5. Limites",
        "Le prototype ne peut pas comparer cette mesure à des percentiles MIMIC-IV spécifiques tant qu’aucun résumé statistique correspondant n’est disponible dans l’index.",
        "",
        "6. Sources utilisées",
    ]
    if source_chunks:
        source_lines.extend([_format_chunk_reference(chunk) for chunk in source_chunks[:5]])
    else:
        source_lines.append("Aucune source fiable exploitée.")
    source_lines.extend(["", MANDATORY_DISCLAIMER])
    return "\n".join(source_lines)


def _patient_value_answer(query: str, retrieved_chunks: list[dict[str, Any]], intent: str) -> str:
    context_lines, context = _patient_context_lines(query, None, intent)
    numeric_value = extract_vital_value_from_query(query, context["vital_sign"])
    context_lines, context = _patient_context_lines(query, numeric_value, intent)
    preferred_itemid = infer_temperature_itemid_from_query(query) if context["vital_sign"] == "Temperature" else None
    summary_chunk = _first_matching_chunk(retrieved_chunks, context["vital_sign"], preferred_itemid)
    details = _extract_summary_details(summary_chunk)

    if context["vital_sign"] and summary_chunk is None:
        return _missing_vital_response(context["vital_sign"], retrieved_chunks)

    short_answer = INSUFFICIENT_MESSAGE if summary_chunk is None else ""
    if summary_chunk is not None and numeric_value is not None:
        threshold_line = _threshold_line(context["vital_sign"], numeric_value, context["direction"], context["threshold_condition"])
        percentile_line = _percentile_line(context["direction"], numeric_value, details)
        if context["direction"] == "low":
            short_answer = f"La valeur {numeric_value:.1f} est interprétée comme basse ou non-basse selon le seuil standard et la distribution récupérée."
        elif context["direction"] == "high":
            short_answer = f"La valeur {numeric_value:.1f} est interprétée comme élevée ou non-élevée selon le seuil standard et la distribution récupérée."
        else:
            short_answer = f"La valeur {numeric_value:.1f} est interprétée dans le contexte du seuil standard et de la distribution récupérée."
        lines = [
            "1. Réponse courte",
            short_answer,
            "",
            "2. Patient context inferred from the question",
            *context_lines,
            "",
            "3. Interprétation selon les seuils standards",
            threshold_line,
            "",
            "4. Interprétation selon les statistiques MIMIC-IV récupérées",
            percentile_line,
            *(_summary_context_lines(summary_chunk, details)),
            "",
            "5. Limites",
            "Cette lecture est descriptive et non diagnostique. Elle compare une valeur contextualisée à des références standards et à une distribution MIMIC-IV limitée, sans tenir compte de l’ensemble du contexte clinique individuel.",
            "",
            "6. Sources utilisées",
            *[_format_chunk_reference(chunk) for chunk in retrieved_chunks[:5]],
            "",
            MANDATORY_DISCLAIMER,
        ]
        return "\n".join(lines)

    return _missing_vital_response(context["vital_sign"], retrieved_chunks)


def _vital_threshold_answer(query: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    _, context = _patient_context_lines(query, extract_vital_value_from_query(query, infer_vital_sign_from_query(query)[0]), INTENT_VITAL_THRESHOLD)
    context_lines, context = _patient_context_lines(query, extract_vital_value_from_query(query, infer_vital_sign_from_query(query)[0]), INTENT_VITAL_THRESHOLD)
    preferred_itemid = infer_temperature_itemid_from_query(query) if context["vital_sign"] == "Temperature" else None
    summary_chunk = _first_matching_chunk(retrieved_chunks, context["vital_sign"], preferred_itemid)
    details = _extract_summary_details(summary_chunk)

    if context["vital_sign"] and summary_chunk is None:
        return _missing_vital_response(context["vital_sign"], retrieved_chunks)

    numeric_value = extract_vital_value_from_query(query, context["vital_sign"])
    direction = context["direction"]
    threshold_condition = context["threshold_condition"]

    if numeric_value is None:
        numeric_value = _parse_number(r"\b(\d+(?:\.\d+)?)\b", query)

    threshold_line = _threshold_line(context["vital_sign"], numeric_value, direction, threshold_condition)
    percentile_line = _percentile_line(direction, numeric_value, details)

    short_answer = "Cette question porte sur un seuil de référence et non sur une valeur mesurée précise."
    if context["vital_sign"] == "Heart Rate" and direction == "high":
        short_answer = "HR > 100 bpm correspond à un seuil standard de tachycardie et doit être lu comme une condition de seuil, pas comme une mesure exacte."
    elif context["vital_sign"] and direction == "low":
        short_answer = f"La condition basse associée à {context['vital_sign']} se lit comme une comparaison à un seuil standard inférieur."

    lines = [
        "1. Réponse courte",
        short_answer,
        "",
        "2. Patient context inferred from the question",
        *context_lines,
        "",
        "3. Interprétation selon les seuils standards",
        threshold_line,
        "",
        "4. Interprétation selon les statistiques MIMIC-IV récupérées",
        percentile_line,
        *(_summary_context_lines(summary_chunk, details) if summary_chunk else ["Les sources récupérées ne permettent pas de positionner ce seuil par rapport à une distribution MIMIC-IV spécifique."]),
        "",
        "5. Limites",
        "Cette lecture est descriptive et non diagnostique. Un seuil clinique est une frontière de référence; son interprétation en ICU reste contextuelle et ne se traduit pas automatiquement par une décision clinique.",
        "",
        "6. Sources utilisées",
        *[_format_chunk_reference(chunk) for chunk in retrieved_chunks[:5]],
        "",
        MANDATORY_DISCLAIMER,
    ]
    return "\n".join(lines)


def _concept_answer(retrieved_chunks: list[dict[str, Any]]) -> str:
    project_sources = [chunk for chunk in retrieved_chunks if str(chunk.get("source_type", "")).lower() in {"project_report", "documentation", "article", "guideline"}]
    sources = project_sources or retrieved_chunks
    lines = [
        "1. Réponse courte",
        "A standard clinical threshold is a fixed reference value; an adaptive percentile-based threshold is derived from the observed distribution in a specific population or context.",
        "",
        "2. Patient context inferred from the question",
        "Aucun contexte patient spécifique n’est requis pour cette question conceptuelle.",
        "",
        "3. Interprétation selon les seuils standards",
        "Les seuils standards sont simples à appliquer et faciles à communiquer, mais ils peuvent ignorer les différences de population et de contexte ICU.",
        "",
        "4. Interprétation selon les statistiques MIMIC-IV récupérées",
        "Les résumés MIMIC-IV servent ici d’exemple de seuils adaptatifs: ils décrivent comment une distribution observée peut être résumée par des percentiles sans prétendre remplacer une règle clinique.",
        "",
        "5. Limites",
        "Les seuils adaptatifs sont descriptifs et contextuels; ils ne sont pas automatiquement des règles de décision clinique. Cette question est conceptuelle, donc les résumés physiologiques ne sont utilisés qu’en illustration.",
        "",
        "6. Sources utilisées",
        *[_format_chunk_reference(chunk) for chunk in sources[:5]],
        "",
        MANDATORY_DISCLAIMER,
    ]
    return "\n".join(lines)


def _dataset_answer(retrieved_chunks: list[dict[str, Any]]) -> str:
    lines = [
        "1. Réponse courte",
        "The ICU vital-sign workflow relies on hosp.patients, icu.icustays, icu.chartevents, and icu.d_items.",
        "",
        "2. Patient context inferred from the question",
        "Aucun patient n’est visé ici; la question porte sur la structure du dataset.",
        "",
        "3. Interprétation selon les seuils standards",
        "Les tables de données ne définissent pas des seuils physiologiques; elles fournissent les variables, identifiants et métadonnées nécessaires à l’analyse.",
        "",
        "4. Interprétation selon les statistiques MIMIC-IV récupérées",
        "hosp.patients fournit subject_id, gender et anchor_age; icu.icustays fournit stay_id, intime, outtime et los; icu.chartevents contient les mesures horodatées; icu.d_items mappe itemid vers label, unitname et category. chartevents doit toujours être filtrée par itemid pour éviter des scans massifs et des variables non pertinentes.",
        "",
        "5. Limites",
        "Cette question est de nature documentaire. Les réponses doivent rester centrées sur les tables et sur la logique de pipeline, sans interpréter des percentiles physiologiques non pertinents.",
        "",
        "6. Sources utilisées",
        *[_format_chunk_reference(chunk) for chunk in retrieved_chunks[:5]],
        "",
        MANDATORY_DISCLAIMER,
    ]
    return "\n".join(lines)


def _pipeline_answer(retrieved_chunks: list[dict[str, Any]]) -> str:
    lines = [
        "1. Réponse courte",
        "The pipeline is BigQuery -> CSV summaries -> RAG documents -> chunks -> local TF-IDF index -> retrieval -> template-based answer -> Streamlit.",
        "",
        "2. Patient context inferred from the question",
        "Aucun contexte patient n’est requis; la question porte sur le fonctionnement du système.",
        "",
        "3. Interprétation selon les seuils standards",
        "Les seuils physiologiques ne sont pas le sujet principal ici; la question porte sur la chaîne de traitement des données.",
        "",
        "4. Interprétation selon les statistiques MIMIC-IV récupérées",
        "BigQuery extrait les données sélectionnées; data/processed contient les CSV propres et les résumés; data/rag_documents transforme les résumés et documents en texte indexable; data/rag_chunks découpe les documents; data/rag_index stocke le TF-IDF vectorizer et la matrice; Streamlit interroge l’index local et non BigQuery en direct.",
        "",
        "5. Limites",
        "Ce pipeline reste une Phase 1 locale: il n’implémente ni agents, ni MCP, ni function calling, ni orchestration.",
        "",
        "6. Sources utilisées",
        *[_format_chunk_reference(chunk) for chunk in retrieved_chunks[:5]],
        "",
        MANDATORY_DISCLAIMER,
    ]
    return "\n".join(lines)


def _missing_vital_answer(query: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    _, context = _patient_context_lines(query, None, INTENT_VITAL_THRESHOLD)
    return _missing_vital_response(context["vital_sign"], retrieved_chunks)


def _template_answer(query: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    intent = detect_query_intent(query)
    inferred_vital_sign, _ = infer_vital_sign_from_query(query)
    preferred_itemid = infer_temperature_itemid_from_query(query) if inferred_vital_sign == "Temperature" else None
    matching_stat = _first_matching_chunk(retrieved_chunks, inferred_vital_sign, preferred_itemid)
    effective_intent = intent
    if intent in {INTENT_PATIENT_VALUE, INTENT_VITAL_THRESHOLD} and inferred_vital_sign and matching_stat is None:
        effective_intent = "unsupported_or_missing_vital_question"

    if effective_intent == INTENT_PATIENT_VALUE:
        return _patient_value_answer(query, retrieved_chunks, intent)
    if effective_intent == INTENT_VITAL_THRESHOLD:
        if inferred_vital_sign and matching_stat is None:
            return _missing_vital_answer(query, retrieved_chunks)
        return _vital_threshold_answer(query, retrieved_chunks)
    if effective_intent == INTENT_CONCEPT:
        return _concept_answer(retrieved_chunks)
    if effective_intent == INTENT_DATASET:
        return _dataset_answer(retrieved_chunks)
    if effective_intent == INTENT_PIPELINE:
        return _pipeline_answer(retrieved_chunks)

    # Fallback: treat as a patient/value-style question if a vital sign is present, otherwise concept-like.
    if inferred_vital_sign:
        return _patient_value_answer(query, retrieved_chunks, INTENT_PATIENT_VALUE)
    return _concept_answer(retrieved_chunks)


def _llm_answer_stub(query: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    raise NotImplementedError(
        "LLM generation is disabled by default in Phase 1. Set use_llm=False or add your own external connector later."
    )


def generate_rag_answer(query: str, retrieved_chunks: list[dict[str, Any]], use_llm: bool = False) -> dict[str, Any]:
    answer_text = _template_answer(query, retrieved_chunks)
    if use_llm:
        llm_provider = os.getenv("RAG_LLM_PROVIDER", "").strip().lower()
        if llm_provider:
            answer_text = _llm_answer_stub(query, retrieved_chunks)

    return {
        "answer": answer_text,
        "sources": _sources_used(retrieved_chunks),
        "retrieved_chunks": retrieved_chunks,
        "intent": detect_query_intent(query),
        "effective_intent": "unsupported_or_missing_vital_question"
        if (
            detect_query_intent(query) in {INTENT_PATIENT_VALUE, INTENT_VITAL_THRESHOLD}
            and infer_vital_sign_from_query(query)[0]
            and _first_matching_chunk(
                retrieved_chunks,
                infer_vital_sign_from_query(query)[0],
                infer_temperature_itemid_from_query(query) if infer_vital_sign_from_query(query)[0] == "Temperature" else None,
            ) is None
        )
        else detect_query_intent(query),
        "inferred_age_group": infer_age_group_from_query(query)[0],
        "inferred_time_window": infer_time_window_from_query(query)[0],
        "inferred_vital_sign": infer_vital_sign_from_query(query)[0],
        "inferred_direction": infer_direction_from_query(query),
        "threshold_condition": detect_threshold_condition(query),
        "insufficient": INSUFFICIENT_MESSAGE in answer_text,
        "disclaimer": MANDATORY_DISCLAIMER,
    }
