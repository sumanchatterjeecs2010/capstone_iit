"""
medical_assistant.py
--------------------
Run one patient case through the LangGraph and package the JSON record.

Two LLMs
- MedGemma 1.5 4B: medical image + note understanding.
- Llama 3.2: entity extraction, triage JSON, and conversation.

LangGraph is the orchestrator.
"""

import json
import os
import time

from dataset_builder import GROUND_TRUTH
from graph_pipeline import DISCLAIMER, build_graph


_GRAPH = None


def get_graph(client):
    """Compile the LangGraph once per process."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph(client)
    return _GRAPH


def read_text_file(path):
    """Read a UTF-8 patient note from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def evaluate_case(case_id, record):
    """
    Compute simple keyword recall and triage match for one synthetic case.

    Because the dataset is synthetic, ground-truth keywords are known.
    This is a lightweight educational metric, not a clinical validation.
    """
    truth = GROUND_TRUTH[case_id]
    blob = json.dumps(record).lower()
    keywords = truth["condition_keywords"]
    hits = [kw for kw in keywords if kw.lower() in blob]
    predicted_triage = str(record.get("clinical_reasoning", {}).get("triage", "")).lower()
    expected_triage = truth["triage"].lower()
    return {
        "case_id": case_id,
        "keyword_recall": round(len(hits) / float(len(keywords)), 3),
        "keywords_found": hits,
        "keywords_expected": keywords,
        "triage_match": int(expected_triage in predicted_triage or predicted_triage in expected_triage),
        "predicted_triage": predicted_triage,
        "expected_triage": expected_triage,
    }


def process_case(client, vision_model, llama_model, case_id, image_path, text_path):
    """
    Run the full two-LLM LangGraph pipeline for a single patient case.

    Parameters
    ----------
    client : OllamaClient
        Local Ollama wrapper.
    vision_model : str
        MedGemma Ollama tag.
    llama_model : str
        Llama 3.2 Ollama tag.
    case_id : int
        Case number (1-5).
    image_path : str
        Medical image path.
    text_path : str
        Prescription / patient-detail path.

    Returns
    -------
    dict
        Complete JSON-serialisable case record.
    """
    started = time.time()
    note = read_text_file(text_path)
    graph = get_graph(client)
    state = graph.invoke(
        {
            "case_id": case_id,
            "image_path": image_path,
            "note": note,
            "vision_model": vision_model,
            "llama_model": llama_model,
        }
    )
    record = {
        "case_id": "patient_{:02d}".format(case_id),
        "input": {
            "image": os.path.basename(image_path),
            "prescription": os.path.basename(text_path),
        },
        "visual_analysis": state.get("visual_analysis"),
        "extracted_entities": state.get("extracted_entities"),
        "clinical_reasoning": state.get("clinical_reasoning"),
        "conversation": state.get("conversation"),
        "disclaimer": DISCLAIMER,
        "models": {
            "vision_llm": vision_model,
            "language_model": llama_model,
            "orchestrator": "langgraph",
        },
        "elapsed_seconds": round(time.time() - started, 2),
    }
    record["evaluation"] = evaluate_case(case_id, record)
    return record


def save_json(path, payload):
    """Write a JSON document with UTF-8 encoding and indentation."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def aggregate_metrics(records):
    """Average per-case keyword recall and triage accuracy."""
    evals = [item["evaluation"] for item in records]
    n = max(len(evals), 1)
    recall = sum(item["keyword_recall"] for item in evals) / float(n)
    triage = sum(item["triage_match"] for item in evals) / float(n)
    return {
        "n_cases": len(evals),
        "mean_keyword_recall": round(recall, 3),
        "triage_accuracy": round(triage, 3),
        "per_case": evals,
        "note": "Metrics are keyword matches on synthetic educational cases, not clinical diagnostic accuracy.",
    }
