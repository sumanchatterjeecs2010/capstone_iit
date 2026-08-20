"""
medical_assistant.py
--------------------
Application layer: run one clinical case through LangGraph and manage
``conversation.json`` I/O.

Public API (reuse from scripts or the TUI)
------------------------------------------
- ``process_case(...)`` — analyze note + image; return a full case ``record`` dict
- ``save_json`` / ``load_conversation`` — persist and reload that record
- ``display_payload(record)`` — slim view for terminal printing
- ``process_follow_up(...)`` — append a clinician question and re-triage

The default output path is ``paths.CONVERSATION_PATH`` (Codebase/conversation.json).
"""

import json
import os
import time

from evaluation import evaluate_record
from graph_pipeline import UnsupportedImageDomainError, build_graph, follow_up_with_llama
from ingestion import IMAGE_DOMAINS, prepare_existing_paths
from paths import CONVERSATION_PATH


# Compiled LangGraph shared across calls in one process (lazy init).
_GRAPH = None


def get_graph(client):
    """Return a process-wide compiled LangGraph bound to *client*."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph(client)
    return _GRAPH


def read_text(path):
    """Read a UTF-8 text file and strip surrounding whitespace."""
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def _resolve_inputs(text_path, image_path):
    """
    De-identify note and image into a temporary work folder for model input.

    Returns
    -------
    tuple
        ``(note_text, safe_note_path, safe_image_path, ingested_dict)``
    """
    ingested = prepare_existing_paths(text_path, image_path)
    return (
        read_text(ingested["text"]["path"]),
        ingested["text"]["path"],
        ingested["image"]["path"],
        ingested,
    )


def process_case(client, vision_model, llama_model, case_id, image_path, text_path):
    """
    Run the full multimodal pipeline on one note+image pair.

    Parameters
    ----------
    client : OllamaClient
        Live Ollama HTTP client.
    vision_model, llama_model : str
        Resolved Ollama model tags.
    case_id : int
        Numeric id used only for labeling ``case_id`` in the record (0 -> "upload").
    image_path, text_path : str
        Paths to the original clinician-supplied files.

    Returns
    -------
    dict
        Case record with visual analysis, entities, reasoning, conversation,
        privacy audit, and label-free ``evaluation`` metrics.

    Raises
    ------
    UnsupportedImageDomainError
        If MedGemma does not classify the image as radiology or pathology.
    """
    started = time.time()
    note, safe_note_path, safe_image_path, ingested = _resolve_inputs(text_path, image_path)
    state = get_graph(client).invoke(
        {
            "image_path": safe_image_path,
            "note": note,
            "vision_model": vision_model,
            "llama_model": llama_model,
        }
    )
    visual = state.get("visual_analysis") or {}
    detected_domain = state.get("image_domain") or visual.get("image_domain")
    if detected_domain not in IMAGE_DOMAINS:
        raise UnsupportedImageDomainError(
            "This assistant accepts radiology scans and histopathology slides only."
        )
    entities = state.get("extracted_entities") or {}
    record = {
        "case_id": "patient_{:02d}".format(case_id) if case_id else "upload",
        "input": {
            "image": os.path.basename(image_path),
            "prescription": os.path.basename(text_path),
            "image_domain": detected_domain,
            "domain_detection": "automatic",
            "deidentified_image": os.path.basename(safe_image_path),
            "deidentified_note": os.path.basename(safe_note_path),
        },
        "privacy": {
            "text": ingested["text"]["privacy"],
            "image": ingested["image"]["privacy"],
            "originals_stored": False,
        },
        "visual_analysis": visual,
        "extracted_entities": entities,
        "clinical_reasoning": state.get("clinical_reasoning"),
        "conversation": state.get("conversation"),
        "note_text": note,
        "models": {
            "vision_llm": vision_model,
            "language_model": llama_model,
            "orchestrator": "langgraph",
        },
        "elapsed_seconds": round(time.time() - started, 2),
    }
    record["evaluation"] = evaluate_record(record)
    return record


def save_json(path, payload):
    """Write *payload* as pretty-printed UTF-8 JSON."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def default_conversation_path():
    """Return the default ``Codebase/conversation.json`` path."""
    return CONVERSATION_PATH


def load_conversation(path=None):
    """Load a previously saved case record from *path* (or the default)."""
    path = path or CONVERSATION_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError("No conversation file at {}".format(path))
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def display_payload(record):
    """
    Flatten a case record into fields used by the terminal summary / TUI.

    Returns a dict with triage, impression, findings, conversation, evaluation, etc.
    """
    visual = record.get("visual_analysis") or {}
    reasoning = record.get("clinical_reasoning") or {}
    return {
        "image_domain": (record.get("input") or {}).get("image_domain")
        or (visual.get("image_domain")),
        "triage": reasoning.get("triage"),
        "triage_rationale": reasoning.get("triage_rationale"),
        "impression": reasoning.get("impression"),
        "differential": reasoning.get("differential"),
        "recommendations": reasoning.get("recommendations"),
        "visual_findings": visual.get("visual_findings"),
        "image_description": visual.get("image_description"),
        "image_note_correlation": visual.get("image_note_correlation")
        or reasoning.get("image_text_correlation"),
        "follow_up_questions": reasoning.get("follow_up_questions") or [],
        "conversation": record.get("conversation") or [],
        "evaluation": record.get("evaluation") or {},
    }


def process_follow_up(client, user_message, llama_model, conversation_path=None):
    """
    Append a clinician follow-up, re-triage with Llama, and rewrite conversation.json.

    Parameters
    ----------
    conversation_path : str or None
        File to load/update; defaults to ``CONVERSATION_PATH``.

    Returns
    -------
    dict
        ``display_payload`` fields plus ``latest_reply`` for the new assistant turn.
    """
    user_message = (user_message or "").strip()
    if not user_message:
        raise ValueError("Empty message")
    path = conversation_path or CONVERSATION_PATH
    record = load_conversation(path)
    result = follow_up_with_llama(record, user_message, client, llama_model)
    record["clinical_reasoning"] = result["updated_reasoning"]
    record["conversation"] = (record.get("conversation") or []) + result["conversation_append"]
    record.pop("references", None)
    record["evaluation"] = evaluate_record(record)
    save_json(path, record)
    payload = display_payload(record)
    payload["latest_reply"] = result["assistant_reply"]
    return payload
