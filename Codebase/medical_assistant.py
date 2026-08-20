"""
medical_assistant.py
--------------------
Run cases through LangGraph and manage upload sessions.
"""

import json
import os
import time

from clinical_references import attach_references
from evaluation import evaluate_record
from graph_pipeline import UnsupportedImageDomainError, build_graph, follow_up_with_llama
from ingestion import IMAGE_DOMAINS, UPLOAD_ROOT, prepare_existing_paths
from paths import SAMPLE_DIR


_GRAPH = None


def get_graph(client):
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph(client)
    return _GRAPH


def read_text(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def _resolve_inputs(text_path, image_path):
    """Load note text; de-identify only when needed."""
    text_norm = os.path.normpath(text_path)
    image_norm = os.path.normpath(image_path)
    upload_root = os.path.normpath(UPLOAD_ROOT)
    sample_root = os.path.normpath(SAMPLE_DIR)

    if text_norm.startswith(upload_root) and image_norm.startswith(upload_root):
        return read_text(text_path), text_path, image_path, {
            "text": {"path": text_path, "privacy": {"already_deidentified": True}},
            "image": {"path": image_path, "privacy": {"already_deidentified": True}},
        }

    if text_norm.startswith(sample_root) and image_norm.startswith(sample_root):
        return read_text(text_path), text_path, image_path, {
            "text": {"path": text_path, "privacy": {"teaching_case": True}},
            "image": {"path": image_path, "privacy": {"teaching_case": True}},
        }

    ingested = prepare_existing_paths(text_path, image_path)
    return (
        read_text(ingested["text"]["path"]),
        ingested["text"]["path"],
        ingested["image"]["path"],
        ingested,
    )


def process_case(client, vision_model, llama_model, case_id, image_path, text_path):
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
        "references": state.get("references") or attach_references(
            state, client=client, llama_model=llama_model
        ),
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
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _valid_session_id(session_id):
    return (
        isinstance(session_id, str)
        and session_id
        and session_id == os.path.basename(session_id)
        and all(ch.isalnum() or ch in "-_" for ch in session_id)
    )


def session_dir_from_id(session_id):
    if not _valid_session_id(session_id):
        raise ValueError("Invalid session id")
    path = os.path.join(UPLOAD_ROOT, session_id)
    if not os.path.isdir(path):
        raise FileNotFoundError("Session not found: {}".format(session_id))
    return path


def load_session_record(session_id):
    path = os.path.join(session_dir_from_id(session_id), "conversation.json")
    if not os.path.exists(path):
        raise FileNotFoundError("No conversation for session {}".format(session_id))
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dashboard_payload(record, session_id):
    visual = record.get("visual_analysis") or {}
    reasoning = record.get("clinical_reasoning") or {}
    return {
        "session_id": session_id,
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
        "references": record.get("references") or [],
        "follow_up_questions": reasoning.get("follow_up_questions") or [],
        "conversation": record.get("conversation") or [],
        "evaluation": record.get("evaluation") or {},
        "image_url": "/session/{}/image".format(session_id),
    }


def process_follow_up(client, session_id, user_message, llama_model):
    user_message = (user_message or "").strip()
    if not user_message:
        raise ValueError("Empty message")
    record = load_session_record(session_id)
    result = follow_up_with_llama(record, user_message, client, llama_model)
    record["clinical_reasoning"] = result["updated_reasoning"]
    record["conversation"] = (record.get("conversation") or []) + result["conversation_append"]
    record["references"] = result.get("references") or attach_references(
        record, client=client, llama_model=llama_model
    )
    record["evaluation"] = evaluate_record(record)
    save_json(os.path.join(session_dir_from_id(session_id), "conversation.json"), record)
    payload = dashboard_payload(record, session_id)
    payload["latest_reply"] = result["assistant_reply"]
    return payload


