"""
graph_pipeline.py
-----------------
LangGraph orchestration for one multimodal clinical case.

Node sequence
-------------
1. ``medgemma_analyze`` — vision LLM: domain + image findings + image–note link
2. ``llama_entities`` — text LLM: structured facts from the note
3. ``llama_reason`` — text LLM: triage JSON (impression, differential, recommendations)
4. ``llama_converse`` — text LLM: clinician-facing summary turn

Follow-up chat (outside the compiled graph) uses ``follow_up_with_llama``, which
re-triages and appends conversation turns.

Reuse
-----
::

    from graph_pipeline import build_graph, follow_up_with_llama
    graph = build_graph(ollama_client)
    state = graph.invoke({
        "image_path": "...", "note": "...",
        "vision_model": "medgemma:4b", "llama_model": "llama3.2:3b",
    })
"""

import json
import re
from os.path import basename
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from ollama_client import extract_json
from runtime_profile import get_profile


class UnsupportedImageDomainError(ValueError):
    """Raised when an image is not radiology or pathology."""


RADIOLOGY_MODALITIES = {"chest_xray", "brain_ct", "bone_xray", "mri", "ct", "xray", "x-ray", "radiograph"}
PATHOLOGY_MODALITIES = {"pathology_slide", "histopathology", "histology", "biopsy_slide"}


class CaseState(TypedDict, total=False):
    """Shared LangGraph state keys passed between nodes."""

    image_path: str
    note: str
    vision_model: str
    llama_model: str
    image_domain: str
    visual_analysis: Dict[str, Any]
    extracted_entities: Dict[str, Any]
    clinical_reasoning: Dict[str, Any]
    conversation: List[Dict[str, str]]


ENTITY_KEYS = (
    "age_sex",
    "chief_complaint",
    "history",
    "vitals",
    "exam",
    "medications",
    "requested_study",
    "clinical_question",
)
ENTITY_KEYS_SHORT = ("age_sex", "chief_complaint", "vitals", "requested_study", "clinical_question")

MEDGEMMA_SYSTEM = (
    "You are a specialist vision assistant for radiology imaging and histopathology. "
    "Use only what is visible in the image and written in the note. "
    "Output compact JSON only."
)
LLAMA_SYSTEM = (
    "You are a clinical documentation and reasoning assistant for radiology and pathology. "
    "Use only the supplied facts. Do not invent imaging or histology signs. "
    "Write clear, structured, clinician-facing prose."
)
LLAMA_REASONING_SYSTEM = (
    "You are a clinical reasoning assistant for radiology and pathology workflows. "
    "Triage: emergency = stroke/airway/shock/bleeding; urgent = hypoxia/pneumonia/fracture; "
    "soon = stable histopathology without sepsis; routine = none of these. "
    "Output compact JSON only."
)


def build_facts(visual, entities, reasoning, max_findings=6):
    """Shared context block for Llama dialogue steps."""
    return {
        "modality": visual.get("inferred_modality"),
        "visual_findings": (visual.get("visual_findings") or [])[:max_findings],
        "image_description": visual.get("image_description"),
        "image_note_correlation": visual.get("image_note_correlation"),
        "entities": {key: entities.get(key) for key in ENTITY_KEYS_SHORT},
        "impression": reasoning.get("impression"),
        "differential": reasoning.get("differential"),
        "triage": reasoning.get("triage"),
        "recommendations": reasoning.get("recommendations"),
        "correlation": reasoning.get("image_text_correlation"),
    }


def resolve_detected_domain(visual):
    """Map MedGemma output to radiology or pathology; reject other domains."""
    domain = str(visual.get("image_domain") or "").lower().strip()
    modality = str(visual.get("inferred_modality") or "").lower().strip()
    description = str(visual.get("image_description") or "").lower()

    if domain in {"radiology", "pathology"}:
        return domain
    if domain == "unsupported":
        raise UnsupportedImageDomainError(
            "This assistant accepts radiology scans and histopathology slides only. "
            "The uploaded image does not appear to belong to either domain."
        )
    if modality in PATHOLOGY_MODALITIES or "pathology" in modality or "histolog" in modality:
        return "pathology"
    if modality in RADIOLOGY_MODALITIES or any(
        token in modality for token in ("xray", "x-ray", "radiograph", "_ct", " mri")
    ):
        return "radiology"
    if any(token in description for token in ("histopathology", "h&e", "microscope slide", "biopsy slide")):
        return "pathology"
    if any(token in description for token in ("x-ray", "xray", "radiograph", " ct ", " mri ", "chest film")):
        return "radiology"
    raise UnsupportedImageDomainError(
        "Could not classify the image as radiology or pathology. "
        "Upload a radiology scan (X-ray, CT, MRI) or a histopathology slide only."
    )


def analyze_with_medgemma(state, client):
    """MedGemma classifies the image domain and inspects the image and note."""
    prompt = (
        "This clinical case contains one image plus a patient note. "
        "The system accepts only radiology scans or histopathology slides. "
        "Return JSON only with keys: "
        "image_domain (radiology|pathology|unsupported), "
        "inferred_modality (chest_xray|brain_ct|bone_xray|mri|pathology_slide|unknown), "
        "image_description, visual_findings, image_note_correlation, likely_conditions, "
        "uncertainties, image_is_synthetic_or_schematic. "
        "Set image_domain to unsupported for dermatology, ophthalmology, ultrasound-only, "
        "clinical photos, or any non-radiology/non-pathology image.\n\nNOTE:\n{}".format(state["note"])
    )
    print("  MedGemma: detecting domain + reading image...", flush=True)
    profile = get_profile()
    raw = client.chat(
        model=state["vision_model"],
        prompt=prompt,
        system=MEDGEMMA_SYSTEM,
        images=[state["image_path"]],
        temperature=0.1,
        max_tokens=280,
        keep_alive=profile.vision_keep_alive,
        json_mode=True,
    )
    visual = normalize_visual(extract_json(raw), state)
    detected_domain = resolve_detected_domain(visual)
    visual["image_domain"] = detected_domain
    print("  Detected image domain:", detected_domain, flush=True)
    return {"visual_analysis": visual, "image_domain": detected_domain}


def extract_entities_with_llama(state, client):
    """Llama extracts structured facts from the note."""
    prompt = (
        "Extract key facts from this patient note. Return JSON only with keys: "
        "age_sex, chief_complaint, history, vitals, exam, medications, "
        "requested_study, clinical_question.\n\nNOTE:\n{}".format(state["note"])
    )
    print("  Llama: extracting entities...", flush=True)
    raw = client.chat(
        model=state["llama_model"],
        prompt=prompt,
        system=LLAMA_SYSTEM,
        temperature=0.1,
        max_tokens=280,
        keep_alive=get_profile().llama_keep_alive,
        json_mode=True,
    )
    return {"extracted_entities": normalize_entities(extract_json(raw), state["note"])}


def reason_with_llama(state, client):
    """Llama writes triage JSON from MedGemma evidence."""
    visual = state.get("visual_analysis") or {}
    entities = state.get("extracted_entities") or {}
    evidence = build_facts(visual, entities, {})
    evidence["likely_conditions"] = (visual.get("likely_conditions") or [])[:6]
    evidence["entities"] = {key: entities.get(key) for key in ENTITY_KEYS}
    prompt = (
        "Fuse image findings with the patient note. Return JSON only with keys: "
        "impression, differential, triage (emergency|urgent|soon|routine), "
        "triage_rationale, recommendations, follow_up_questions, "
        "image_text_correlation, safety_flags.\n\n"
        "EVIDENCE:\n{}\n\nFULL_NOTE:\n{}\n".format(json.dumps(evidence, indent=2), state["note"])
    )
    print("  Llama: writing triage JSON...", flush=True)
    raw = client.chat(
        model=state["llama_model"],
        prompt=prompt,
        system=LLAMA_REASONING_SYSTEM,
        temperature=0.2,
        max_tokens=520,
        keep_alive=get_profile().llama_keep_alive,
        json_mode=True,
    )
    reasoning = normalize_reasoning(extract_json(raw), visual)
    conditions = visual.get("likely_conditions") or []
    if conditions and not reasoning.get("differential"):
        reasoning["differential"] = conditions[:4]
    reasoning = calibrate_triage(reasoning, state["note"])
    return {"clinical_reasoning": reasoning}


def converse_with_llama(state, client):
    """Produce the initial clinician-facing summary (follow-ups use the TUI)."""
    visual = state.get("visual_analysis") or {}
    entities = state.get("extracted_entities") or {}
    reasoning = state.get("clinical_reasoning") or {}
    facts = json.dumps(build_facts(visual, entities, reasoning), indent=2)
    print("  Llama: writing summary...", flush=True)
    summary = llama_turn(
        client,
        state["llama_model"],
        (
            "Write one clinician-facing reply that covers: (1) patient presentation, "
            "(2) image findings correlated with the note, (3) triage category with rationale, "
            "(4) recommended next steps, (5) two brief follow-up questions. "
            "Ground every claim in the supplied image findings and note facts."
        ),
        facts,
        keep_alive=get_profile().llama_keep_alive,
    )
    return {
        "conversation": [
            {
                "role": "clinician",
                "content": "Uploaded {} with clinical note for review.".format(
                    visual.get("file_name", "medical image")
                ),
            },
            {"role": "assistant", "content": summary},
        ],
    }


def _chat_history(record, limit=6):
    return "\n".join(
        "{}: {}".format(turn.get("role", "user"), str(turn.get("content", ""))[:320])
        for turn in (record.get("conversation") or [])[-limit:]
    )


def refresh_reasoning_after_follow_up(record, user_message, client, llama_model):
    """Re-run Llama triage JSON using note, image evidence, and follow-up context."""
    visual = record.get("visual_analysis") or {}
    entities = record.get("extracted_entities") or {}
    prior = record.get("clinical_reasoning") or {}
    note = record.get("note_text") or ""
    history = _chat_history(record)
    combined_context = note
    if history:
        combined_context += "\n\nRECENT CHAT:\n" + history
    combined_context += "\n\nCLINICIAN FOLLOW-UP:\n" + user_message.strip()

    evidence = build_facts(visual, entities, prior, max_findings=8)
    evidence["likely_conditions"] = (visual.get("likely_conditions") or [])[:6]
    evidence["entities"] = {key: entities.get(key) for key in ENTITY_KEYS}
    evidence["prior_reasoning"] = {
        "impression": prior.get("impression"),
        "differential": prior.get("differential"),
        "triage": prior.get("triage"),
        "recommendations": prior.get("recommendations"),
    }
    prompt = (
        "Re-assess this clinical case after a clinician follow-up. Update impression, differential, "
        "triage, recommendations, and suggest exactly two new adaptive follow-up questions "
        "based on the latest chat. Return JSON only with keys: impression, differential, "
        "triage (emergency|urgent|soon|routine), triage_rationale, recommendations, "
        "follow_up_questions, image_text_correlation, safety_flags.\n\n"
        "EVIDENCE:\n{}\n\nNOTE_AND_FOLLOWUP:\n{}\n"
    ).format(json.dumps(evidence, indent=2), combined_context)
    print("  Llama: re-triaging after follow-up...", flush=True)
    raw = client.chat(
        model=llama_model,
        prompt=prompt,
        system=LLAMA_REASONING_SYSTEM,
        temperature=0.2,
        max_tokens=520,
        keep_alive=get_profile().llama_keep_alive,
        json_mode=True,
    )
    reasoning = normalize_reasoning(extract_json(raw), visual)
    conditions = visual.get("likely_conditions") or []
    if conditions and not reasoning.get("differential"):
        reasoning["differential"] = conditions[:4]
    reasoning = calibrate_triage(reasoning, combined_context)
    previous = str(prior.get("triage", "")).lower()
    if str(reasoning.get("triage", "")).lower() != previous and previous:
        reasoning["triage_rationale"] = (
            str(reasoning.get("triage_rationale", "")) + " Updated after follow-up."
        ).strip()
    return reasoning


def llama_turn(client, model, instruction, context, keep_alive=None, word_range="120-180"):
    """Ask Llama for one clinician-facing reply grounded in case facts."""
    if keep_alive is None:
        keep_alive = get_profile().llama_keep_alive
    prompt = (
        "{}\n\nFacts:\n{}\n\nWrite {} words for a clinician. "
        "Cite image findings and note details explicitly; do not invent external sources."
    ).format(instruction, context, word_range).strip()
    text = client.chat(
        model=model,
        prompt=prompt,
        system=LLAMA_SYSTEM,
        temperature=0.3,
        max_tokens=380,
        keep_alive=keep_alive,
    )
    return text.strip()


def follow_up_with_llama(record, user_message, client, llama_model):
    """Answer follow-up and re-triage with Llama using image + note evidence."""
    visual = record.get("visual_analysis") or {}
    entities = record.get("extracted_entities") or {}
    reasoning = refresh_reasoning_after_follow_up(record, user_message, client, llama_model)

    history = _chat_history(record, limit=6)
    facts = json.dumps(build_facts(visual, entities, reasoning, max_findings=8), indent=2)
    print("  Llama: follow-up reply...", flush=True)
    prompt = (
        "Answer the clinician follow-up using image findings, updated triage, and note context. "
        "Reference specific visual findings and note details.\n\n"
        "Updated facts:\n{}\n\nPrior chat:\n{}\n\nQuestion:\n{}\n\n"
        "Write 100-150 words for a clinician."
    ).format(facts, history, user_message.strip())
    reply = client.chat(
        model=llama_model,
        prompt=prompt,
        system=LLAMA_SYSTEM,
        temperature=0.3,
        max_tokens=300,
        keep_alive=get_profile().llama_keep_alive,
    )
    return {
        "assistant_reply": reply.strip(),
        "updated_reasoning": reasoning,
        "conversation_append": [
            {"role": "clinician", "content": user_message.strip()},
            {"role": "assistant", "content": reply.strip()},
        ],
    }


def normalize_visual(parsed, state):
    """Ensure visual analysis has required keys."""
    base = {
        "file_name": basename(state["image_path"]),
        "image_domain": "unknown",
        "inferred_modality": "unknown",
        "image_description": "Image description unavailable.",
        "visual_findings": ["See patient note."],
        "likely_conditions": [],
        "image_note_correlation": "Correlate image with the requested study in the note.",
        "uncertainties": [],
        "image_is_synthetic_or_schematic": False,
    }
    if isinstance(parsed, dict):
        for key in base:
            if parsed.get(key) not in (None, "", []):
                base[key] = parsed[key]
        if isinstance(base["visual_findings"], str):
            base["visual_findings"] = [base["visual_findings"]]
        if isinstance(base["likely_conditions"], str):
            base["likely_conditions"] = [base["likely_conditions"]]
    return base


def normalize_entities(parsed, note):
    """Ensure entity record has required keys."""
    base = {key: "see note" for key in ENTITY_KEYS}
    base["chief_complaint"] = note.split("\n")[2] if "\n" in note else note[:200]
    if isinstance(parsed, dict):
        for key in ENTITY_KEYS:
            if parsed.get(key):
                base[key] = parsed[key]
    return base


def _as_string_list(value, fallback=None):
    """Coerce LLM list fields that sometimes arrive as a plain string."""
    if value is None or value == "":
        return list(fallback or [])
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return list(fallback or [])
        # Split multi-sentence blobs into bullet-sized lines when possible.
        parts = [p.strip(" -•\t") for p in re.split(r"[\n;]+|(?<=\.)\s+(?=[A-Z])", text) if p.strip()]
        return parts if parts else [text]
    return [str(value)]


def normalize_reasoning(parsed, visual):
    """Ensure reasoning record has required keys; list fields stay lists."""
    base = {
        "impression": "Multimodal clinical summary pending review.",
        "differential": (visual.get("likely_conditions") or [])[:4],
        "triage": "urgent",
        "triage_rationale": "Based on image findings and presenting complaint.",
        "recommendations": ["Clinician review of image and note."],
        "follow_up_questions": [
            "When did symptoms start and are they worsening?",
            "Any red-flag features (hypoxia, neurologic deficit, bleeding)?",
        ],
        "image_text_correlation": visual.get("image_note_correlation", ""),
        "safety_flags": [],
    }
    if isinstance(parsed, dict):
        for key in base:
            if parsed.get(key):
                base[key] = parsed[key]
    list_keys = ("differential", "recommendations", "follow_up_questions", "safety_flags")
    for key in list_keys:
        base[key] = _as_string_list(base.get(key), fallback=base.get(key) if isinstance(base.get(key), list) else None)
    if not base["differential"]:
        base["differential"] = _as_string_list(visual.get("likely_conditions"), fallback=[])[:4]
    if not base["recommendations"]:
        base["recommendations"] = ["Clinician review of image and note."]
    return base


def calibrate_triage(reasoning, note):
    """Adjust triage using explicit red-flag phrases in the note or follow-up."""
    text = note.lower()
    current = str(reasoning.get("triage", "urgent")).lower().strip()
    if any(cue in text for cue in ("hemiparesis", "aphasia", "speech difficulty", "sudden weakness")):
        suggested = "emergency"
    elif any(cue in text for cue in ("dyspnoea", "dyspnea", "spo2", "pneumonia", "fever")):
        suggested = "urgent"
    elif any(cue in text for cue in ("fracture", "foosh", "wrist")):
        suggested = "urgent"
    elif any(cue in text for cue in ("lymphoma", "hodgkin", "carcinoma", "histology", "biopsy")):
        suggested = "soon"
    else:
        suggested = current if current in {"emergency", "urgent", "soon", "routine"} else "urgent"
    reasoning = dict(reasoning)
    reasoning["triage"] = suggested
    return reasoning


def build_graph(client):
    """
    Compile MedGemma → Llama entity → triage → summary as a LangGraph.

    Parameters
    ----------
    client : OllamaClient
        Shared HTTP client closed over by each node.

    Returns
    -------
    CompiledStateGraph
        Call ``.invoke(state_dict)`` with image_path, note, vision_model, llama_model.
    """
    graph = StateGraph(CaseState)
    graph.add_node("medgemma_analyze", lambda state: analyze_with_medgemma(state, client))
    graph.add_node("llama_entities", lambda state: extract_entities_with_llama(state, client))
    graph.add_node("llama_reason", lambda state: reason_with_llama(state, client))
    graph.add_node("llama_converse", lambda state: converse_with_llama(state, client))
    graph.add_edge(START, "medgemma_analyze")
    graph.add_edge("medgemma_analyze", "llama_entities")
    graph.add_edge("llama_entities", "llama_reason")
    graph.add_edge("llama_reason", "llama_converse")
    graph.add_edge("llama_converse", END)
    return graph.compile()
