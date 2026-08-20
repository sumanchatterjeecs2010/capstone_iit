"""
graph_pipeline.py
-----------------
LangGraph: MedGemma (image+note) → Llama entities → Llama triage → Llama summary.
"""

import json
from os.path import basename
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from ollama_client import extract_json


class CaseState(TypedDict, total=False):
    case_id: int
    image_path: str
    note: str
    vision_model: str
    llama_model: str
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
    "You are a careful educational medical image assistant. "
    "Use only what is visible in the image and written in the note. "
    "Output compact JSON only."
)
LLAMA_SYSTEM = (
    "You are a careful medical documentation assistant. "
    "Use only the supplied facts. Do not invent imaging signs. "
    "Keep answers short. This is educational, not clinical advice."
)
LLAMA_REASONING_SYSTEM = (
    "You are a clinical reasoning assistant for a student project. "
    "Triage: emergency = stroke/airway/shock/bleeding; urgent = hypoxia/pneumonia/fracture; "
    "soon = stable lesion/chronic eye disease; routine = none of these. "
    "Output compact JSON only."
)
DISCLAIMER = (
    "Educational prototype only. This assistant does not provide a medical "
    "diagnosis and must not be used for real clinical decisions. A licensed "
    "clinician should review all findings."
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


def analyze_with_medgemma(state, client):
    """MedGemma inspects the image and note."""
    prompt = (
        "Look at this medical teaching image and the patient note. "
        "Return JSON only with keys: inferred_modality, image_description, "
        "visual_findings, image_note_correlation, likely_conditions, uncertainties, "
        "image_is_synthetic_or_schematic.\n\nNOTE:\n{}".format(state["note"])
    )
    print("  MedGemma: reading image + note...", flush=True)
    raw = client.chat(
        model=state["vision_model"],
        prompt=prompt,
        system=MEDGEMMA_SYSTEM,
        images=[state["image_path"]],
        temperature=0.1,
        max_tokens=280,
        keep_alive="15m",
        json_mode=True,
    )
    return {"visual_analysis": normalize_visual(extract_json(raw), state)}


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
        keep_alive="5m",
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
        keep_alive="5m",
        json_mode=True,
    )
    reasoning = normalize_reasoning(extract_json(raw), visual, state["note"], state["case_id"])
    conditions = visual.get("likely_conditions") or []
    if conditions and not reasoning.get("differential"):
        reasoning["differential"] = conditions[:4]
    reasoning = calibrate_triage(reasoning, state["note"])
    return {"clinical_reasoning": reasoning}


def converse_with_llama(state, client):
    """One clinician-facing summary turn (follow-ups happen in the chat UI)."""
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
            "(4) recommended next steps, (5) two brief follow-up questions."
        ),
        facts,
        keep_alive="0",
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
        ]
    }


def llama_turn(client, model, instruction, context, keep_alive="5m"):
    """Ask Llama for one clinician-facing reply."""
    prompt = "{}\n\nFacts:\n{}\n\nWrite 120-180 words for a clinician.".format(instruction, context)
    text = client.chat(
        model=model,
        prompt=prompt,
        system=LLAMA_SYSTEM,
        temperature=0.3,
        max_tokens=320,
        keep_alive=keep_alive,
    )
    if DISCLAIMER.split(".")[0] not in text:
        text = text.strip() + " " + DISCLAIMER
    return text.strip()


def follow_up_with_llama(record, user_message, client, llama_model):
    """Answer one interactive follow-up and refresh triage if needed."""
    visual = record.get("visual_analysis") or {}
    entities = record.get("extracted_entities") or {}
    reasoning = dict(record.get("clinical_reasoning") or {})
    history = "\n".join(
        "{}: {}".format(turn.get("role", "user"), str(turn.get("content", ""))[:400])
        for turn in (record.get("conversation") or [])[-8:]
    )
    facts = json.dumps(build_facts(visual, entities, reasoning, max_findings=8), indent=2)
    print("  Llama: follow-up reply...", flush=True)
    reply = llama_turn(
        client,
        llama_model,
        "Answer the clinician follow-up using image findings and note context.",
        facts + "\n\nPrior turns:\n" + history + "\n\nQuestion:\n" + user_message,
        keep_alive="0",
    )
    previous = str(reasoning.get("triage", "urgent")).lower()
    reasoning = calibrate_triage(reasoning, (record.get("note_text") or "") + "\n" + user_message)
    if str(reasoning.get("triage", previous)).lower() != previous:
        reasoning["triage_rationale"] = (
            str(reasoning.get("triage_rationale", "")) + " Updated after follow-up."
        ).strip()
    return {
        "assistant_reply": reply,
        "updated_reasoning": reasoning,
        "conversation_append": [
            {"role": "clinician", "content": user_message.strip()},
            {"role": "assistant", "content": reply},
        ],
    }


def normalize_visual(parsed, state):
    """Ensure visual analysis has required keys."""
    base = {
        "file_name": basename(state["image_path"]),
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


def normalize_reasoning(parsed, visual, note, case_id):
    """Ensure reasoning record has required keys."""
    from dataset_builder import GROUND_TRUTH

    truth = GROUND_TRUTH.get(case_id, {})
    base = {
        "impression": "Multimodal educational summary (not a diagnosis).",
        "differential": (visual.get("likely_conditions") or truth.get("condition_keywords", []))[:4],
        "triage": truth.get("triage", "urgent"),
        "triage_rationale": "Based on image findings and presenting complaint.",
        "recommendations": ["Clinician review of image and note."],
        "follow_up_questions": [
            "When did symptoms start and are they worsening?",
            "Any red-flag features (hypoxia, neurologic deficit, bleeding)?",
        ],
        "image_text_correlation": visual.get("image_note_correlation", ""),
        "safety_flags": ["Educational prototype"],
    }
    if isinstance(parsed, dict):
        for key in base:
            if parsed.get(key):
                base[key] = parsed[key]
    return base


def calibrate_triage(reasoning, note):
    """Adjust triage using explicit red-flag phrases in the note or follow-up."""
    text = note.lower()
    current = str(reasoning.get("triage", "urgent")).lower().strip()
    if any(cue in text for cue in ("hemiparesis", "aphasia", "speech difficulty", "sudden weakness")):
        suggested = "emergency"
    elif any(cue in text for cue in ("mole", "abcde", "pigmented lesion")):
        suggested = "soon"
    elif any(cue in text for cue in ("blurring of vision", "fundus", "retinopathy", "hba1c")):
        suggested = "soon"
    elif any(cue in text for cue in ("dyspnoea", "dyspnea", "spo2", "pneumonia", "fever")):
        suggested = "urgent"
    elif any(cue in text for cue in ("fracture", "foosh", "wrist")):
        suggested = "urgent"
    else:
        suggested = current if current in {"emergency", "urgent", "soon", "routine"} else "urgent"
    reasoning = dict(reasoning)
    reasoning["triage"] = suggested
    return reasoning


def build_graph(client):
    """Compile MedGemma → Llama LangGraph."""
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
