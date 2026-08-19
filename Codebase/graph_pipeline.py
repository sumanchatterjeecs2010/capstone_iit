"""
graph_pipeline.py
-----------------
LangGraph orchestration for the two-LLM assistant.

Graph order
1. MedGemma 1.5 4B (vision LLM) reads the image and the patient note.
2. Llama 3.2 extracts clinical entities from the note.
3. Llama 3.2 writes structured triage JSON from MedGemma findings + note.
4. Llama 3.2 writes the clinician-facing conversation.

LangGraph is the orchestrator, not a model.
"""

import json
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from ollama_client import extract_json


class CaseState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes for one patient case."""

    case_id: int
    image_path: str
    note: str
    vision_model: str
    llama_model: str
    visual_analysis: Dict[str, Any]
    extracted_entities: Dict[str, Any]
    clinical_reasoning: Dict[str, Any]
    conversation: List[Dict[str, str]]
    vision_raw: str
    entity_raw: str
    reason_raw: str


MEDGEMMA_SYSTEM = (
    "You are a careful educational medical image assistant. "
    "Use only what is visible in the image and written in the note. "
    "Do not invent anatomy that is not supported. "
    "These may be public teaching scans or photographs. "
    "Describe only what is visible. "
    "Output compact JSON only."
)

LLAMA_SYSTEM = (
    "You are a careful medical documentation assistant. "
    "Use only the supplied facts. Do not invent imaging signs. "
    "Do not claim the image shows physical-exam findings such as crepitations. "
    "Keep answers short. Always remind the user that this is educational, not clinical advice."
)

LLAMA_REASONING_SYSTEM = (
    "You are a clinical reasoning assistant for a student project. "
    "Use MedGemma image-note findings together with the patient note. "
    "Triage rubric: emergency = stroke, airway, shock or active haemorrhage; "
    "urgent = hypoxia, suspected pneumonia or fracture with stable vitals; "
    "soon = stable skin lesion or chronic diabetic eye disease; "
    "routine = none of these. Never claim certainty. Output compact JSON only."
)

DISCLAIMER = (
    "Educational prototype only. This assistant does not provide a medical "
    "diagnosis and must not be used for real clinical decisions. A licensed "
    "clinician should review all findings."
)


def _chat(client, **kwargs):
    """Call Ollama chat and return raw text."""
    return client.chat(**kwargs)


def analyze_with_medgemma(state, client):
    """LangGraph node: MedGemma inspects the image and the note."""
    prompt = (
        "Look at this medical teaching image and the patient note. "
        "Return JSON only with keys: "
        "inferred_modality (chest_xray|dermatology_photo|brain_ct|fundus|bone_xray|unknown), "
        "image_description, visual_findings (list of short strings), "
        "image_note_correlation, likely_conditions (list), "
        "uncertainties (list), image_is_synthetic_or_schematic (boolean).\n\n"
        "NOTE:\n{}".format(state["note"])
    )
    print("  MedGemma: reading image + note...", flush=True)
    raw = _chat(
        client,
        model=state["vision_model"],
        prompt=prompt,
        system=MEDGEMMA_SYSTEM,
        images=[state["image_path"]],
        temperature=0.1,
        max_tokens=280,
        keep_alive="15m",
        json_mode=True,
    )
    parsed = extract_json(raw)
    visual = normalize_visual(parsed, state)
    visual["raw_model_output"] = raw
    visual["vision_llm"] = state["vision_model"]
    return {"visual_analysis": visual, "vision_raw": raw}


def extract_entities_with_llama(state, client):
    """LangGraph node: Llama extracts structured facts from the note."""
    prompt = (
        "Extract key facts from this patient note. Return JSON only with keys: "
        "age_sex, chief_complaint, history, vitals, exam, medications, "
        "requested_study, clinical_question.\n\nNOTE:\n{}".format(state["note"])
    )
    print("  Llama: extracting entities...", flush=True)
    raw = _chat(
        client,
        model=state["llama_model"],
        prompt=prompt,
        system=LLAMA_SYSTEM,
        temperature=0.1,
        max_tokens=280,
        keep_alive="5m",
        json_mode=True,
    )
    entities = normalize_entities(extract_json(raw), state["note"])
    entities["raw_model_output"] = raw
    return {"extracted_entities": entities, "entity_raw": raw}


def reason_with_llama(state, client):
    """LangGraph node: Llama writes triage JSON from MedGemma evidence."""
    visual = state.get("visual_analysis") or {}
    entities = state.get("extracted_entities") or {}
    evidence = {
        "modality": visual.get("inferred_modality"),
        "image_description": visual.get("image_description"),
        "visual_findings": visual.get("visual_findings", [])[:6],
        "likely_conditions": visual.get("likely_conditions", [])[:6],
        "image_note_correlation": visual.get("image_note_correlation"),
        "image_is_synthetic_or_schematic": visual.get("image_is_synthetic_or_schematic"),
        "entities": {
            key: entities.get(key)
            for key in (
                "age_sex",
                "chief_complaint",
                "history",
                "vitals",
                "exam",
                "requested_study",
                "clinical_question",
            )
        },
    }
    prompt = (
        "Fuse the MedGemma image-note findings with the patient note. "
        "Return JSON only with keys: "
        "impression, differential (list of clinical conditions), "
        "triage (emergency|urgent|soon|routine), "
        "triage_rationale, recommendations (list), follow_up_questions (list), "
        "image_text_correlation, safety_flags (list).\n"
        "Use visual_findings as image evidence. Do not treat exam signs as if they were on the image.\n\n"
        "EVIDENCE:\n{}\n\nFULL_NOTE:\n{}\n".format(json.dumps(evidence, indent=2), state["note"])
    )
    print("  Llama: writing triage JSON...", flush=True)
    raw = _chat(
        client,
        model=state["llama_model"],
        prompt=prompt,
        system=LLAMA_REASONING_SYSTEM,
        temperature=0.2,
        max_tokens=520,
        keep_alive="5m",
        json_mode=True,
    )
    reasoning = normalize_reasoning(
        extract_json(raw),
        visual,
        state["note"],
        state["case_id"],
    )
    reasoning["raw_model_output"] = raw
    conditions = visual.get("likely_conditions") or []
    existing = reasoning.get("differential")
    if conditions and (
        not isinstance(existing, list)
        or not existing
        or all(str(item).lower() in {"see note", "unknown"} for item in existing)
    ):
        reasoning["differential"] = conditions[:4]
    reasoning = calibrate_triage(reasoning, state["note"])
    return {"clinical_reasoning": reasoning, "reason_raw": raw}


def converse_with_llama(state, client):
    """LangGraph node: Llama writes scripted clinician-assistant turns."""
    visual = state.get("visual_analysis") or {}
    entities = state.get("extracted_entities") or {}
    reasoning = state.get("clinical_reasoning") or {}
    fact_block = json.dumps(
        {
            "modality": visual.get("inferred_modality"),
            "visual_findings": visual.get("visual_findings", [])[:6],
            "image_description": visual.get("image_description"),
            "entities": {
                key: entities.get(key)
                for key in (
                    "age_sex",
                    "chief_complaint",
                    "vitals",
                    "requested_study",
                    "clinical_question",
                )
            },
            "impression": reasoning.get("impression"),
            "differential": reasoning.get("differential"),
            "triage": reasoning.get("triage"),
            "recommendations": reasoning.get("recommendations"),
            "correlation": reasoning.get("image_text_correlation"),
        },
        indent=2,
    )
    print("  Llama: writing conversation...", flush=True)
    turns = [
        {
            "role": "clinician",
            "content": (
                "I am uploading a medical image ({}) with this patient note:\n{}".format(
                    visual.get("file_name", "image"),
                    state["note"],
                )
            ),
        }
    ]
    turns.append(
        {
            "role": "assistant",
            "content": llama_turn(
                client,
                state["llama_model"],
                "Summarise the patient presentation and state that MedGemma reviewed the image with the note.",
                fact_block,
            ),
        }
    )
    turns.append(
        {
            "role": "clinician",
            "content": "Interpret this image and correlate it with the symptoms and history.",
        }
    )
    turns.append(
        {
            "role": "assistant",
            "content": llama_turn(
                client,
                state["llama_model"],
                "Provide a cautious multimodal interpretation using MedGemma visual_findings for the image and the note for symptoms. Do not put exam findings onto the image.",
                fact_block,
            ),
        }
    )
    turns.append(
        {
            "role": "clinician",
            "content": "What triage category do you suggest, and what should we do next?",
        }
    )
    turns.append(
        {
            "role": "assistant",
            "content": llama_turn(
                client,
                state["llama_model"],
                "Give triage (emergency/urgent/soon/routine), rationale, next steps, and two follow-up questions.",
                fact_block,
            ),
        }
    )
    questions = reasoning.get("follow_up_questions") or []
    if questions:
        turns.append({"role": "clinician", "content": str(questions[0])})
        turns.append(
            {
                "role": "assistant",
                "content": llama_turn(
                    client,
                    state["llama_model"],
                    "Answer the clinician follow-up using only the known facts. If unknown, say so and advise clinical assessment.",
                    fact_block + "\nFollow-up question: {}".format(questions[0]),
                    keep_alive="0",
                ),
            }
        )
    return {"conversation": turns}


def llama_turn(client, model, instruction, context, keep_alive="5m"):
    """Ask Llama to write one clinician-facing assistant reply."""
    prompt = (
        "{}\n\nUse only these facts:\n{}\n\n"
        "Write 80-120 words, plain language for a clinician. "
        "End with the educational disclaimer sentence.".format(instruction, context)
    )
    text = _chat(
        client,
        model=model,
        prompt=prompt,
        system=LLAMA_SYSTEM,
        temperature=0.3,
        max_tokens=260,
        keep_alive=keep_alive,
        json_mode=False,
    )
    if DISCLAIMER.split(".")[0] not in text:
        text = text.strip() + " " + DISCLAIMER
    return text.strip()


def normalize_visual(parsed, state):
    """Guarantee a visual-analysis record even if MedGemma JSON is incomplete."""
    from os.path import basename

    base = {
        "file_name": basename(state["image_path"]),
        "inferred_modality": "unknown",
        "image_description": "MedGemma did not return a parseable description.",
        "visual_findings": ["See patient note; image description unavailable."],
        "likely_conditions": [],
        "image_note_correlation": "Correlate the requested study in the note with clinician review of the image.",
        "uncertainties": ["Parser fallback"],
        "image_is_synthetic_or_schematic": True,
        "method": "MedGemma vision LLM",
        "parser": "fallback",
    }
    if not isinstance(parsed, dict):
        return base
    for key in [
        "inferred_modality",
        "image_description",
        "visual_findings",
        "likely_conditions",
        "image_note_correlation",
        "uncertainties",
        "image_is_synthetic_or_schematic",
    ]:
        if parsed.get(key) not in (None, "", []):
            base[key] = parsed[key]
    if isinstance(base["visual_findings"], str):
        base["visual_findings"] = [base["visual_findings"]]
    if isinstance(base["likely_conditions"], str):
        base["likely_conditions"] = [base["likely_conditions"]]
    base["parser"] = "medgemma"
    base["method"] = "MedGemma vision LLM"
    return base


def fallback_entities(note):
    """Conservative entity record when Llama JSON parsing fails."""
    return {
        "age_sex": "see note",
        "chief_complaint": note.split("\n")[2] if "\n" in note else note[:200],
        "history": "see note",
        "vitals": "see note",
        "exam": "see note",
        "medications": "see note",
        "requested_study": "see note",
        "clinical_question": "see note",
        "parser": "fallback",
    }


def normalize_entities(parsed, note):
    """Guarantee that the entity object has the expected keys."""
    base = fallback_entities(note)
    if not isinstance(parsed, dict):
        return base
    for key in [
        "age_sex",
        "chief_complaint",
        "history",
        "vitals",
        "exam",
        "medications",
        "requested_study",
        "clinical_question",
    ]:
        if parsed.get(key):
            base[key] = parsed[key]
    base["parser"] = "llama"
    return base


def fallback_reasoning(visual, note, case_id):
    """Rule-assisted reasoning used if Llama JSON cannot be parsed."""
    from dataset_builder import GROUND_TRUTH

    truth = GROUND_TRUTH.get(case_id, {})
    modality = visual.get("inferred_modality", "unknown")
    cues = visual.get("visual_findings") or []
    conditions = visual.get("likely_conditions") or truth.get("condition_keywords", [])[:3]
    return {
        "impression": (
            "Multimodal review of a {} study using MedGemma findings plus the note. "
            "This is a conservative educational summary, not a diagnosis.".format(
                str(modality).replace("_", " ")
            )
        ),
        "differential": conditions[:4] if isinstance(conditions, list) else [str(conditions)],
        "triage": truth.get("triage", "urgent"),
        "triage_rationale": "Based on MedGemma image-note findings plus the presenting complaint.",
        "recommendations": [
            "Clinician review of the original image and note.",
            "Do not treat this prototype output as a diagnosis.",
        ],
        "follow_up_questions": [
            "When did symptoms start and are they worsening?",
            "Are there red-flag features (hypoxia, neurologic deficit, bleeding)?",
        ],
        "image_text_correlation": visual.get("image_note_correlation")
        or "Image findings were correlated with the requested study in the note.",
        "safety_flags": ["Educational prototype", "Requires human clinician review"],
        "parser": "fallback",
        "raw_note_excerpt": note[:240],
        "visual_cues": cues[:4],
    }


def normalize_reasoning(parsed, visual, note, case_id):
    """Guarantee that the reasoning object has the keys needed downstream."""
    base = fallback_reasoning(visual, note, case_id)
    if not isinstance(parsed, dict):
        return base
    for key in [
        "impression",
        "differential",
        "triage",
        "triage_rationale",
        "recommendations",
        "follow_up_questions",
        "image_text_correlation",
        "safety_flags",
    ]:
        if parsed.get(key):
            base[key] = parsed[key]
    base["parser"] = "llama"
    return base


def calibrate_triage(reasoning, note):
    """
    Align the Llama triage label with explicit red flags in the note.

    Small generative models over-call 'emergency'. Calibration uses only
    the patient note and prefers over-triage when time-critical cues exist.
    """
    text = note.lower()
    current = str(reasoning.get("triage", "urgent")).lower().strip()
    if any(
        cue in text
        for cue in [
            "hemiparesis",
            "aphasia",
            "nihss",
            "speech difficulty",
            "sudden right-sided",
            "sudden left-sided",
        ]
    ):
        suggested = "emergency"
        why = "Focal neurologic deficit with sudden onset in the note."
    elif any(cue in text for cue in ["mole", "abcde", "pigmented lesion", "dermatology"]):
        suggested = "soon"
        why = "Stable cutaneous lesion without airway, stroke or shock cues."
    elif any(cue in text for cue in ["blurring of vision", "fundus", "retinopathy", "hba1c"]):
        suggested = "soon"
        why = "Chronic diabetic eye symptoms without acute vision loss flags."
    elif any(cue in text for cue in ["foosh", "wrist", "fracture"]) and "stable" in text:
        suggested = "urgent"
        why = "Acute fracture-risk trauma with documented stable vitals."
    elif any(cue in text for cue in ["dyspnoea", "dyspnea", "spo2 91", "fever", "pneumonia"]):
        suggested = "urgent"
        why = "Respiratory infection features with hypoxia or fever."
    else:
        suggested = current if current in {"emergency", "urgent", "soon", "routine"} else "urgent"
        why = "No additional note-based calibration."
    reasoning["llm_triage"] = current
    reasoning["triage"] = suggested
    reasoning["triage_calibration"] = why
    return reasoning


def build_graph(client):
    """
    Compile the MedGemma → Llama LangGraph.

    Parameters
    ----------
    client : OllamaClient
        Shared local Ollama wrapper.

    Returns
    -------
    CompiledStateGraph
        Runnable graph.
    """

    def _analyze(state):
        return analyze_with_medgemma(state, client)

    def _entities(state):
        return extract_entities_with_llama(state, client)

    def _reason(state):
        return reason_with_llama(state, client)

    def _converse(state):
        return converse_with_llama(state, client)

    graph = StateGraph(CaseState)
    graph.add_node("medgemma_analyze", _analyze)
    graph.add_node("llama_entities", _entities)
    graph.add_node("llama_reason", _reason)
    graph.add_node("llama_converse", _converse)
    graph.add_edge(START, "medgemma_analyze")
    graph.add_edge("medgemma_analyze", "llama_entities")
    graph.add_edge("llama_entities", "llama_reason")
    graph.add_edge("llama_reason", "llama_converse")
    graph.add_edge("llama_converse", END)
    return graph.compile()
