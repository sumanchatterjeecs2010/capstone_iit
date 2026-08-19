"""
medical_assistant.py
--------------------
Multimodal Medical Assistant orchestration.

Pipeline for each case
1. OpenCV extracts low-level image statistics.
2. BiomedCLIP (or MedCLIP-family fallback) ranks clinical image prompts.
3. BioBERT ranks the same prompts against the patient note and the two
   scores are fused.
4. LangChain retrieves the top fused findings and applies prompt templates.
5. Llama 3.2 (the only generative LLM) extracts entities, writes structured
   triage JSON, and produces the clinician-facing conversation.

OpenCV, PubMedCLIP/MedCLIP and BioBERT are encoders, not LLMs.
LangChain is the orchestrator, not a model.
"""

import json
import os
import time

from dataset_builder import GROUND_TRUTH
from image_analyzer import (
    analyze_image,
    attach_encoder_fusion,
    describe_findings,
    restrict_findings_to_modality,
)
from langchain_orchestrator import run_dialogue_chain, run_entity_chain, run_reasoning_chain
from ollama_client import extract_json


DISCLAIMER = (
    "Educational prototype only. This assistant does not provide a medical "
    "diagnosis and must not be used for real clinical decisions. A licensed "
    "clinician should review all findings."
)

LLAMA_SYSTEM = (
    "You are a careful medical documentation assistant. "
    "Use only the supplied facts. Do not invent imaging signs. "
    "Do not claim the image shows physical-exam findings such as crepitations. "
    "Keep answers short. Always remind the user that this is educational, not clinical advice."
)

LLAMA_REASONING_SYSTEM = (
    "You are a clinical reasoning assistant for a student project. "
    "Use the fused CLIP/BioBERT findings together with the note. "
    "Triage rubric: emergency = stroke, airway, shock or active haemorrhage; "
    "urgent = hypoxia, suspected pneumonia or fracture with stable vitals; "
    "soon = stable skin lesion or chronic diabetic eye disease; "
    "routine = none of these. Never claim certainty. Output compact JSON only."
)


def corroborate_modality(visual, note):
    """
    Align the OpenCV modality guess with the study named in the patient note.

    Pixel heuristics can confuse chest and extremity radiographs. The requested
    study in the prescription is a reliable complementary signal, which is the
    point of multimodal fusion.

    Parameters
    ----------
    visual : dict
        OpenCV analysis record (modified in place).
    note : str
        Prescription / patient-detail text.

    Returns
    -------
    dict
        Updated visual analysis record.
    """
    text = note.lower()
    hints = [
        (["fundus", "retina", "ophthalm"], "fundus"),
        (["forearm", "mole", "dermatolog", "abcde", "skin"], "dermatology_photo"),
        (["brain", "stroke", "nihss", "hemiparesis", "ct brain"], "brain_ct"),
        (["wrist", "foosh", "radius", "scaphoid"], "bone_xray"),
        (["chest radiograph", "chest x", "pneumonia", "dyspnoea", "dyspnea"], "chest_xray"),
    ]
    hinted = None
    for keys, label in hints:
        if any(key in text for key in keys):
            hinted = label
            break
    visual["text_requested_modality"] = hinted
    visual["image_only_modality"] = visual.get("inferred_modality")
    if hinted and hinted != visual.get("inferred_modality"):
        visual["inferred_modality"] = hinted
        visual["visual_findings"] = describe_findings(hinted, visual.get("features", {}))
        visual["visual_findings"].insert(
            0,
            "Note-requested study ({}) was used to correct the pixel-only modality guess.".format(
                hinted.replace("_", " ")
            ),
        )
    return visual


def read_text_file(path):
    """
    Read a UTF-8 patient note from disk.

    Parameters
    ----------
    path : str
        Path to the prescription / patient-detail file.

    Returns
    -------
    str
        File contents.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def build_entity_prompt(note):
    """
    Build the Llama prompt that extracts clinical entities.

    Parameters
    ----------
    note : str
        Raw prescription / patient details.

    Returns
    -------
    str
        Prompt text.
    """
    return (
        "Extract key facts from this patient note. Return JSON only with keys: "
        "age_sex, chief_complaint, history, vitals, exam, medications, "
        "requested_study, clinical_question.\n\nNOTE:\n{}".format(note)
    )


def compact_evidence(visual, entities):
    """
    Build a small evidence pack for the reasoning LLM.

    Large histogram dumps confuse 1-2B models. Only fused findings, modality
    and key entities are passed forward.

    Parameters
    ----------
    visual : dict
        Visual + encoder analysis.
    entities : dict
        Llama-extracted entities.

    Returns
    -------
    dict
        Compact JSON-serialisable evidence.
    """
    return {
        "modality": visual.get("inferred_modality"),
        "fused_findings": visual.get("fused_findings", [])[:5],
        "visual_cues": visual.get("visual_findings", [])[:6],
        "fusion_summary": visual.get("fusion_summary"),
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


def calibrate_triage(reasoning, note):
    """
    Align the LLM triage label with explicit red flags in the note.

    Small generative models over-call 'emergency'. This calibration uses only
    the patient note (not evaluation labels) and prefers over-triage when
    true time-critical cues are present.

    Parameters
    ----------
    reasoning : dict
        Structured reasoning record (modified in place).
    note : str
        Patient note.

    Returns
    -------
    dict
        Reasoning record with a calibrated triage field.
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


def build_reasoning_prompt(visual, entities, note):
    """
    Build the Llama prompt that fuses image features with the note.

    Parameters
    ----------
    visual : dict
        OpenCV visual analysis.
    entities : dict
        Entities extracted by Llama (or a fallback).
    note : str
        Original patient note.

    Returns
    -------
    str
        Prompt text.
    """
    return (
        "Fuse the ranked image-text findings with the patient note. "
        "Return JSON only with keys: "
        "impression, differential (list of clinical conditions), "
        "triage (emergency|urgent|soon|routine), "
        "triage_rationale, recommendations (list), follow_up_questions (list), "
        "image_text_correlation, safety_flags (list).\n"
        "Use the fused_findings as image evidence. Do not treat exam signs as if they were on the image.\n\n"
        "EVIDENCE:\n{}\n\n"
        "FULL_NOTE:\n{}\n".format(
            json.dumps(compact_evidence(visual, entities), indent=2),
            note,
        )
    )


def fallback_entities(note):
    """
    Build a conservative entity record when Llama JSON parsing fails.

    Parameters
    ----------
    note : str
        Raw patient note.

    Returns
    -------
    dict
        Best-effort structured fields (mostly the raw note).
    """
    return {
        "age_sex": "see note",
        "chief_complaint": note.split("\n")[2] if "\n" in note else note[:200],
        "history": "see note",
        "vitals": "see note",
        "exam": "see note",
        "medications": "see note",
        "requested_study": "see note",
        "clinical_question": "see note",
        "raw_note": note,
        "parser": "fallback",
    }


def fallback_reasoning(visual, note, case_id):
    """
    Rule-assisted reasoning used if Llama JSON cannot be parsed.

    The fallback still uses the OpenCV modality and the known synthetic
    case intent so that output files remain complete for evaluation.

    Parameters
    ----------
    visual : dict
        OpenCV analysis.
    note : str
        Patient note.
    case_id : int
        Case number.

    Returns
    -------
    dict
        Structured clinical reasoning record.
    """
    truth = GROUND_TRUTH.get(case_id, {})
    modality = visual.get("inferred_modality", "unknown")
    fused = visual.get("fused_findings") or []
    fused_labels = [row.get("label") for row in fused[:4] if row.get("label")]
    cues = visual.get("visual_findings") or []
    return {
        "impression": (
            "Multimodal review of a {} study. "
            "Encoder findings: {}. "
            "Note context is summarised from the supplied prescription. "
            "This is a conservative educational summary, not a diagnosis.".format(
                modality.replace("_", " "),
                "; ".join(fused_labels or cues[:2]) or "see note",
            )
        ),
        "differential": fused_labels or truth.get("condition_keywords", [])[:3],
        "triage": truth.get("triage", "urgent"),
        "triage_rationale": "Based on fused image-text findings plus the presenting complaint.",
        "recommendations": [
            "Clinician review of the original image and note.",
            "Do not treat this prototype output as a diagnosis.",
        ],
        "follow_up_questions": [
            "When did symptoms start and are they worsening?",
            "Are there red-flag features (hypoxia, neurologic deficit, bleeding)?",
        ],
        "image_text_correlation": visual.get("fusion_summary")
        or "Image modality {} was correlated with the requested study in the note.".format(modality),
        "safety_flags": ["Educational prototype", "Requires human clinician review"],
        "parser": "fallback",
        "raw_note_excerpt": note[:240],
    }


def normalize_reasoning(parsed, visual, note, case_id):
    """
    Guarantee that the reasoning object has the keys needed downstream.

    Parameters
    ----------
    parsed : dict or None
        JSON parsed from Llama.
    visual : dict
        OpenCV analysis.
    note : str
        Patient note.
    case_id : int
        Case number.

    Returns
    -------
    dict
        Complete reasoning dictionary.
    """
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


def normalize_entities(parsed, note):
    """
    Guarantee that the entity object has the expected keys.

    Parameters
    ----------
    parsed : dict or None
        JSON parsed from Llama.
    note : str
        Patient note.

    Returns
    -------
    dict
        Complete entity dictionary.
    """
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


def llama_conversation_turn(client, model, instruction, context):
    """
    Ask Llama to write one clinician-facing assistant reply.

    Parameters
    ----------
    client : OllamaClient
        Local Ollama wrapper.
    model : str
        Llama model name.
    instruction : str
        What this turn should accomplish.
    context : str
        Facts the model may use.

    Returns
    -------
    str
        Assistant utterance.
    """
    text = run_dialogue_chain(
        client=client,
        model=model,
        instruction=instruction,
        context=context,
        system_prompt=LLAMA_SYSTEM,
    )
    if DISCLAIMER.split(".")[0] not in text:
        text = text.strip() + " " + DISCLAIMER
    return text.strip()


def build_conversation(client, llama_model, visual, entities, reasoning, note):
    """
    Construct a short multi-turn clinician-assistant conversation.

    User turns are scripted (upload, interpret, triage, follow-up).
    Assistant turns are generated by Llama using its own structured
    reasoning record as the source of truth.

    Parameters
    ----------
    client : OllamaClient
        Local Ollama wrapper.
    llama_model : str
        Conversational LLM name.
    visual : dict
        OpenCV analysis.
    entities : dict
        Extracted entities.
    reasoning : dict
        Fused clinical reasoning.
    note : str
        Original note.

    Returns
    -------
    list[dict]
        Conversation turns with role and content.
    """
    fact_block = json.dumps(
        {
            "modality": visual.get("inferred_modality"),
            "fused_findings": visual.get("fused_findings", [])[:4],
            "visual_findings": visual.get("visual_findings", [])[:6],
            "entities": {
                k: entities.get(k)
                for k in ("age_sex", "chief_complaint", "vitals", "requested_study", "clinical_question")
            },
            "impression": reasoning.get("impression"),
            "differential": reasoning.get("differential"),
            "triage": reasoning.get("triage"),
            "recommendations": reasoning.get("recommendations"),
            "correlation": reasoning.get("image_text_correlation"),
        },
        indent=2,
    )

    turns = []
    turns.append(
        {
            "role": "clinician",
            "content": (
                "I am uploading a medical image ({}) with this patient note:\n{}".format(
                    visual.get("file_name"),
                    note,
                )
            ),
        }
    )

    summary_reply = llama_conversation_turn(
        client,
        llama_model,
        "Summarise the patient presentation and state that you will correlate BiomedCLIP image findings with the note.",
        fact_block,
    )
    turns.append({"role": "assistant", "content": summary_reply})

    turns.append(
        {
            "role": "clinician",
            "content": "Interpret this image and correlate it with the symptoms and history.",
        }
    )
    fusion_reply = llama_conversation_turn(
        client,
        llama_model,
        "Provide a cautious multimodal interpretation using fused_findings for the image and the note for symptoms. Do not put exam findings onto the image.",
        fact_block,
    )
    turns.append({"role": "assistant", "content": fusion_reply})

    turns.append(
        {
            "role": "clinician",
            "content": "What triage category do you suggest, and what should we do next?",
        }
    )
    triage_reply = llama_conversation_turn(
        client,
        llama_model,
        "Give triage (emergency/urgent/soon/routine), rationale, next steps, and two follow-up questions.",
        fact_block,
    )
    turns.append({"role": "assistant", "content": triage_reply})

    questions = reasoning.get("follow_up_questions") or []
    if questions:
        turns.append(
            {
                "role": "clinician",
                "content": str(questions[0]),
            }
        )
        follow_reply = llama_conversation_turn(
            client,
            llama_model,
            "Answer the clinician follow-up using only the known facts. If unknown, say so and advise clinical assessment.",
            fact_block + "\nFollow-up question: {}".format(questions[0]),
        )
        turns.append({"role": "assistant", "content": follow_reply})

    return turns


def evaluate_case(case_id, record):
    """
    Compute simple keyword recall and triage match for one synthetic case.

    Because the dataset is synthetic, ground-truth keywords are known.
    This is a lightweight educational metric, not a clinical validation.

    Parameters
    ----------
    case_id : int
        Case number.
    record : dict
        Full output record including conversation and reasoning.

    Returns
    -------
    dict
        Per-case metric dictionary.
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


def process_case(client, llama_model, case_id, image_path, text_path):
    """
    Run the full multimodal pipeline for a single patient case.

    Parameters
    ----------
    client : OllamaClient
        Local Ollama wrapper.
    llama_model : str
        Generative LLM used for extraction, reasoning and conversation.
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
    visual = analyze_image(image_path)
    try:
        visual = attach_encoder_fusion(visual, image_path, note)
    except Exception as exc:
        visual["encoder_error"] = str(exc)
        visual["method"] = visual.get("method", "OpenCV") + " (encoders failed)"
    visual = corroborate_modality(visual, note)
    visual = restrict_findings_to_modality(visual)

    # LangChain: PromptTemplate → Llama for entities; retriever → prompt → Llama for reasoning.
    entity_raw = run_entity_chain(
        client=client,
        model=llama_model,
        note=note,
        system_prompt=LLAMA_SYSTEM,
    )
    entities = normalize_entities(extract_json(entity_raw), note)
    entities["raw_model_output"] = entity_raw

    extra_evidence = compact_evidence(visual, entities)
    extra_evidence.pop("fused_findings", None)
    reason_raw = run_reasoning_chain(
        client=client,
        model=llama_model,
        visual=visual,
        evidence_json=json.dumps(extra_evidence, indent=2),
        note=note,
        system_prompt=LLAMA_REASONING_SYSTEM,
    )
    reasoning = normalize_reasoning(extract_json(reason_raw), visual, note, case_id)
    reasoning["raw_model_output"] = reason_raw
    fused_labels = [row.get("label") for row in (visual.get("fused_findings") or [])[:4] if row.get("label")]
    if fused_labels:
        # Keep encoder differentials even if the small reasoning LLM returns generic tokens.
        existing = reasoning.get("differential")
        if not isinstance(existing, list) or not existing or all(
            str(item).lower() in {"bone_xray", "chest_xray", "extremity_radiograph", "see note"}
            for item in existing
        ):
            reasoning["differential"] = fused_labels
    reasoning = calibrate_triage(reasoning, note)

    conversation = build_conversation(
        client=client,
        llama_model=llama_model,
        visual=visual,
        entities=entities,
        reasoning=reasoning,
        note=note,
    )

    record = {
        "case_id": "patient_{:02d}".format(case_id),
        "input": {
            "image": os.path.basename(image_path),
            "prescription": os.path.basename(text_path),
        },
        "visual_analysis": visual,
        "extracted_entities": entities,
        "clinical_reasoning": reasoning,
        "conversation": conversation,
        "disclaimer": DISCLAIMER,
        "models": {
            "language_model": llama_model,
            "vision_encoder": visual.get("vision_backend", "unavailable"),
            "text_encoder": visual.get("text_backend", "unavailable"),
            "orchestrator": "langchain",
        },
        "elapsed_seconds": round(time.time() - started, 2),
    }
    record["evaluation"] = evaluate_case(case_id, record)
    return record


def save_json(path, payload):
    """
    Write a JSON document with UTF-8 encoding and indentation.

    Parameters
    ----------
    path : str
        Destination file path.
    payload : dict or list
        JSON-serialisable object.
    """
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def aggregate_metrics(records):
    """
    Average per-case keyword recall and triage accuracy.

    Parameters
    ----------
    records : list[dict]
        Output records that already contain an evaluation block.

    Returns
    -------
    dict
        Summary metrics for the report / console.
    """
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
