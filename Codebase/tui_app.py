"""
tui_app.py
----------
Terminal UI for upload, analysis review, and follow-up chat.
"""

import os
import sys

from ingestion import ingest_pair
from medical_assistant import (
    dashboard_payload,
    get_graph,
    process_case,
    process_follow_up,
    save_json,
)
from ollama_client import OllamaError
from runtime_profile import get_profile

DOMAIN_NOTICE = (
    "Accepted images: radiology scans (X-ray, CT, MRI) or histopathology slides only. "
    "The image domain is detected automatically — do not upload dermatology, "
    "ophthalmology, clinical photos, or other out-of-scope images."
)


def _prompt(label, default=None):
    suffix = " [{}]".format(default) if default else ""
    value = input("{}{}: ".format(label, suffix)).strip()
    return value or (default or "")


def _read_file(path):
    path = os.path.expanduser(path.strip().strip('"'))
    if not os.path.isfile(path):
        raise FileNotFoundError("File not found: {}".format(path))
    return path


def _section(title):
    line = "=" * min(72, max(len(title) + 4, 40))
    print("\n{}".format(line))
    print(title)
    print(line)


def _bullet_list(items):
    for item in items or []:
        print("  • {}".format(item))


def _wrap(text, width=72):
    text = (text or "").strip()
    if not text:
        return
    words = text.split()
    line = ""
    for word in words:
        candidate = (line + " " + word).strip()
        if len(candidate) > width:
            print(line)
            line = word
        else:
            line = candidate
    if line:
        print(line)


def _render_dashboard(payload):
    domain = payload.get("image_domain") or "unknown"
    _section("AUTO-DETECTED DOMAIN: {}".format(domain.upper()))
    _section("TRIAGE: {}".format((payload.get("triage") or "—").upper()))
    _wrap(payload.get("triage_rationale"))
    _section("IMPRESSION")
    _wrap(payload.get("impression"))
    _section("VISUAL FINDINGS")
    _bullet_list(payload.get("visual_findings"))
    _section("DIFFERENTIAL")
    _bullet_list(payload.get("differential"))
    _section("RECOMMENDATIONS")
    _bullet_list(payload.get("recommendations"))
    correlation = payload.get("image_note_correlation")
    if correlation:
        _section("IMAGE–NOTE CORRELATION")
        _wrap(correlation)
    refs = payload.get("references") or []
    if refs:
        _section("REFERENCES")
        for ref in refs:
            print("  • {} — {}".format(ref.get("title", "Link"), ref.get("url", "")))
    suggested = payload.get("follow_up_questions") or []
    if suggested:
        _section("SUGGESTED FOLLOW-UP QUESTIONS")
        _bullet_list(suggested)
    conversation = payload.get("conversation") or []
    if conversation:
        _section("CONVERSATION")
        for turn in conversation:
            role = turn.get("role", "user").upper()
            print("\n[{}]".format(role))
            _wrap(turn.get("content"))
    _render_evaluation(payload.get("evaluation") or {})
    print()


def _render_evaluation(evaluation):
    if not evaluation:
        return
    sat = evaluation.get("user_satisfaction") or {}
    _section("EVALUATION METRICS")
    print("  Cross-modal correlation: {}".format((evaluation.get("cross_modal_correlation") or {}).get("score")))
    print("  Robustness to data variation: {}".format((evaluation.get("robustness_to_data_variation") or {}).get("score")))
    print("  Explanation quality: {}".format((evaluation.get("explanation_quality") or {}).get("score")))
    print("  User satisfaction (interface, 1–5): {}".format(sat.get("interface_likert_1_to_5")))
    print("  User satisfaction (explanations, 1–5): {}".format(sat.get("explanation_likert_1_to_5")))


def _unload_vision(client, vision_model):
    if not get_profile().unload_vision_after_analysis:
        return
    try:
        client.unload_model(vision_model)
    except OllamaError:
        pass


def _wait_hint():
    profile = get_profile()
    if profile.is_gpu:
        return "This usually completes in under a few minutes on GPU…"
    return "This may take several minutes on a 16 GB CPU laptop…"


def _follow_up_hint():
    profile = get_profile()
    if profile.is_gpu:
        return "Llama is re-triaging and replying (typically faster on GPU)…"
    return "Llama is re-triaging and replying (may take 1–2 minutes on CPU)…"


def _collect_case_paths():
    _section("NEW CASE")
    print(DOMAIN_NOTICE)
    print()
    note_path = _read_file(_prompt("Clinical note path (.txt/.md)"))
    image_path = _read_file(_prompt("Radiology or pathology image path (.jpg/.png/.dcm)"))
    return note_path, image_path


def _ingest_paths(note_path, image_path):
    with open(note_path, "rb") as handle:
        note_bytes = handle.read()
    with open(image_path, "rb") as handle:
        image_bytes = handle.read()
    return ingest_pair(
        note_bytes,
        os.path.basename(note_path),
        image_bytes,
        os.path.basename(image_path),
    )


def run_tui(_args, client, vision_model, llama_model):
    """Interactive terminal session: upload, analyze, chat."""
    print("\nMultimodal Medical Assistant — Terminal UI (TUI)")
    print("Clinical decision support: radiology and pathology image–text analysis with follow-up chat.")
    print(DOMAIN_NOTICE)
    print()

    get_graph(client)

    try:
        note_path, image_path = _collect_case_paths()
        ingested = _ingest_paths(note_path, image_path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print("ERROR:", exc, file=sys.stderr)
        return 1

    session_id = os.path.basename(ingested["session_dir"])
    print("\nAnalyzing case (MedGemma detects domain, then Llama reasons).")
    print(_wait_hint() + "\n")

    try:
        record = process_case(
            client=client,
            vision_model=vision_model,
            llama_model=llama_model,
            case_id=0,
            image_path=ingested["image"]["path"],
            text_path=ingested["text"]["path"],
        )
    except ValueError as exc:
        print("ERROR:", exc, file=sys.stderr)
        return 1
    except OllamaError as exc:
        print("ERROR:", exc, file=sys.stderr)
        return 2

    record["privacy"] = {
        "text": ingested["text"]["privacy"],
        "image": ingested["image"]["privacy"],
        "originals_stored": False,
    }
    save_json(os.path.join(ingested["session_dir"], "conversation.json"), record)
    _unload_vision(client, vision_model)

    payload = dashboard_payload(record, session_id)
    _render_dashboard(payload)
    print("Session saved: {}".format(ingested["session_dir"]))

    print("\nFollow-up chat (type 'quit' or 'exit' to leave).")
    while True:
        message = input("\nYou> ").strip()
        if not message:
            continue
        if message.lower() in {"quit", "exit", "q"}:
            print("Leaving TUI. Goodbye.")
            break
        print(_follow_up_hint())
        try:
            _unload_vision(client, vision_model)
            payload = process_follow_up(client, session_id, message, llama_model)
        except (ValueError, FileNotFoundError) as exc:
            print("ERROR:", exc, file=sys.stderr)
            continue
        except OllamaError as exc:
            print("ERROR:", exc, file=sys.stderr)
            return 2
        print("\nAssistant>")
        _wrap(payload.get("latest_reply"))
        triage = payload.get("triage")
        if triage:
            print("\n(updated triage: {})".format(triage))
        if payload.get("impression"):
            _section("UPDATED IMPRESSION")
            _wrap(payload.get("impression"))
        if payload.get("differential"):
            _section("UPDATED DIFFERENTIAL")
            _bullet_list(payload.get("differential"))
        if payload.get("recommendations"):
            _section("UPDATED RECOMMENDATIONS")
            _bullet_list(payload.get("recommendations"))
        suggested = payload.get("follow_up_questions") or []
        if suggested:
            _section("SUGGESTED NEXT QUESTIONS")
            _bullet_list(suggested)

    return 0
