"""
main.py
-------
Entry point. Default: clinician dashboard at http://127.0.0.1:8000/

    python main.py              # dashboard + chat
    python main.py --batch      # optional 5-case demo in sample_data/
    python main.py --text a.txt --image b.jpg
"""

import argparse
import json
import os
import sys

from dataset_builder import generate_all_cases, list_input_pairs
from medical_assistant import aggregate_metrics, get_graph, process_case, save_json
from ollama_client import OllamaClient, OllamaError


ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(ROOT, "sample_data")
UPLOADS_DIR = os.path.join(ROOT, "uploads", "processed")
VISION_ALIASES = ["medgemma:4b", "medgemma", "medgemma:latest", "medgemma:1.5"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Multimodal Medical Assistant")
    parser.add_argument("--model", default="llama3.2:3b", help="Text LLM (default llama3.2:3b)")
    parser.add_argument("--vision_model", default="medgemma:4b", help="Vision LLM (default medgemma:4b)")
    parser.add_argument("--host", default="http://127.0.0.1:11434", help="Ollama URL")
    parser.add_argument("--skip_dataset_build", action="store_true", help="Skip sample_data download")
    parser.add_argument("--batch", action="store_true", help="Run five teaching cases")
    parser.add_argument("--serve", action="store_true", help="Start dashboard (default)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--text", default="", help="CLI note path")
    parser.add_argument("--image", default="", help="CLI image path")
    parser.add_argument("--image_domain", default="radiology")
    return parser.parse_args(argv)


def _is_installed(client, name):
    installed = [item.lower() for item in client.list_model_names()]
    lowered = name.lower()
    return lowered in installed or any(lowered in item for item in installed)


def ensure_llama(client, model):
    resolved = client.resolve_model(model)
    if not _is_installed(client, resolved) and model.lower().startswith("llama3.2:3b"):
        print("llama3.2:3b missing; using llama3.2:1b", flush=True)
        model = "llama3.2:1b"
        resolved = client.resolve_model(model)
    if not _is_installed(client, resolved):
        raise OllamaError("Model '{}' not installed. Run: ollama pull {}".format(model, model))
    return resolved


def ensure_vision(client, model):
    for candidate in [model] + [a for a in VISION_ALIASES if a != model]:
        resolved = client.resolve_model(candidate)
        if _is_installed(client, resolved):
            if resolved != model:
                print("Using vision model:", resolved, flush=True)
            return resolved
    raise OllamaError("MedGemma not installed. Run: ollama pull medgemma:4b")


def setup_models(args):
    client = OllamaClient(host=args.host)
    if not client.is_available():
        raise OllamaError("Ollama not reachable at {}.".format(client.host))
    vision = ensure_vision(client, args.vision_model)
    llama = ensure_llama(client, args.model)
    return client, vision, llama


def run_single_upload(args, client, vision_model, llama_model):
    if not os.path.exists(args.text) or not os.path.exists(args.image):
        print("Missing --text or --image.", file=sys.stderr)
        return 1
    record = process_case(
        client, vision_model, llama_model, 0, args.image, args.text, args.image_domain
    )
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    out = os.path.join(UPLOADS_DIR, "conversation_upload.json")
    save_json(out, record)
    print("Wrote", out, flush=True)
    return 0


def run_batch(args, client, vision_model, llama_model):
    if not args.skip_dataset_build:
        generate_all_cases(overwrite=False)
    records = []
    for case_id, image_path, text_path in list_input_pairs():
        if not os.path.exists(image_path) or not os.path.exists(text_path):
            print("Missing sample_data case", case_id, file=sys.stderr)
            return 1
        print("\n=== patient_{:02d} ===".format(case_id), flush=True)
        record = process_case(client, vision_model, llama_model, case_id, image_path, text_path)
        save_json(os.path.join(SAMPLE_DIR, "conversation_{:02d}.json".format(case_id)), record)
        records.append(record)
        print("Triage:", record["clinical_reasoning"].get("triage"), flush=True)
    summary = aggregate_metrics(records)
    save_json(os.path.join(SAMPLE_DIR, "evaluation_summary.json"), summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def run(args):
    os.chdir(ROOT)
    use_dashboard = args.serve or (not args.batch and not (args.text and args.image))
    if use_dashboard:
        client, vision, llama = setup_models(args)
        os.environ["OLLAMA_HOST"] = args.host
        os.environ["VISION_MODEL"] = vision
        os.environ["LLAMA_MODEL"] = llama
        from upload_app import serve

        print("Open http://127.0.0.1:{}/".format(args.port), flush=True)
        serve(port=args.port)
        return 0

    client, vision, llama = setup_models(args)
    print("Vision:", vision, "| Language:", llama, flush=True)
    get_graph(client)
    if args.text and args.image:
        return run_single_upload(args, client, vision, llama)
    if args.batch:
        return run_batch(args, client, vision, llama)
    print("Use --batch or --text with --image.", file=sys.stderr)
    return 1


def main():
    args = parse_args()
    try:
        code = run(args)
    except OllamaError as exc:
        print("ERROR:", exc, file=sys.stderr)
        code = 2
    sys.exit(code)


if __name__ == "__main__":
    main()
