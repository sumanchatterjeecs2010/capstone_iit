"""
main.py
-------
Entry point for the HPPCS[04] Multimodal Medical Assistant.

Required behaviour
- Read 5 medical images paired with 5 prescription / patient-detail files.
- Use two LLMs: MedGemma 1.5 4B (image + note) and Llama 3.2 (generation).
- Use LangGraph to order those LLM calls.
- Write one conversation JSON file per input (conversation_01.json ... _05.json).

Usage
    python main.py
    python main.py --model llama3.2:3b --vision_model medgemma:4b
    python main.py --model llama3.2:1b
"""

import argparse
import json
import os
import sys

from dataset_builder import generate_all_cases, list_input_pairs
from medical_assistant import aggregate_metrics, get_graph, process_case, save_json
from ollama_client import OllamaClient, OllamaError


ROOT = os.path.dirname(os.path.abspath(__file__))

VISION_ALIASES = [
    "medgemma:4b",
    "medgemma",
    "medgemma:latest",
    "medgemma:1.5",
    "dcarrascosa/medgemma-1.5-4b-it:Q4_K_M",
    "dcarrascosa/medgemma-1.5-4b-it",
]


def parse_args(argv=None):
    """Parse command-line arguments for the assistant."""
    parser = argparse.ArgumentParser(
        description="Multimodal Medical Assistant using MedGemma, Llama 3.2 and LangGraph."
    )
    parser.add_argument(
        "--model",
        "--model1",
        dest="model",
        default="llama3.2:3b",
        help="Text LLM for entities, triage JSON and dialogue (default: llama3.2:3b).",
    )
    parser.add_argument(
        "--vision_model",
        default="medgemma:4b",
        help="Vision LLM for image + note understanding (default: medgemma:4b).",
    )
    parser.add_argument(
        "--api_key",
        default="",
        help="Optional API key (ignored for local Ollama).",
    )
    parser.add_argument(
        "--host",
        default="http://127.0.0.1:11434",
        help="Ollama server URL.",
    )
    parser.add_argument(
        "--skip_dataset_build",
        action="store_true",
        help="Do not re-download public teaching images or rewrite notes.",
    )
    return parser.parse_args(argv)


def _is_installed(client, name):
    """Return True if an Ollama tag is present locally."""
    installed = [item.lower() for item in client.list_model_names()]
    lowered = name.lower()
    return lowered in installed or any(lowered in item or item.startswith(lowered.split(":")[0]) for item in installed)


def ensure_llama(client, model):
    """Verify Llama 3.2 is installed, falling back from 3B to 1B if needed."""
    resolved = client.resolve_model(model)
    if not _is_installed(client, resolved) and model.lower().startswith("llama3.2:3b"):
        print("llama3.2:3b is not installed; falling back to llama3.2:1b.", flush=True)
        model = "llama3.2:1b"
        resolved = client.resolve_model(model)
    if not _is_installed(client, resolved):
        raise OllamaError("Model '{}' is not installed. Run: ollama pull {}".format(model, model))
    return resolved


def ensure_vision(client, model):
    """Verify a MedGemma vision tag is installed, trying common aliases."""
    candidates = [model] + [alias for alias in VISION_ALIASES if alias != model]
    last = model
    for candidate in candidates:
        resolved = client.resolve_model(candidate)
        last = resolved
        if _is_installed(client, resolved):
            if resolved != model:
                print("Using installed vision model:", resolved, flush=True)
            return resolved
    raise OllamaError(
        "MedGemma is not installed. Run: ollama pull medgemma:4b "
        "(tried '{}')".format(last)
    )


def run(args):
    """Execute the five-case multimodal pipeline and write JSON outputs."""
    os.chdir(ROOT)
    if not args.skip_dataset_build:
        generate_all_cases(overwrite=False)

    client = OllamaClient(host=args.host)
    if not client.is_available():
        raise OllamaError(
            "Ollama is not reachable at {}. Start it, then re-run python main.py.".format(client.host)
        )
    vision_model = ensure_vision(client, args.vision_model)
    llama_model = ensure_llama(client, args.model)
    print("Using vision LLM     :", vision_model, flush=True)
    print("Using language LLM   :", llama_model, flush=True)
    print("Using orchestrator   : LangGraph", flush=True)
    if args.api_key:
        print("API key received as a parameter and will be ignored for local Ollama calls.", flush=True)

    get_graph(client)

    records = []
    for case_id, image_path, text_path in list_input_pairs():
        if not os.path.exists(image_path) or not os.path.exists(text_path):
            print("Missing input for case", case_id, file=sys.stderr)
            return 1
        print("\n=== Processing patient_{:02d} ===".format(case_id), flush=True)
        record = process_case(
            client=client,
            vision_model=vision_model,
            llama_model=llama_model,
            case_id=case_id,
            image_path=image_path,
            text_path=text_path,
        )
        out_name = "conversation_{:02d}.json".format(case_id)
        save_json(os.path.join(ROOT, out_name), record)
        records.append(record)
        print("Wrote", out_name, "in", record["elapsed_seconds"], "s", flush=True)
        print("Triage:", record["clinical_reasoning"].get("triage"), flush=True)

    summary = aggregate_metrics(records)
    save_json(os.path.join(ROOT, "evaluation_summary.json"), summary)
    print("\n=== Evaluation summary (synthetic keyword metrics) ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def main():
    """Command-line entry point used by `python main.py`."""
    args = parse_args()
    try:
        code = run(args)
    except OllamaError as exc:
        print("ERROR:", exc, file=sys.stderr)
        code = 2
    sys.exit(code)


if __name__ == "__main__":
    main()
