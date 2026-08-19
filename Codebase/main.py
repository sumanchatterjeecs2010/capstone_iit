"""
main.py
-------
Entry point for the HPPCS[04] Multimodal Medical Assistant.

Required behaviour (from the capstone instructions)
- Read 5 medical images paired with 5 prescription / patient-detail files.
- Use Llama 3.2 via Ollama as the generative LLM, plus OpenCV, PubMedCLIP
  (MedCLIP family) and BioBERT for image-text evidence, with LangChain
  for prompt templates, finding retrieval and chain orchestration.
- Write one conversation JSON file per input (conversation_01.json ... _05.json).

Usage
    python main.py
    python main.py --model llama3.2:3b
    python main.py --model llama3.2:1b
    python main.py --api_key YOUR_KEY   # accepted for API-based runs; unused for Ollama
"""

import argparse
import os
import sys

from dataset_builder import generate_all_cases, list_input_pairs
from medical_assistant import aggregate_metrics, process_case, save_json
from multimodal_encoder import preload_encoders
from ollama_client import OllamaClient, OllamaError


# All input and output files live next to this script so they can be reached with "./".
ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv=None):
    """
    Parse command-line arguments for the assistant.

    Parameters
    ----------
    argv : list[str] or None
        Argument vector; defaults to sys.argv[1:].

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Multimodal Medical Assistant (HPPCS[04]) using Llama 3.2 plus medical encoders."
    )
    parser.add_argument(
        "--model",
        "--model1",
        dest="model",
        default="llama3.2:3b",
        help="Generative LLM (default: llama3.2:3b; use llama3.2:1b on 4 GB RAM).",
    )
    parser.add_argument(
        "--api_key",
        default="",
        help="Optional API key (required only if a cloud LLM API is used; ignored for Ollama).",
    )
    parser.add_argument(
        "--host",
        default="http://127.0.0.1:11434",
        help="Ollama server URL.",
    )
    parser.add_argument(
        "--skip_dataset_build",
        action="store_true",
        help="Do not regenerate the five synthetic image-text pairs.",
    )
    return parser.parse_args(argv)


def ensure_ready(client, model):
    """
    Verify that Ollama is running and the selected Llama model can be resolved.

    Parameters
    ----------
    client : OllamaClient
        Local Ollama wrapper.
    model : str
        Requested Ollama model name.

    Returns
    -------
    str
        Resolved installed model name.

    Raises
    ------
    OllamaError
        If the daemon is down or the model is missing.
    """
    if not client.is_available():
        raise OllamaError(
            "Ollama is not reachable at {}. Start it, then re-run python main.py.".format(client.host)
        )
    resolved = client.resolve_model(model)
    installed = [name.lower() for name in client.list_model_names()]

    def _is_installed(name):
        """Return True if an Ollama tag is present locally."""
        return name.lower() in installed or any(name.lower() in item for item in installed)

    if not _is_installed(resolved) and model.lower().startswith("llama3.2:3b"):
        print("llama3.2:3b is not installed; falling back to llama3.2:1b.", flush=True)
        model = "llama3.2:1b"
        resolved = client.resolve_model(model)
    if not _is_installed(resolved):
        raise OllamaError(
            "Model '{}' is not installed. Run: ollama pull {}".format(model, model)
        )
    return resolved


def run(args):
    """
    Execute the five-case multimodal pipeline and write JSON outputs.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    os.chdir(ROOT)
    if not args.skip_dataset_build:
        generate_all_cases(overwrite=False)

    client = OllamaClient(host=args.host)
    llama_model = ensure_ready(client, args.model)
    print("Using generative LLM :", llama_model, flush=True)
    try:
        encoders = preload_encoders()
        print("Using vision encoder :", encoders.get("vision_encoder"), flush=True)
        print("Using text encoder   : BioBERT", flush=True)
        print("Using orchestrator    : LangChain (prompts + finding retriever)", flush=True)
    except Exception as exc:
        print("WARNING: medical encoders failed to load:", exc, flush=True)
        print("Continuing with OpenCV + Llama only.", flush=True)
    if args.api_key:
        print("API key received as a parameter and will be ignored for local Ollama calls.", flush=True)

    records = []
    for case_id, image_path, text_path in list_input_pairs():
        if not os.path.exists(image_path) or not os.path.exists(text_path):
            print("Missing input for case", case_id, file=sys.stderr)
            return 1
        print("\n=== Processing patient_{:02d} ===".format(case_id), flush=True)
        record = process_case(
            client=client,
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
    print(json_pretty(summary), flush=True)
    return 0


def json_pretty(payload):
    """
    Pretty-print a JSON-serialisable object for the console.

    Parameters
    ----------
    payload : dict
        Metrics dictionary.

    Returns
    -------
    str
        Indented JSON string.
    """
    import json

    return json.dumps(payload, indent=2)


def main():
    """
    Command-line entry point used by `python main.py`.
    """
    args = parse_args()
    try:
        code = run(args)
    except OllamaError as exc:
        print("ERROR:", exc, file=sys.stderr)
        code = 2
    sys.exit(code)


if __name__ == "__main__":
    main()
