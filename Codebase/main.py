"""
main.py
-------
Command-line entry point for the Multimodal Medical Assistant (HPPCS[04]).

Typical usage (run from the Codebase/ directory)::

    python main.py --text note.txt --image scan.jpg
    python main.py --text note.txt --image scan.jpg --device gpu
    python main.py --tui

Flow
----
1. Resolve CPU/GPU via ``runtime_profile``.
2. Connect to local Ollama and resolve MedGemma + Llama model tags.
3. Either run a one-shot CLI case (``--text`` + ``--image``) or open the TUI.
4. Persist the full case record as ``conversation.json`` (see ``paths.CONVERSATION_PATH``).

Exit codes: 0 success, 1 usage/input error, 2 Ollama/model error.
See ``execution.txt`` for setup and demo commands.
"""

import argparse
import os
import sys

from medical_assistant import (
    default_conversation_path,
    display_payload,
    get_graph,
    process_case,
    save_json,
)
from ollama_client import OllamaClient, OllamaError
from runtime_profile import apply_ollama_env, describe_startup, get_profile


ROOT = os.path.dirname(os.path.abspath(__file__))
# Fallback tags if the requested MedGemma name is not installed locally.
VISION_ALIASES = ["medgemma:4b", "medgemma", "medgemma:latest", "medgemma:1.5"]


def parse_args(argv=None):
    """Parse CLI flags. Returns an argparse namespace."""
    parser = argparse.ArgumentParser(
        description="Multimodal Medical Assistant - CLI (note + image -> conversation.json)"
    )
    parser.add_argument("--model", default="llama3.2:3b", help="Text LLM (default llama3.2:3b)")
    parser.add_argument("--vision_model", default="medgemma:4b", help="Vision LLM (default medgemma:4b)")
    parser.add_argument("--host", default="http://127.0.0.1:11434", help="Ollama URL")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Compute device: auto (default), cpu, or gpu",
    )
    parser.add_argument("--text", default="", help="Path to clinical note (.txt / .md)")
    parser.add_argument("--image", default="", help="Path to radiology/pathology image (.jpg / .png / .dcm)")
    parser.add_argument(
        "--output",
        default="",
        help="Output conversation JSON path (default: Codebase/conversation.json)",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Interactive terminal UI (prompts for paths if --text/--image omitted)",
    )
    return parser.parse_args(argv)


def _is_installed(client, name):
    """Return True if *name* matches an installed Ollama model tag."""
    installed = [item.lower() for item in client.list_model_names()]
    lowered = name.lower()
    return lowered in installed or any(lowered in item for item in installed)


def ensure_llama(client, model):
    """
    Resolve the text LLM, falling back from llama3.2:3b to :1b on low-RAM hosts.

    Raises
    ------
    OllamaError
        If no matching Llama model is installed.
    """
    resolved = client.resolve_model(model)
    if not _is_installed(client, resolved) and model.lower().startswith("llama3.2:3b"):
        print("llama3.2:3b missing; using llama3.2:1b", flush=True)
        model = "llama3.2:1b"
        resolved = client.resolve_model(model)
    if not _is_installed(client, resolved):
        raise OllamaError("Model '{}' not installed. Run: ollama pull {}".format(model, model))
    return resolved


def ensure_vision(client, model):
    """
    Resolve a MedGemma vision model, trying ``VISION_ALIASES`` if needed.

    Raises
    ------
    OllamaError
        If no MedGemma tag is installed.
    """
    for candidate in [model] + [a for a in VISION_ALIASES if a != model]:
        resolved = client.resolve_model(candidate)
        if _is_installed(client, resolved):
            if resolved != model:
                print("Using vision model:", resolved, flush=True)
            return resolved
    raise OllamaError("MedGemma not installed. Run: ollama pull medgemma:4b")


def setup_models(args):
    """
    Apply the runtime profile, verify Ollama, and return resolved models.

    Returns
    -------
    tuple
        ``(client, vision_model, llama_model, profile)``
    """
    profile = get_profile(args.device)
    apply_ollama_env(profile)
    print(describe_startup(profile), flush=True)
    if args.device == "gpu" and profile.device == "cpu":
        print("No NVIDIA GPU detected; falling back to CPU.", flush=True)
    client = OllamaClient(host=args.host)
    if not client.is_available():
        raise OllamaError("Ollama not reachable at {}.".format(client.host))
    vision = ensure_vision(client, args.vision_model)
    llama = ensure_llama(client, args.model)
    return client, vision, llama, profile


def resolve_output_path(args):
    """Absolute path for conversation.json (``--output`` or Codebase default)."""
    if args.output:
        return os.path.abspath(os.path.expanduser(args.output))
    return default_conversation_path()


def run_cli_case(args, client, vision_model, llama_model):
    """
    Run one note+image case, print the summary, then enter follow-up chat.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    text_path = os.path.abspath(os.path.expanduser(args.text))
    image_path = os.path.abspath(os.path.expanduser(args.image))
    if not os.path.isfile(text_path):
        print("Note file not found: {}".format(text_path), file=sys.stderr)
        return 1
    if not os.path.isfile(image_path):
        print("Image file not found: {}".format(image_path), file=sys.stderr)
        return 1

    out_path = resolve_output_path(args)
    print("Vision:", vision_model, "| Language:", llama_model, flush=True)
    print("Note:", text_path, flush=True)
    print("Image:", image_path, flush=True)
    get_graph(client)

    record = process_case(
        client, vision_model, llama_model, 0, image_path, text_path
    )
    save_json(out_path, record)
    print("Wrote", out_path, flush=True)

    from tui_app import render_record_summary, run_follow_up_chat

    render_record_summary(display_payload(record))
    return run_follow_up_chat(client, vision_model, llama_model, out_path)


def run(args):
    """Dispatch to TUI or CLI after model setup. Returns an exit code."""
    os.chdir(ROOT)
    client, vision, llama, profile = setup_models(args)
    os.environ["OLLAMA_HOST"] = args.host
    os.environ["VISION_MODEL"] = vision
    os.environ["LLAMA_MODEL"] = llama
    os.environ["ASSISTANT_DEVICE"] = profile.device

    if args.tui:
        from tui_app import run_tui

        return run_tui(args, client, vision, llama)

    if args.text and args.image:
        return run_cli_case(args, client, vision, llama)

    print(
        "Usage:\n"
        "  python main.py --text <note.txt> --image <scan.jpg>\n"
        "  python main.py --text <note.txt> --image <scan.jpg> --device gpu\n"
        "  python main.py --tui\n"
        "Output defaults to Codebase/conversation.json",
        file=sys.stderr,
    )
    return 1


def main():
    """Parse args, run the assistant, and exit with a status code."""
    args = parse_args()
    try:
        code = run(args)
    except OllamaError as exc:
        print("ERROR:", exc, file=sys.stderr)
        code = 2
    except ValueError as exc:
        print("ERROR:", exc, file=sys.stderr)
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
