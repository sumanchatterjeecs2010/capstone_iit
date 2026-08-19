"""
ollama_client.py
----------------
Thin HTTP client for local Ollama models.

Models are invoked one at a time and unloaded afterwards (keep_alive=0) so
that two LLMs can share a machine with about 4 GB of RAM.
"""

import json
import re
import time

import requests


class OllamaError(RuntimeError):
    """Raised when the local Ollama server cannot complete a request."""


class OllamaClient:
    """
    Minimal wrapper around the Ollama generate API.

    Parameters
    ----------
    host : str
        Base URL of the Ollama server (default: local daemon).
    timeout : int
        Per-request timeout in seconds.
    """

    def __init__(self, host="http://127.0.0.1:11434", timeout=180):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def is_available(self):
        """
        Return True if the Ollama daemon responds to /api/tags.

        Returns
        -------
        bool
            True when the server is reachable.
        """
        try:
            response = requests.get(self.host + "/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_model_names(self):
        """
        List model names currently available on the local Ollama instance.

        Returns
        -------
        list[str]
            Installed model names (may include tag suffixes).
        """
        response = requests.get(self.host + "/api/tags", timeout=10)
        response.raise_for_status()
        payload = response.json()
        return [item.get("name", "") for item in payload.get("models", [])]

    def resolve_model(self, requested):
        """
        Resolve a requested model name against installed Ollama tags.

        Ollama sometimes stores names with different capitalisation
        (for example llama3.2:3b vs Llama3.2:3b).

        Parameters
        ----------
        requested : str
            User-supplied or default model name.

        Returns
        -------
        str
            An installed model name, or the original request if none match.
        """
        installed = self.list_model_names()
        lowered = requested.lower()
        for name in installed:
            if name.lower() == lowered:
                return name
        # Allow prefix match: "llama3.2:3b" vs "llama3.2:3b-instruct-q4"
        for name in installed:
            if name.lower().startswith(lowered) or lowered.startswith(name.lower()):
                return name
        return requested

    def generate(self, model, prompt, system=None, temperature=0.2, max_tokens=400, keep_alive="0", json_mode=False):
        """
        Run a single non-streaming generation.

        Parameters
        ----------
        model : str
            Ollama model name.
        prompt : str
            User / task prompt.
        system : str or None
            Optional system instruction.
        temperature : float
            Decoding temperature (kept low for clinical structure).
        max_tokens : int
            Maximum number of tokens to generate.
        keep_alive : str or int
            How long Ollama should keep the weights in memory. Use 0 when
            switching to the other LLM so both models are never resident
            at once (needed for ~4 GB RAM).
        json_mode : bool
            If True, request a JSON object from Ollama's constrained decoder.

        Returns
        -------
        str
            Model text with optional <think> traces removed.
        """
        body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            body["format"] = "json"
        if system:
            body["system"] = system

        url = self.host + "/api/generate"
        last_error = None
        for attempt in range(3):
            try:
                response = requests.post(url, json=body, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
                # Ollama 0.32+ may split chain-of-thought into a "thinking" field
                # and keep the visible answer in "response".
                visible = payload.get("response") or ""
                thinking = payload.get("thinking") or ""
                cleaned = strip_thinking(visible)
                if not cleaned.strip():
                    cleaned = strip_thinking(thinking) or thinking or visible
                return cleaned.strip()
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
        raise OllamaError("Ollama generate failed for model '{}': {}".format(model, last_error))


def strip_thinking(text):
    """
    Remove chain-of-thought <think> blocks from a raw completion.

    Parameters
    ----------
    text : str
        Raw model output.

    Returns
    -------
    str
        Visible answer text only.
    """
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # If the model opened a think block but did not close it, drop that prefix.
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def extract_json(text):
    """
    Extract the first JSON object or array from a free-form LLM reply.

    Small models often wrap JSON in markdown fences or extra prose.

    Parameters
    ----------
    text : str
        Model output that is expected to contain JSON.

    Returns
    -------
    dict or list or None
        Parsed JSON, or None if parsing fails.
    """
    if not text:
        return None
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, (dict, list)):
            return parsed
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
        candidate = match.group(1) if match else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Try a lighter repair: keep only the outermost braces.
        try:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start != -1 and end > start:
                return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None
