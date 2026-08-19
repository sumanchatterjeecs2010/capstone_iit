"""
ollama_client.py
----------------
HTTP client for local Ollama models.

MedGemma is called through /api/chat with an image. Llama 3.2 is called
through the same chat API with text only. Models are unloaded after each
call (keep_alive=0) so the two LLMs can share a machine with limited RAM.
"""

import base64
import json
import re
import time

import requests


class OllamaError(RuntimeError):
    """Raised when the local Ollama server cannot complete a request."""


class OllamaClient:
    """
    Minimal wrapper around Ollama chat and tags APIs.

    Parameters
    ----------
    host : str
        Base URL of the Ollama server.
    timeout : int
        Per-request timeout in seconds.
    """

    def __init__(self, host="http://127.0.0.1:11434", timeout=1800):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def is_available(self):
        """Return True if the Ollama daemon responds to /api/tags."""
        try:
            response = requests.get(self.host + "/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_model_names(self):
        """Return installed Ollama model names."""
        response = requests.get(self.host + "/api/tags", timeout=10)
        response.raise_for_status()
        payload = response.json()
        return [item.get("name", "") for item in payload.get("models", [])]

    def resolve_model(self, requested):
        """
        Resolve a requested model name against installed Ollama tags.

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
        for name in installed:
            if name.lower().startswith(lowered) or lowered.startswith(name.lower()):
                return name
        needle = lowered.split("/")[-1].split(":")[0]
        for name in installed:
            if needle and needle in name.lower():
                return name
        return requested

    def chat(
        self,
        model,
        prompt,
        system=None,
        images=None,
        temperature=0.2,
        max_tokens=400,
        keep_alive="0",
        json_mode=False,
    ):
        """
        Run a single non-streaming chat completion.

        Parameters
        ----------
        model : str
            Ollama model name.
        prompt : str
            User / task prompt.
        system : str or None
            Optional system instruction.
        images : list[str] or None
            Optional local image paths for a vision LLM.
        temperature : float
            Decoding temperature.
        max_tokens : int
            Maximum generated tokens.
        keep_alive : str or int
            How long Ollama should keep weights loaded.
        json_mode : bool
            If True, request a JSON object from Ollama.

        Returns
        -------
        str
            Visible model text.
        """
        message = {"role": "user", "content": prompt}
        if images:
            encoded = []
            for path in images:
                with open(path, "rb") as handle:
                    encoded.append(base64.b64encode(handle.read()).decode("ascii"))
            message["images"] = encoded

        body = {
            "model": model,
            "messages": [message],
            "stream": False,
            "keep_alive": keep_alive,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            body["messages"].insert(0, {"role": "system", "content": system})
        if json_mode:
            body["format"] = "json"

        url = self.host + "/api/chat"
        last_error = None
        attempts = 2 if images else 3
        for attempt in range(attempts):
            try:
                response = requests.post(url, json=body, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
                message_out = payload.get("message") or {}
                visible = message_out.get("content") or payload.get("response") or ""
                thinking = message_out.get("thinking") or payload.get("thinking") or ""
                cleaned = strip_thinking(visible)
                if not cleaned.strip():
                    cleaned = strip_thinking(thinking) or thinking or visible
                return cleaned.strip()
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
        raise OllamaError("Ollama chat failed for model '{}': {}".format(model, last_error))


def strip_thinking(text):
    """Remove chain-of-thought <think> blocks from a raw completion."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def extract_json(text):
    """
    Extract the first JSON object or array from a free-form LLM reply.

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
        try:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start != -1 and end > start:
                return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None
