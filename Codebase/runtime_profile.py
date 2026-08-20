"""runtime_profile.py — Adaptive CPU/GPU settings for Ollama-backed inference.

Ollama already uses a GPU when one is present. This module detects the host,
tunes keep-alive / unload policy so MedGemma and Llama share memory safely on
a 16 GB CPU laptop, and keeps models warmer on GPU (Colab T4 or local CUDA).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeProfile:
    """Memory and keep-alive policy for the current host."""

    device: str  # "gpu" | "cpu"
    label: str
    vision_keep_alive: str
    llama_keep_alive: str
    unload_vision_after_analysis: bool
    request_timeout: int

    @property
    def is_gpu(self) -> bool:
        return self.device == "gpu"


_PROFILE: RuntimeProfile | None = None


def _nvidia_smi_ok() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.returncode == 0 and bool((result.stdout or "").strip())
    except (OSError, subprocess.SubprocessError):
        return False


def detect_device(preference="auto"):
    """
    Resolve compute device.

    preference: auto | cpu | gpu
      auto — use NVIDIA GPU when nvidia-smi lists one; otherwise CPU
      cpu  — force CPU (Ollama num_gpu=0)
      gpu  — prefer GPU; fall back to CPU with a warning if none found
    """
    pref = (preference or "auto").strip().lower()
    has_gpu = _nvidia_smi_ok()
    if pref == "cpu":
        return "cpu"
    if pref == "gpu":
        return "gpu" if has_gpu else "cpu"
    return "gpu" if has_gpu else "cpu"


def build_profile(preference="auto") -> RuntimeProfile:
    device = detect_device(preference)
    if device == "gpu":
        return RuntimeProfile(
            device="gpu",
            label="GPU (CUDA / Ollama offload)",
            vision_keep_alive="2m",
            llama_keep_alive="15m",
            unload_vision_after_analysis=True,  # free VRAM before Llama chat
            request_timeout=900,
        )
    return RuntimeProfile(
        device="cpu",
        label="CPU (16 GB-class laptop / Colab CPU)",
        vision_keep_alive="0",  # unload MedGemma immediately after vision
        llama_keep_alive="10m",
        unload_vision_after_analysis=True,
        request_timeout=1800,
    )


def get_profile(preference=None) -> RuntimeProfile:
    """Return cached profile; rebuild if preference is given."""
    global _PROFILE
    if preference is not None or _PROFILE is None:
        pref = preference if preference is not None else os.environ.get("ASSISTANT_DEVICE", "auto")
        _PROFILE = build_profile(pref)
        os.environ["ASSISTANT_DEVICE"] = _PROFILE.device
        os.environ["ASSISTANT_RUNTIME_LABEL"] = _PROFILE.label
    return _PROFILE


def apply_ollama_env(profile: RuntimeProfile) -> None:
    """Hint Ollama to use GPU or stay on CPU without changing API callers."""
    if profile.device == "cpu":
        # Hide GPUs from the process when the user forces CPU (or no GPU found).
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["OLLAMA_NUM_GPU"] = "0"
    else:
        os.environ.pop("OLLAMA_NUM_GPU", None)
        # Do not clear a user-set CUDA_VISIBLE_DEVICES (e.g. Colab multi-GPU).
        if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)


def describe_startup(profile: RuntimeProfile) -> str:
    return (
        "Runtime: {} | vision keep_alive={} | llama keep_alive={} | "
        "unload vision after analysis={}"
    ).format(
        profile.label,
        profile.vision_keep_alive,
        profile.llama_keep_alive,
        profile.unload_vision_after_analysis,
    )
