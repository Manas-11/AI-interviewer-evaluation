"""NVIDIA NIM configuration helpers."""

from __future__ import annotations

import os

DEFAULT_MODEL = "meta/llama-4-scout-17b-16e-instruct"
RETIRED_MODELS = {"meta/llama-3.3-70b-instruct"}


def model_name() -> str:
    configured = os.getenv("NVIDIA_MODEL", "").strip()
    if configured in RETIRED_MODELS or not configured:
        return DEFAULT_MODEL
    return configured
