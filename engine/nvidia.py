"""NVIDIA NIM configuration helpers."""

from __future__ import annotations

import os
from functools import lru_cache

from openai import APIError, OpenAI

DEFAULT_MODEL = "meta/llama-4-scout-17b-16e-instruct"
RETIRED_MODELS = {"meta/llama-3.3-70b-instruct"}
FALLBACK_MODELS = (
    "meta/llama-3.1-8b-instruct",
    "meta/llama3-8b-instruct",
    "meta/llama3-70b-instruct",
)


def model_name() -> str:
    configured = os.getenv("NVIDIA_MODEL", "").strip()
    if configured in RETIRED_MODELS or not configured:
        return DEFAULT_MODEL
    return configured


@lru_cache(maxsize=4)
def model_candidates(api_key: str) -> tuple[str, ...]:
    configured = model_name()
    discovered: list[str] = []
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1",
            timeout=8.0,
            max_retries=0,
        )
        discovered = [
            item.id for item in client.models.list().data
            if item.id not in RETIRED_MODELS
        ]
    except (APIError, OSError, TypeError, ValueError):
        discovered = []
    preferred = [configured, *FALLBACK_MODELS]
    candidates = list(dict.fromkeys(
        model for model in [*preferred, *discovered]
        if model and model not in RETIRED_MODELS
    ))
    return tuple(candidates)


def nvidia_chat_completion(api_key: str, **kwargs):
    """Call an available NVIDIA model, retrying only model-not-found errors."""

    last_error: Exception | None = None
    for model in model_candidates(api_key):
        try:
            client = OpenAI(
                api_key=api_key,
                base_url="https://integrate.api.nvidia.com/v1",
                timeout=12.0,
                max_retries=0,
            )
            return client.chat.completions.create(model=model, **kwargs)
        except APIError as error:
            last_error = error
            if getattr(error, "status_code", None) not in (404, 410):
                raise RuntimeError(
                    "NVIDIA request failed before a response was received. "
                    "The local meaning-aware evaluation was used instead."
                ) from error
    raise RuntimeError(
        "NVIDIA has no available model for this API key. "
        "Update NVIDIA_MODEL to an active model in the NVIDIA catalog."
    ) from last_error
