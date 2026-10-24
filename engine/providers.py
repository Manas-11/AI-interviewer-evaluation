"""Small provider adapters used by reference generation and grading."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def active_provider() -> str | None:
    if os.getenv("GEMINI_API_KEY"):
        return "Gemini"
    if os.getenv("OPENAI_API_KEY"):
        return "OpenAI"
    return None


def gemini_generate(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    configured_model = os.getenv("GEMINI_MODEL", "").strip()
    models = _available_gemini_models(api_key, configured_model)
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 900},
    }).encode("utf-8")
    last_error = ""
    for model in models:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        request = Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=45) as response:
                data = json.loads(response.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except HTTPError as error:
            try:
                detail = error.read().decode("utf-8")[:300]
            except OSError:
                detail = "No response details were returned."
            last_error = f"HTTP {error.code} for {model}: {detail}"
            if error.code != 404:
                break
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "Gemini could not process the request. Check GEMINI_API_KEY, "
                "quota, and API access."
            ) from error
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("Gemini returned an empty or invalid response.") from error
    raise RuntimeError(
        f"Gemini could not use any discovered model ({last_error}). "
        "Check the API key's Gemini model access and quota."
    )


def _available_gemini_models(api_key: str, configured_model: str) -> list[str]:
    """Discover models enabled for this key instead of guessing model aliases."""

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models"
        f"?key={api_key}&pageSize=100"
    )
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
        discovered = [
            item["name"].removeprefix("models/")
            for item in data.get("models", [])
            if "generateContent" in item.get("supportedGenerationMethods", [])
        ]
    except HTTPError as error:
        try:
            detail = error.read().decode("utf-8")[:400]
        except OSError:
            detail = "No response details were returned."
        raise RuntimeError(
            f"Gemini model discovery failed with HTTP {error.code}: {detail}. "
            "This key may not be a Google AI Studio Gemini API key, or the "
            "Generative Language API may not be enabled."
        ) from error
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "Gemini model discovery could not connect. Check network access and API key."
        ) from error

    preferred = [
        configured_model,
        "gemini-3.8-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    ]
    selected = [model for model in preferred if model and model in discovered]
    if selected:
        return list(dict.fromkeys(selected))
    flash_models = [model for model in discovered if "flash" in model.lower()]
    if flash_models:
        return flash_models
    if discovered:
        return discovered
    raise RuntimeError(
        "Gemini returned no models that support generateContent for this API key. "
        "Create a Gemini API key in Google AI Studio and enable the Generative "
        "Language API."
    )
