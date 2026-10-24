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
    models = list(dict.fromkeys(
        model for model in (
            configured_model,
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
        ) if model
    ))
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
        f"Gemini could not find an available model ({last_error}). Set GEMINI_MODEL "
        "to a model enabled for this API key, such as gemini-2.5-flash."
    )
