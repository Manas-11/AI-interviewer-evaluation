"""Reference-answer generation with an optional hosted model and local fallback."""

from __future__ import annotations

import os
import re
from typing import Any

from engine.nvidia import nvidia_chat_completion

def _clean_question(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")
    return question.strip()


def _local_reference(question: str) -> str:
    """Create a useful rubric when no model credentials are available."""

    normalized = question.lower()
    known_answers = {
        "list and tuple": (
            "A list is a mutable, ordered collection, while a tuple is an "
            "immutable, ordered collection. Lists use square brackets and "
            "support updates; tuples use parentheses and are suitable for "
            "fixed data. Both preserve order and can contain duplicate values."
        ),
        "dependency injection": (
            "Dependency injection supplies a class or function with its "
            "dependencies from outside instead of constructing them internally. "
            "This reduces coupling, improves testability, and makes "
            "implementations replaceable."
        ),
        "duplicate payment": (
            "Use an idempotency key supplied by the client and enforce a "
            "unique constraint for it in durable storage. Process the request "
            "transactionally, save the first result, and return that same "
            "result for retries. Handle concurrent requests safely."
        ),
        "caching": (
            "Caching stores frequently or expensively computed data closer to "
            "the consumer so later reads are faster. A good answer should "
            "discuss cache keys, TTL or invalidation, stale data, capacity, "
            "and the trade-off between latency and consistency."
        ),
    }
    for phrase, answer in known_answers.items():
        if phrase in normalized:
            return answer

    question_terms = re.findall(r"[A-Za-z][A-Za-z0-9+#.-]{2,}", question)
    topics = ", ".join(dict.fromkeys(question_terms[:8])) or "the requested topic"
    return (
        f"A strong answer to this question should define {topics}, explain how "
        "it works, and connect it to the question's practical context. It "
        "should mention the main benefits, limitations or trade-offs, and "
        "include a concise example where appropriate. Evaluate the candidate "
        "on technical accuracy, directness, completeness, and clarity."
    )


def _openai_reference(question: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import APIError, OpenAI

        client = OpenAI(api_key=api_key, timeout=12.0, max_retries=0)
        response: Any = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            input=(
                "Create a concise, technically accurate reference answer for "
                "an interview evaluator. Include the essential concepts, "
                "trade-offs, and an example when relevant. Do not grade the "
                f"candidate. Question: {question}"
            ),
            temperature=0.2,
            max_output_tokens=300,
        )
        answer = response.output_text.strip()
        return answer or None
    except ImportError as error:
        raise RuntimeError(
            "The OpenAI package is not installed. Remove OPENAI_API_KEY or redeploy "
            "after installing requirements.txt."
        ) from error
    except (APIError, TypeError, ValueError) as error:
        raise RuntimeError(
            "The configured OpenAI model could not generate a reference answer. "
            "Check OPENAI_API_KEY, OPENAI_MODEL, and account access."
        ) from error


def _nvidia_reference(question: str) -> str | None:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        return None
    try:
        response: Any = nvidia_chat_completion(
            api_key,
            messages=[{
                "role": "user",
                "content": (
                    "Create a concise, technically accurate reference answer for "
                    "an interview evaluator. Include essential concepts, trade-offs, "
                    f"and an example when relevant. Question: {question}"
                ),
            }],
            temperature=0.2,
            max_tokens=220,
        )
        answer = response.choices[0].message.content.strip()
        return answer or None
    except (ImportError, IndexError, AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError(
            "NVIDIA reference generation failed. Check NVIDIA_API_KEY, "
            "NVIDIA_MODEL, and NVIDIA API credits."
        ) from error


def generate_reference_answer(question: str) -> dict[str, str | bool]:
    """Generate a reference answer from a question.

    Returns the answer and whether it came from a hosted model. The local
    fallback is intentionally explicit so users can distinguish it from
    model-generated content.
    """

    question = _clean_question(question)
    if os.getenv("NVIDIA_API_KEY"):
        try:
            answer = _nvidia_reference(question)
            if answer:
                return {"answer": answer, "source": "NVIDIA", "is_model_generated": True}
        except RuntimeError as error:
            nvidia_warning = str(error)
        else:
            nvidia_warning = "NVIDIA returned an empty answer."
    else:
        nvidia_warning = ""
    if os.getenv("OPENAI_API_KEY"):
        try:
            answer = _openai_reference(question)
            if answer:
                return {"answer": answer, "source": "OpenAI", "is_model_generated": True}
        except RuntimeError as error:
            openai_warning = str(error)
        else:
            openai_warning = "OpenAI returned an empty answer."
    else:
        openai_warning = ""
    warning_parts = [part for part in (
        f"NVIDIA unavailable: {nvidia_warning}" if nvidia_warning else "",
        f"OpenAI unavailable: {openai_warning}" if openai_warning else "",
    ) if part]
    fallback = {
        "answer": _local_reference(question),
        "source": "Local fallback",
        "is_model_generated": False,
    }
    if warning_parts:
        fallback["warning"] = "; ".join(warning_parts)
    return fallback
