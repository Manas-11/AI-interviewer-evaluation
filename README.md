# AI Interviewer Evaluation

A deployable Streamlit application for evaluating interview answers with a
transparent, deterministic scoring baseline. It validates transcript data with
Pydantic and scores answers across correctness, relevance, completeness, and
clarity (0–10 each), while exposing matched and missing concepts as evidence.
Reference answers are generated automatically from the question. If
`OPENAI_API_KEY` is configured, the app uses the selected OpenAI model;
otherwise it uses a deterministic local fallback.

The dashboard is focused on the primary workflow: evaluating one question and
one candidate answer at a time. Transcript validation remains available in the
`evaluation` package for future full-interview workflows but is not shown in
the current UI.

## Why this design

The application does not silently call an external AI provider or require an
API key. This makes local runs reproducible and keeps candidate data private.
The evaluator is deliberately a baseline: it is useful for triage and
consistent feedback, but hiring decisions must include human review.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the URL printed by Streamlit, then use **Evaluate an answer**.

To enable higher-quality model-generated reference answers, set:

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:OPENAI_MODEL = "gpt-4o-mini"
```

The candidate answer can remain manually entered. Leave the reference-answer
field empty and it will be generated automatically from the question before
evaluation. Generated text remains editable so an interviewer can correct or
extend the rubric.

## Test and CLI smoke check

```powershell
python -m pytest -q
python -m py_compile app.py engine\evaluator.py evaluation\transcript.py
```

The core library can also be imported directly:

```python
from engine.evaluator import evaluate_answer

result = evaluate_answer(
    "What is dependency injection?",
    "It supplies dependencies from outside a class.",
    "Dependency injection supplies a class's dependencies from outside.",
)
```

## Input contract

Transcript JSON must contain `role`, `instructions`, and a non-empty
`transcript` array. Each turn has a `speaker` of `interviewer` or `candidate`
and non-empty `text`. Unknown fields are rejected to catch malformed data
early.

## Deployment

Deploy on Streamlit Community Cloud, Render, Railway, or any Docker-capable
host using the start command:

```text
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

The project has no database or secret requirement. For a production hiring
workflow, add authentication, encrypted storage, retention controls, audit
logging, and a human-review gate before storing candidate data.
