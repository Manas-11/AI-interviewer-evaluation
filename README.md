# AI Interviewer Evaluation

A deployable Streamlit application for evaluating interview answers with a
transparent, deterministic scoring baseline. It validates transcript data with
Pydantic and scores answers across correctness, relevance, completeness, and
clarity (0–10 each), while exposing matched and missing concepts as evidence.
Reference answers are generated automatically from the question. If `NVIDIA_API_KEY` is configured, the app uses NVIDIA NIM; OpenAI remains a
secondary compatible provider, and the local evaluator is always available.

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

To enable model-generated reference answers and semantic grading with NVIDIA:

```powershell
$env:NVIDIA_API_KEY = "your-nvidia-api-key"
$env:NVIDIA_MODEL = "meta/llama-4-scout-17b-16e-instruct"
```

For Streamlit Community Cloud, open **App settings → Secrets** and add TOML
entries (do not commit this value to the repository):

```toml
NVIDIA_API_KEY = "your-nvidia-api-key"
NVIDIA_MODEL = "meta/llama-4-scout-17b-16e-instruct"
```

After saving the secrets, restart the app. The dashboard will show
`Semantic model grading is configured` when the key is being loaded. A valid NVIDIA API key with available credits is required. NVIDIA NIM is
OpenAI-compatible, but it uses the NVIDIA endpoint and model IDs.
The retired `meta/llama-3.3-70b-instruct` ID is automatically replaced with
`meta/llama-4-scout-17b-16e-instruct`.

The candidate answer can remain manually entered. Leave the reference-answer
field empty and it will be generated automatically from the question before
evaluation. Generated text remains editable so an interviewer can correct or
extend the rubric.

The local meaning-aware evaluator is always available and evaluates concept
coverage using phrase aliases, synonyms, fuzzy matching, and reference-derived
concept statements. NVIDIA provides the primary semantic pass, OpenAI is optional, and an
unavailable hosted model never prevents local evaluation from completing.

After evaluation, the dashboard provides a plain-language performance summary,
criterion-level deduction explanations, and concept evidence. Use **Start a
new evaluation** to clear the current form for the next candidate.

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
