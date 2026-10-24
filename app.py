"""Streamlit application for transparent AI-interviewer evaluation."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st
from pydantic import ValidationError

from engine.evaluator import evaluate_answer
from engine.reference import generate_reference_answer
from evaluation.transcript import process_transcript

st.set_page_config(page_title="Interview Evaluator", page_icon="🎯", layout="wide")

st.title("🎯 Interview Evaluator")
st.caption("A transparent, deterministic baseline for structured interview feedback.")

with st.sidebar:
    st.header("About this evaluator")
    st.write(
        "Scores are based on explainable concept overlap and response structure. "
        "Use the evidence panels to review every result; this tool should support, "
        "not replace, human hiring decisions."
    )
    st.divider()
    st.metric("Scoring criteria", "4 × 10")

tab_single, tab_transcript = st.tabs(["Evaluate an answer", "Validate a transcript"])

with tab_single:
    st.subheader("Single-answer evaluation")
    question = st.text_area("Interview question", placeholder="e.g. What is dependency injection?")
    answer = st.text_area("Candidate answer", height=160)
    if "generated_reference" not in st.session_state:
        st.session_state.generated_reference = ""
    reference = st.text_area(
        "Reference answer / expected concepts (generated automatically)",
        value=st.session_state.generated_reference,
        height=160,
        help="Generated from the question. You may edit it before scoring.",
    )
    generate_clicked, evaluate_clicked = st.columns(2)
    with generate_clicked:
        if st.button("Generate reference answer"):
            try:
                generated = generate_reference_answer(question)
            except (RuntimeError, TypeError, ValueError) as error:
                st.error(str(error))
            else:
                st.session_state.generated_reference = str(generated["answer"])
                st.success(f"Reference generated using {generated['source']}.")
                st.rerun()
    with evaluate_clicked:
        should_evaluate = st.button("Evaluate answer", type="primary")
    if should_evaluate:
        try:
            if not reference.strip():
                generated = generate_reference_answer(question)
                reference = generated["answer"]
                st.session_state.generated_reference = reference
                st.info(
                    f"Reference answer generated automatically using {generated['source']}."
                )
            result = evaluate_answer(question, answer, reference)
        except (RuntimeError, TypeError, ValueError) as error:
            st.error(str(error))
        else:
            left, right = st.columns(2)
            with left:
                st.metric("Total score", f"{result['total_score']} / {result['maximum_score']}")
            with right:
                st.metric("Percentage", f"{result['percentage']}%")
            st.subheader("Scores")
            st.bar_chart(result["scores"], horizontal=True)
            st.subheader("Actionable feedback")
            for item in result["feedback"]:
                st.write(f"- {item}")
            with st.expander("Evidence and missing concepts"):
                st.write("Matched concepts:", ", ".join(result["matched_concepts"]) or "None")
                st.write("Missing concepts:", ", ".join(result["missing_concepts"]) or "None")

with tab_transcript:
    st.subheader("Transcript validation")
    st.write("Paste JSON matching `role`, `instructions`, and a `transcript` list.")
    example = {
        "role": "Backend Developer",
        "instructions": "Assess technical reasoning and communication.",
        "transcript": [
            {"speaker": "interviewer", "text": "How do you prevent duplicate payments?"},
            {"speaker": "candidate", "text": "I would use an idempotency key."},
        ],
    }
    raw_json = st.text_area("Transcript JSON", value=json.dumps(example, indent=2), height=300)
    if st.button("Validate transcript"):
        try:
            payload: Any = json.loads(raw_json)
            processed = process_transcript(payload)
        except json.JSONDecodeError as error:
            st.error(f"Invalid JSON: {error}")
        except (TypeError, ValidationError, ValueError) as error:
            st.error(f"Invalid transcript: {error}")
        else:
            st.success(f"Validated {processed['turn_count']} turns for {processed['role']}.")
            st.json(processed)
