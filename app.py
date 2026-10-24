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

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 8% 8%, rgba(91, 110, 225, .42), transparent 28rem),
            radial-gradient(circle at 92% 18%, rgba(193, 77, 214, .30), transparent 25rem),
            radial-gradient(circle at 72% 92%, rgba(25, 168, 145, .25), transparent 28rem),
            linear-gradient(135deg, #dfe9fb 0%, #e8e0f7 48%, #dff4ef 100%);
        background-attachment: fixed;
    }
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: .22;
        background-image: radial-gradient(rgba(16, 42, 67, .15) .7px, transparent .7px);
        background-size: 18px 18px;
        mask-image: linear-gradient(to bottom, black, transparent 78%);
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.2rem 2.4rem;
        border-radius: 24px;
        background:
            radial-gradient(circle at 88% 10%, rgba(255, 255, 255, .25), transparent 11rem),
            linear-gradient(120deg, #102a43 0%, #176b87 55%, #35b5a5 100%);
        color: white;
        box-shadow: 0 14px 35px rgba(16, 42, 67, .16);
        margin-bottom: 1.6rem;
    }
    .hero::after {
        content: "";
        position: absolute;
        width: 14rem;
        height: 14rem;
        right: -4rem;
        bottom: -8rem;
        border: 2rem solid rgba(255, 255, 255, .10);
        border-radius: 50%;
    }
    .hero-kicker {
        color: #9ee7dc;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .13em;
        text-transform: uppercase;
    }
    .hero h1 {
        margin: .45rem 0 .35rem;
        color: white;
        font-size: 2.35rem;
    }
    .hero p {
        color: #e5f4f5;
        font-size: 1.05rem;
        margin: 0;
        max-width: 720px;
    }
    .section-card {
        background: rgba(255, 255, 255, .78);
        border: 1px solid rgba(255, 255, 255, .85);
        backdrop-filter: blur(12px);
        border-radius: 18px;
        padding: 1.2rem 1.35rem .8rem;
        box-shadow: 0 8px 24px rgba(16, 42, 67, .05);
        margin-bottom: 1rem;
    }
    .score-card {
        background: linear-gradient(145deg, #173f5f 0%, #206b82 100%);
        border: 1px solid rgba(255, 255, 255, .25);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        min-height: 94px;
        box-shadow: 0 6px 18px rgba(16, 42, 67, .05);
    }
    .score-name {
        color: #bcebe7;
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
    }
    .score-value {
        color: #ffffff;
        font-size: 1.9rem;
        font-weight: 800;
        line-height: 1.25;
    }
    .score-value span {
        color: #829ab1;
        font-size: .9rem;
        font-weight: 500;
    }
    .hint {
        color: #263f58;
        font-size: .9rem;
    }
    .stMarkdown, .stMarkdown p, .stMarkdown li, label,
    [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] p {
        color: #172b40;
    }
    [data-testid="stTextArea"] label, [data-testid="stTextInput"] label {
        color: #102a43 !important;
        font-weight: 700 !important;
    }
    [data-testid="stTextArea"] textarea {
        background: rgba(255, 255, 255, .94);
        color: #102a43 !important;
        border: 1px solid rgba(16, 42, 67, .24);
        border-radius: 12px;
    }
    [data-testid="stTextArea"] textarea::placeholder {
        color: #627d98 !important;
        opacity: 1;
    }
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, .78);
        border: 1px solid rgba(16, 42, 67, .15);
        border-radius: 14px;
    }
    [data-testid="stAlert"] p {
        color: #102a43 !important;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #102a43 0%, #176b87 100%);
        border: 1px solid rgba(255, 255, 255, .28);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        box-shadow: 0 6px 18px rgba(16, 42, 67, .05);
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
        color: #bcebe7 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] div {
        color: #ffffff !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #d8f8f2 !important;
    }
    div[data-baseweb="tab-list"] {
        gap: .5rem;
        background: rgba(255, 255, 255, .55);
        border-radius: 14px;
        padding: .3rem;
    }
    button[data-baseweb="tab"] {
        border-radius: 10px;
    }
    </style>
    <div class="hero">
      <div class="hero-kicker">Interview intelligence · transparent evaluation</div>
      <h1>🎯 Interview Evaluator</h1>
      <p>Turn a question and candidate response into clear, evidence-backed feedback in seconds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🎯 Interview Evaluator")
    st.caption("A calm, explainable workspace for structured interview review.")
    st.divider()
    st.markdown("### How it works")
    st.write(
        "1. Add the question and candidate answer.\n"
        "2. Generate the reference answer.\n"
        "3. Review scores and evidence."
    )
    st.divider()
    st.metric("Scoring criteria", "4 × 10", help="Correctness, relevance, completeness, and clarity.")
    st.caption("Human review remains essential for hiring decisions.")

tab_single, tab_transcript = st.tabs(["Evaluate an answer", "Validate a transcript"])

with tab_single:
    st.markdown("### Evaluate a candidate answer")
    st.markdown('<p class="hint">Start with a question. The reference answer can be generated automatically and edited before scoring.</p>', unsafe_allow_html=True)
    question = st.text_area("Interview question", placeholder="e.g. What is dependency injection?", height=90)
    answer = st.text_area("Candidate answer", height=145, placeholder="Paste or type the candidate's response here...")
    if "generated_reference" not in st.session_state:
        st.session_state.generated_reference = ""
    reference = st.text_area(
        "Reference answer / expected concepts (generated automatically)",
        value=st.session_state.generated_reference,
        height=145,
        help="Generated from the question. You may edit it before scoring.",
    )
    generate_clicked, evaluate_clicked = st.columns([1, 1])
    with generate_clicked:
        if st.button("✨ Generate reference answer", use_container_width=True):
            try:
                generated = generate_reference_answer(question)
            except (RuntimeError, TypeError, ValueError) as error:
                st.error(str(error))
            else:
                st.session_state.generated_reference = str(generated["answer"])
                if generated.get("warning"):
                    st.warning(
                        f"{generated['warning']} Using the local reference generator instead."
                    )
                st.success(f"Reference generated using {generated['source']}.")
                st.rerun()
    with evaluate_clicked:
        should_evaluate = st.button("🚀 Evaluate answer", type="primary", use_container_width=True)
    if should_evaluate:
        try:
            if not reference.strip():
                generated = generate_reference_answer(question)
                reference = generated["answer"]
                st.session_state.generated_reference = reference
                if generated.get("warning"):
                    st.warning(
                        f"{generated['warning']} Using the local reference generator instead."
                    )
                st.info(
                    f"Reference answer generated automatically using {generated['source']}."
                )
            result = evaluate_answer(question, answer, reference)
        except (RuntimeError, TypeError, ValueError) as error:
            st.error(str(error))
        else:
            st.markdown("### Evaluation snapshot")
            summary = st.columns(2 + len(result["scores"]))
            summary[0].metric("Overall", f"{result['total_score']}/{result['maximum_score']}")
            summary[1].metric("Percentage", f"{result['percentage']}%")
            for index, (criterion, score) in enumerate(result["scores"].items(), start=2):
                summary[index].markdown(
                    f'<div class="score-card"><div class="score-name">{criterion}</div>'
                    f'<div class="score-value">{score}<span> / 10</span></div></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("### Score breakdown")
            st.bar_chart(result["scores"], horizontal=True, height=220)
            st.markdown("### Actionable feedback")
            for item in result["feedback"]:
                st.markdown(f"✅ {item}")
            with st.expander("Evidence and missing concepts"):
                evidence_left, evidence_right = st.columns(2)
                evidence_left.markdown("**Matched concepts**")
                evidence_left.info(", ".join(result["matched_concepts"]) or "None")
                evidence_right.markdown("**Missing concepts**")
                evidence_right.warning(", ".join(result["missing_concepts"]) or "None")

with tab_transcript:
    st.markdown("### Validate an interview transcript")
    st.markdown('<p class="hint">Check transcript structure before sending it into a broader evaluation workflow.</p>', unsafe_allow_html=True)
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
