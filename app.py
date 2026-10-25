"""Streamlit application for transparent AI-interviewer evaluation."""

from __future__ import annotations

import html
import os

import altair as alt
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from engine.evaluator import evaluate_answer_with_model
from engine.reference import generate_reference_answer

st.set_page_config(page_title="Interview Evaluator", layout="wide")


def _load_model_settings() -> bool:
    """Load hosted-model settings from Streamlit secrets without exposing keys."""

    if os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY"):
        return True
    try:
        secret_key = st.secrets.get("NVIDIA_API_KEY")
        if secret_key:
            os.environ["NVIDIA_API_KEY"] = str(secret_key).strip()
            secret_model = st.secrets.get("NVIDIA_MODEL")
            if secret_model:
                os.environ["NVIDIA_MODEL"] = str(secret_model).strip()
            return True
    except StreamlitSecretNotFoundError:
        pass
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY")
        if secret_key:
            os.environ["OPENAI_API_KEY"] = str(secret_key).strip()
            secret_model = st.secrets.get("OPENAI_MODEL")
            if secret_model:
                os.environ["OPENAI_MODEL"] = str(secret_model).strip()
            return True
    except StreamlitSecretNotFoundError:
        pass
    return False


model_configured = _load_model_settings()


def _performance_summary(percentage: float) -> tuple[str, str]:
    if percentage >= 85:
        return "Strong response", "The answer covers the expected ideas and communicates them clearly."
    if percentage >= 70:
        return "Good foundation", "The answer is on the right track, with a few concepts or details to strengthen."
    if percentage >= 50:
        return "Needs development", "The answer shows partial understanding but needs more complete reasoning."
    return "Needs significant improvement", "The response needs stronger technical coverage and clearer explanation."


def _reset_evaluation() -> None:
    """Clear all evaluation widget state before the next app render."""

    for key in list(st.session_state):
        if key.startswith(("question_input_", "candidate_answer_input_", "reference_input_")):
            del st.session_state[key]
    st.session_state.evaluation_form_id = st.session_state.get("evaluation_form_id", 0) + 1
    st.session_state.generated_reference = ""

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > section,
    [data-testid="stAppViewContainer"] .main, .main, .block-container {
        background:
            radial-gradient(circle at 8% 8%, rgba(91, 110, 225, .42), transparent 28rem),
            radial-gradient(circle at 92% 18%, rgba(193, 77, 214, .30), transparent 25rem),
            radial-gradient(circle at 72% 92%, rgba(25, 168, 145, .25), transparent 28rem),
            linear-gradient(135deg, #dfe9fb 0%, #e8e0f7 48%, #dff4ef 100%);
        background-size: 140% 140%;
        animation: ambient-shift 18s ease-in-out infinite alternate;
        background-attachment: fixed;
        color: #172b40;
        scroll-behavior: smooth;
        overscroll-behavior-y: none;
        -webkit-overflow-scrolling: touch;
    }
    [data-testid="stAppViewContainer"] > section,
    [data-testid="stAppViewContainer"] .main,
    .main {
        scroll-behavior: smooth;
        scrollbar-width: thin;
        scrollbar-color: rgba(23, 107, 135, .65) rgba(255, 255, 255, .35);
    }
    [data-testid="stAppViewContainer"] > section::-webkit-scrollbar,
    [data-testid="stAppViewContainer"] .main::-webkit-scrollbar,
    .main::-webkit-scrollbar {
        width: 10px;
    }
    [data-testid="stAppViewContainer"] > section::-webkit-scrollbar-track,
    [data-testid="stAppViewContainer"] .main::-webkit-scrollbar-track,
    .main::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, .28);
        border-radius: 99px;
    }
    [data-testid="stAppViewContainer"] > section::-webkit-scrollbar-thumb,
    [data-testid="stAppViewContainer"] .main::-webkit-scrollbar-thumb,
    .main::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #176b87, #35b5a5);
        border: 2px solid rgba(255, 255, 255, .35);
        border-radius: 99px;
    }
    [data-testid="stAppViewContainer"] {
        min-height: 100vh;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 1180px;
        transition: padding .25s ease;
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
        box-shadow: 0 16px 38px rgba(16, 42, 67, .18);
        margin-bottom: 1.6rem;
        transition: transform .3s ease, box-shadow .3s ease;
    }
    .hero:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 44px rgba(16, 42, 67, .22);
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
        animation: float-orb 7s ease-in-out infinite;
    }
    .hero::before {
        content: "";
        position: absolute;
        width: 8rem;
        height: 8rem;
        right: 16%;
        top: -5rem;
        background: rgba(158, 231, 220, .13);
        border-radius: 50%;
        filter: blur(1px);
        animation: float-orb 9s ease-in-out infinite reverse;
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
    .workflow-heading {
        color: #102a43;
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -.01em;
        margin: .5rem 0 .35rem;
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
        padding: 1.25rem 1.35rem;
        min-height: 132px;
        margin: .35rem .45rem 1.15rem;
        box-shadow: 0 8px 20px rgba(16, 42, 67, .12);
        transition: transform .2s ease, box-shadow .2s ease;
    }
    .score-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 26px rgba(16, 42, 67, .18);
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
    .score-description {
        color: #d8f8f2;
        font-size: .78rem;
        line-height: 1.35;
        margin-top: .45rem;
    }
    .score-bar {
        background: rgba(255, 255, 255, .2);
        border-radius: 99px;
        height: 7px;
        margin-top: .8rem;
        overflow: hidden;
    }
    .score-fill {
        background: linear-gradient(90deg, #9ee7dc, #ffffff);
        border-radius: 99px;
        height: 100%;
        transform-origin: left;
        animation: score-grow .75s ease-out both;
    }
    .criterion-heading {
        margin-top: 1.8rem;
        margin-bottom: .8rem;
        color: #102a43;
    }
    .chart-heading {
        margin-top: 1.8rem;
        margin-bottom: .4rem;
        color: #102a43;
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
        font-size: 1.08rem !important;
        font-weight: 800 !important;
        letter-spacing: .01em;
        line-height: 1.35 !important;
        margin-bottom: .35rem !important;
    }
    [data-testid="stTextArea"] label p {
        font-size: 1.08rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stTextArea"] textarea {
        background: rgba(255, 255, 255, .94);
        color: #102a43 !important;
        border: 1px solid rgba(16, 42, 67, .24);
        border-radius: 12px;
        transition: border-color .2s ease, box-shadow .2s ease, background .2s ease;
    }
    [data-testid="stTextArea"] textarea:focus {
        background: #ffffff;
        border-color: #35b5a5 !important;
        box-shadow: 0 0 0 3px rgba(53, 181, 165, .20) !important;
        outline: none;
    }
    [data-testid="stTextArea"] textarea::placeholder {
        color: #627d98 !important;
        opacity: 1;
    }
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, .78);
        border: 1px solid rgba(16, 42, 67, .15);
        border-radius: 14px;
        transition: background .2s ease, border-color .2s ease, box-shadow .2s ease;
    }
    [data-testid="stExpander"]:hover {
        background: rgba(255, 255, 255, .9);
        border-color: rgba(23, 107, 135, .30);
        box-shadow: 0 8px 20px rgba(16, 42, 67, .10);
    }
    .evidence-card {
        border-radius: 14px;
        padding: 1rem 1.1rem;
        min-height: 115px;
        box-shadow: 0 7px 18px rgba(16, 42, 67, .10);
        transition: transform .2s ease, box-shadow .2s ease;
    }
    .evidence-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 28px rgba(16, 42, 67, .20);
    }
    .evidence-card h5 {
        margin: 0 0 .7rem;
        font-size: .85rem;
        letter-spacing: .06em;
        text-transform: uppercase;
    }
    .evidence-card p {
        margin: .45rem 0;
        line-height: 1.45;
        font-size: .9rem;
    }
    .concept-item {
        border-top: 1px solid rgba(255, 255, 255, .22);
        cursor: help;
        padding: .55rem .2rem .35rem;
        position: relative;
    }
    .concept-item:first-of-type {
        border-top: 0;
    }
    .concept-detail {
        color: rgba(16, 59, 74, .82);
        display: block;
        font-size: .78rem;
        line-height: 1.35;
        margin-top: .25rem;
    }
    .concept-item:hover .concept-detail {
        color: #102a43;
    }
    .matched-evidence {
        background: linear-gradient(145deg, #5ba9b2 0%, #75c5bf 100%);
        border: 1px solid rgba(255, 255, 255, .7);
        color: #103b4a;
    }
    .matched-evidence h5 {
        color: #103b4a;
    }
    .missing-evidence {
        background: linear-gradient(145deg, #7399c4 0%, #8bb8d2 100%);
        border: 1px solid rgba(255, 255, 255, .72);
        color: #173b58;
    }
    .missing-evidence h5 {
        color: #173b58;
    }
    [data-testid="stAlert"] p {
        color: #102a43 !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #176b87 0%, #35a99b 100%);
        border: 1px solid rgba(16, 42, 67, .18);
        border-radius: 12px;
        color: #ffffff !important;
        font-size: .98rem;
        min-height: 2.8rem;
        font-weight: 700;
        transition: transform .2s ease, box-shadow .2s ease, background .2s ease;
    }
    .stButton > button p,
    .stButton > button div {
        color: #ffffff !important;
        font-weight: 800 !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #125873 0%, #278f84 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(16, 42, 67, .16);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #102a43 0%, #176b87 100%);
        border-color: #102a43;
    }
    .stButton > button:active {
        transform: translateY(0);
    }
    [data-testid="stMetric"], .score-card, .evidence-card {
        animation: soft-enter .6s cubic-bezier(.22, 1, .36, 1) both;
        will-change: transform, opacity;
    }
    @keyframes ambient-shift {
        0% { background-position: 0% 0%; }
        50% { background-position: 55% 35%; }
        100% { background-position: 100% 70%; }
    }
    @keyframes float-orb {
        0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
        50% { transform: translate3d(-10px, 12px, 0) scale(1.06); }
    }
    @keyframes score-grow {
        from { transform: scaleX(0); opacity: .3; }
        to { transform: scaleX(1); opacity: 1; }
    }
    @keyframes soft-enter {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }
    footer, [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
        display: none !important;
    }
    [data-testid="stExpander"] details,
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] div[role="group"] {
        background: rgba(16, 42, 67, .12) !important;
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
      <h1>Interview Evaluator</h1>
      <p>Turn a question and candidate response into clear, evidence-backed feedback in seconds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
if model_configured:
    st.caption(
        "Semantic model grading is configured "
        f"({os.getenv('NVIDIA_MODEL', os.getenv('OPENAI_MODEL', 'meta/llama-3.3-70b-instruct'))})."
    )
else:
    st.info(
        "Semantic model grading is not configured. Add NVIDIA_API_KEY to "
        "Streamlit Secrets; local meaning-aware grading remains available."
    )

with st.sidebar:
    st.markdown("## Interview Evaluator")
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

st.markdown('<div class="workflow-heading">Evaluate a candidate answer</div>', unsafe_allow_html=True)
st.markdown('<p class="hint">Start with a question. The reference answer can be generated automatically and edited before scoring.</p>', unsafe_allow_html=True)
if "evaluation_form_id" not in st.session_state:
    st.session_state.evaluation_form_id = 0
if "generated_reference" not in st.session_state:
    st.session_state.generated_reference = ""
form_id = st.session_state.evaluation_form_id
question = st.text_area(
    "Interview question",
    placeholder="e.g. What is dependency injection?",
    height=90,
    key=f"question_input_{form_id}",
)
answer = st.text_area(
    "Candidate answer",
    height=145,
    placeholder="Paste or type the candidate's response here...",
    key=f"candidate_answer_input_{form_id}",
)
generate_clicked = st.button("Generate reference answer", use_container_width=True)
if generate_clicked:
    try:
        generated = generate_reference_answer(question)
    except (RuntimeError, TypeError, ValueError) as error:
        st.error(str(error))
    else:
        st.session_state.generated_reference = str(generated["answer"])
        st.session_state[f"reference_input_{form_id}"] = str(generated["answer"])
        if generated.get("warning"):
            st.warning(
                f"{generated['warning']} Using the local reference generator instead."
            )
        st.success(f"Reference generated using {generated['source']}.")
        st.rerun()

reference_key = f"reference_input_{form_id}"
if reference_key not in st.session_state:
    st.session_state[reference_key] = st.session_state.generated_reference
reference = st.text_area(
    "Reference answer / expected concepts (generated automatically)",
    height=145,
    help="Generated from the question. You may edit it before scoring.",
    key=reference_key,
)
should_evaluate = st.button("Evaluate answer", type="primary", use_container_width=True)
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
        result = evaluate_answer_with_model(question, answer, reference)
    except (RuntimeError, TypeError, ValueError) as error:
        st.error(str(error))
    else:
        if result.get("evaluation_warning"):
            st.info(
                f"{result['evaluation_warning']} The completed local result remains valid."
            )
        st.markdown("### Evaluation snapshot")
        st.caption(f"Evaluation method: {result['evaluation_source']}")
        summary_title, summary_text = _performance_summary(result["percentage"])
        st.info(f"**{summary_title}** — {summary_text}")
        summary = st.columns(2)
        summary[0].metric("Overall score", f"{result['total_score']}/{result['maximum_score']}")
        summary[1].metric("Percentage", f"{result['percentage']}%")
        st.markdown('<h4 class="criterion-heading">Criterion performance</h4>', unsafe_allow_html=True)
        criterion_descriptions = {
            "correctness": "Technical accuracy and alignment with the expected concepts.",
            "relevance": "How directly the response addresses the question.",
            "completeness": "Coverage of the important ideas and reasoning.",
            "clarity": "Structure, concision, and ease of understanding.",
        }
        criteria = list(result["scores"].items())
        for row_start in range(0, len(criteria), 2):
            criterion_row = st.columns(2)
            for column, (criterion, score) in zip(criterion_row, criteria[row_start:row_start + 2]):
                column.markdown(
                    f'<div class="score-card">'
                    f'<div class="score-name">{criterion}</div>'
                    f'<div class="score-value">{score}<span> / 10</span></div>'
                    f'<div class="score-description">{criterion_descriptions.get(criterion, "Evaluation criterion score.")}</div>'
                    f'<div class="score-bar"><div class="score-fill" style="width: {score * 10}%"></div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                column.caption(f"Why this score: {result['score_reasons'][criterion]}")
        st.markdown('<h3 class="chart-heading">Score breakdown</h3>', unsafe_allow_html=True)
        chart_data = [
            {"Criterion": criterion.title(), "Score": score}
            for criterion, score in result["scores"].items()
        ]
        score_chart = (
            alt.Chart(alt.Data(values=chart_data))
            .mark_bar(
                cornerRadiusTopRight=8,
                cornerRadiusBottomRight=8,
                color="#176b87",
            )
            .encode(
                y=alt.Y(
                    "Criterion:N",
                    sort=None,
                    title=None,
                    axis=alt.Axis(
                        labelColor="#102a43",
                        labelFontSize=13,
                        labelFontWeight="bold",
                        labelPadding=10,
                        ticks=False,
                        domain=False,
                    ),
                ),
                x=alt.X(
                    "Score:Q",
                    title="Score out of 10",
                    scale=alt.Scale(domain=[0, 10]),
                    axis=alt.Axis(
                        labelColor="#486581",
                        titleColor="#102a43",
                        gridColor="#c9d6e3",
                        tickColor="#9fb3c8",
                        titlePadding=12,
                    ),
                ),
                tooltip=[
                    alt.Tooltip("Criterion:N", title="Criterion"),
                    alt.Tooltip("Score:Q", title="Score", format=".0f"),
                ],
            )
            .properties(height=220, background="transparent")
            .configure_view(stroke=None)
            .configure_axis(gridOpacity=.65)
        )
        st.altair_chart(score_chart, use_container_width=True)
        st.markdown("### Actionable feedback")
        for item in result["feedback"]:
            st.markdown(f"- {item}")
        with st.expander("Evidence and missing concepts"):
            st.caption("Open this panel for complete evidence. Hover over any concept statement for its interpretation.")
            evidence_left, evidence_right = st.columns(2)
            matched_items = "".join(
                f'<div class="concept-item"><p>{html.escape(concept)}</p>'
                f'<div class="concept-detail">Supported: this idea is reflected in the candidate answer. '
                f'Hovering reveals the evidence interpretation.</div></div>'
                for concept in result["matched_concepts"]
            ) or "<p>No supported concept statements found.</p>"
            missing_items = "".join(
                f'<div class="concept-item"><p>{html.escape(concept)}</p>'
                f'<div class="concept-detail">Not supported: this expected idea was not sufficiently '
                f'found in the candidate answer.</div></div>'
                for concept in result["missing_concepts"]
            ) or "<p>No missing concept statements.</p>"
            evidence_left.markdown(
                f'<div class="evidence-card matched-evidence">'
                f'<h5>Matched concept statements</h5>{matched_items}</div>',
                unsafe_allow_html=True,
            )
            evidence_right.markdown(
                f'<div class="evidence-card missing-evidence">'
                f'<h5>Missing concept statements</h5>{missing_items}</div>',
                unsafe_allow_html=True,
            )
        st.divider()
        st.button(
            "Start a new evaluation",
            use_container_width=True,
            on_click=_reset_evaluation,
        )
