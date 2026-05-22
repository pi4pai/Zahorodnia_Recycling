import os
import tempfile
import base64

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

from predict import predict_image
from database import save_prediction, get_all_predictions
from research import (
    get_class_distribution,
    get_average_confidence_by_class,
    get_analysis_count_by_date,
)


st.set_page_config(
    page_title="EcoTextile Studio",
    page_icon="◇",
    layout="wide",
    initial_sidebar_state="expanded",
)


GREEN = "#62FF7A"
PINK = "#FF7AB6"
YELLOW = "#FFD84D"
TEXT = "#F7F8F5"
MUTED = "#A7B0B8"


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    color: #F7F8F5;
    background:
        radial-gradient(circle at 10% 8%, rgba(98,255,122,.13), transparent 24%),
        radial-gradient(circle at 88% 0%, rgba(255,122,182,.12), transparent 24%),
        linear-gradient(135deg, #070A0E 0%, #090E13 50%, #07090C 100%);
}

header[data-testid="stHeader"] {
    background: transparent;
}

a[href^="#"] {
    display: none !important;
}

.block-container {
    max-width: 1480px;
    padding: 1.2rem 1.8rem 2.4rem 1.8rem;
}

/* sidebar */
section[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 50% 0%, rgba(255,122,182,.10), transparent 26%),
        linear-gradient(180deg, rgba(17,22,30,.98), rgba(8,11,15,.98));
    border-right: 1px solid rgba(255,255,255,.10);
}

section[data-testid="stSidebar"] * {
    color: #F7F8F5;
}

.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 8px 0 28px 0;
}

.logo-symbol {
    width: 44px;
    height: 44px;
    border-radius: 15px;
    display: grid;
    place-items: center;
    border: 1px solid rgba(98,255,122,.40);
    color: #62FF7A;
    font-size: 26px;
    background: rgba(98,255,122,.08);
}

.logo-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px;
    line-height: 1.05;
    font-weight: 700;
    letter-spacing: -0.04em;
}

.logo-title span {
    color: #62FF7A;
}

.stRadio label {
    min-height: 46px;
    padding: 10px 12px;
    border-radius: 15px;
    border: 1px solid transparent;
}

.stRadio label:hover {
    background: rgba(98,255,122,.08);
    border-color: rgba(98,255,122,.24);
}

/* native bordered containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid rgba(255,255,255,.12) !important;
    background:
        linear-gradient(135deg, rgba(20,27,36,.88), rgba(8,11,15,.92)) !important;
    box-shadow: 0 18px 50px rgba(0,0,0,.22);
    border-radius: 26px !important;
}

[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 1.25rem 1.35rem;
}

/* typography */
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(44px, 5vw, 76px);
    line-height: .94;
    letter-spacing: -0.07em;
    font-weight: 700;
    margin: 0;
}

.green {
    color: #62FF7A;
}

.pink {
    background: linear-gradient(135deg, #FF7AB6, #FFC2D8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.welcome {
    color: #62FF7A;
    font-weight: 900;
    letter-spacing: .42em;
    font-size: 13px;
    margin-bottom: 18px;
}

.hero-text {
    color: #D4D9D7;
    line-height: 1.6;
    font-size: 16px;
    margin-top: 16px;
}

.stat-number {
    color: #62FF7A;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 52px;
    font-weight: 700;
    letter-spacing: -0.05em;
    line-height: 1;
}

.stat-text {
    color: #CED5D1;
    margin-top: 14px;
    line-height: 1.35;
    font-size: 14px;
}

.step-row {
    display: flex;
    align-items: center;
    gap: 13px;
    margin-bottom: 12px;
}

.step {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    font-weight: 900;
    font-family: 'Space Grotesk', sans-serif;
    border: 1px solid #62FF7A;
    color: #62FF7A;
    background: rgba(98,255,122,.08);
    flex-shrink: 0;
}

.step.pink-step {
    color: #FF7AB6;
    border-color: #FF7AB6;
    background: rgba(255,122,182,.08);
}

.block-title {
    font-size: 23px;
    font-weight: 850;
    letter-spacing: -0.045em;
}

.muted {
    color: #A7B0B8;
    line-height: 1.58;
    font-size: 14px;
}

.format {
    color: #62FF7A;
    font-weight: 800;
}

.upload-zone {
    width: 100%;
    height: 330px !important;
    min-height: 330px !important;
    max-height: 330px !important;

    border-radius: 20px;
    border: 1px dashed rgba(98,255,122,.55);

    background:
        radial-gradient(circle at 50% 50%, rgba(98,255,122,.10), transparent 42%),
        rgba(255,255,255,.025);

    display: flex;
    align-items: center;
    justify-content: center;

    text-align: center;
    padding: 22px;
    overflow: hidden !important;
    box-sizing: border-box;
    margin-top: 14px;
    margin-bottom: 14px;
}

.upload-symbol {
    color: #62FF7A;
    font-size: 46px;
    line-height: 1;
}

.upload-main {
    margin-top: 12px;
    font-weight: 800;
}

.upload-small {
    margin-top: 8px;
    color: #A7B0B8;
    font-size: 13px;
}

.result-box {
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,.10);
    background:
        radial-gradient(circle at 10% 10%, rgba(98,255,122,.12), transparent 30%),
        radial-gradient(circle at 90% 70%, rgba(255,122,182,.14), transparent 32%),
        rgba(255,255,255,.035);
    padding: 18px;
    
}

.pred-title {
    color: #A7B0B8;
    font-size: 13px;
    margin-bottom: 8px;
}

.pred-class {
    color: #FF7AB6;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 42px;
    font-weight: 700;
    letter-spacing: -0.05em;
    line-height: 1;
}

.pred-desc {
    color: #A7B0B8;
    font-size: 13px;
    margin-top: 10px;
}

.prob-row {
    display: grid;
    grid-template-columns: 74px 1fr 58px;
    align-items: center;
    gap: 12px;
    color: #DCE2DE;
    font-size: 13px;
    margin-top: 12px;
    
}

.track {
    height: 8px;
    border-radius: 999px;
    background: rgba(255,255,255,.09);
    overflow: hidden;
}

.fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #62FF7A, #FF7AB6);
}

.rec-card {
    border-radius: 18px;

    height: 320px;
    min-height: 320px;
    max-height: 320px;

    padding: 18px;

    border: 1px solid rgba(255,255,255,.09);
    background: rgba(255,255,255,.035);

    display: flex;
    flex-direction: column;
    margin-bottom: 14px;
}
.rec-desc {
    color: #C8D0CC;
    font-size: 12.5px;
    line-height: 1.58;

    flex-grow: 1;
}

.rec-card.green-border {
    border-color: rgba(98,255,122,.30);
    background: linear-gradient(135deg, rgba(98,255,122,.10), rgba(255,255,255,.025));
}

.rec-card.yellow-border {
    border-color: rgba(255,216,77,.32);
    background: linear-gradient(135deg, rgba(255,216,77,.10), rgba(255,255,255,.025));
}

.rec-card.pink-border {
    border-color: rgba(255,122,182,.30);
    background: linear-gradient(135deg, rgba(255,122,182,.10), rgba(255,255,255,.025));
}

.rec-icon {
    width: 38px;
    height: 38px;
    border-radius: 14px;
    display: grid;
    place-items: center;
    margin-bottom: 12px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    border: 1px solid rgba(255,255,255,.18);
}

.rec-title {
    font-size: 17px;
    font-weight: 850;
    line-height: 1.18;
    margin-bottom: 10px;
}

.rec-badge {
    width: fit-content;
    padding: 5px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 800;
    margin-bottom: 12px;
    color: #071007;
}

.badge-green {
    background: #62FF7A;
}

.badge-yellow {
    background: #FFD84D;
}

.badge-pink {
    background: #FF7AB6;
}

.rec-desc {
    color: #C8D0CC;
    font-size: 12.5px;
    line-height: 1.58;
}

.footer-note {
    border-radius: 18px;
    padding: 16px 22px;
    color: #A7B0B8;
    font-size: 13px;
    border: 1px solid rgba(98,255,122,.17);
    background: rgba(255,255,255,.028);
}

.footer-note b {
    color: #62FF7A;
}

.page-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(42px, 5vw, 68px);
    line-height: .95;
    letter-spacing: -0.065em;
    font-weight: 700;
    margin-bottom: 22px;
}
.empty-state {
    height: 100%;

    border-radius: 22px;

    display: flex;
    align-items: center;
    justify-content: center;

    padding: 30px;

    color: #A7B0B8;
    text-align: center;

    border: 1px dashed rgba(255,255,255,.14);

    background: rgba(255,255,255,.03);
}

[data-testid="stFileUploader"] {
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,.08);
    background: rgba(255,255,255,.045);
    padding: 12px;
}

[data-testid="stFileUploader"] section {
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
}

.stButton > button {
    min-height: 50px;
    width: 100%;
    border: 0;
    border-radius: 14px;
    color: #17060E;
    font-weight: 900;
    background: linear-gradient(135deg, #FFC2D8, #FF7AB6);
    box-shadow: 0 16px 34px rgba(255,122,182,.20);
    margin-top: 14px;
}

.stButton > button:hover {
    color: #17060E;
    transform: translateY(-1px);
    box-shadow: 0 18px 42px rgba(255,122,182,.28);
}

img {
    border-radius: 18px;
}

[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,.08);
}

.fixed-preview {
    width: 100%;
    height: 330px !important;
    min-height: 330px !important;
    max-height: 330px !important;

    border-radius: 20px;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,.10);

    background:
        radial-gradient(circle at 50% 50%, rgba(98,255,122,.08), transparent 45%),
        rgba(255,255,255,.035);

    display: block;
    position: relative;
    box-sizing: border-box !important;
    flex-shrink: 0 !important;

    margin-top: 14px;
    margin-bottom: 14px;
}

.fixed-preview img {
    width: 100% !important;
    height: 330px !important;
    min-height: 330px !important;
    max-height: 330px !important;

    object-fit: cover !important;
    object-position: center center !important;

    display: block !important;
    border-radius: 0 !important;
}

/* One-piece result card */
.result-gradient-card {
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,.12);

    background:
        radial-gradient(circle at 16% 20%, rgba(98,255,122,.18), transparent 34%),
        radial-gradient(circle at 88% 70%, rgba(255,122,182,.22), transparent 36%),
        linear-gradient(135deg, rgba(16,22,30,.92), rgba(8,11,15,.96));

    padding: 24px;

    height: 330px !important;
    min-height: 330px !important;
    max-height: 330px !important;

    overflow: hidden !important;

    margin-top: 18px;
    box-sizing: border-box;
    margin-bottom: 14px;
}

.result-grid {
    display: grid;
    grid-template-columns: 220px 1fr;
    gap: 30px;
    align-items: center;
    height: 100%;
}

.css-donut {
    width: 200px;
    height: 200px;
    border-radius: 50%;
    background: conic-gradient(#FF7AB6 var(--p), rgba(255,255,255,.10) 0);
    display: grid;
    place-items: center;
    position: relative;
}

.css-donut::after {
    content: "";
    width: 128px;
    height: 128px;
    border-radius: 50%;
    background: #070A0E;
    position: absolute;
}

.donut-center {
    position: relative;
    z-index: 2;
    text-align: center;
}

.donut-value {
    color: #FF7AB6;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 36px;
    line-height: 1;
    font-weight: 700;
    letter-spacing: -0.05em;
}

.donut-label {
    margin-top: 8px;
    color: #F7F8F5;
    font-size: 13px;
}

.result-info {
    border-left: 1px solid rgba(255,255,255,.12);
    padding-left: 34px;
}

@media (max-width: 900px) {
    .result-grid {
        grid-template-columns: 1fr;
    }

    .result-info {
        border-left: 0;
        padding-left: 0;
    }

    .fixed-preview,
    .upload-zone {
        height: 330px !important;
        min-height: 330px !important;
        max-height: 330px !important;
    }
}


/* stable spacing inside main panels */
[data-testid="stFileUploader"] {
    margin-bottom: 0 !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    overflow: visible !important;
}

.rec-card {
    box-sizing: border-box;
}
/* history cards */
[data-testid="stImage"] img {
    height: 220px !important;
    object-fit: cover !important;
}
details {
    border-radius: 16px !important;
    overflow: hidden;
}

summary {
    font-weight: 700 !important;
}
/* Recycling guide cards */
[data-testid="column"] [data-testid="stVerticalBlockBorderWrapper"] {
    height: 100%;
}

</style>
""",
    unsafe_allow_html=True,
)



def uploaded_image_preview(uploaded_file):
    data = uploaded_file.getvalue()
    ext = os.path.splitext(uploaded_file.name)[1].lower().replace(".", "")

    mime_map = {
        "jpg": "jpeg",
        "jpeg": "jpeg",
        "png": "png",
        "webp": "webp",
        "bmp": "bmp",
        "tif": "tiff",
        "tiff": "tiff",
    }
    mime = mime_map.get(ext, "jpeg")
    encoded = base64.b64encode(data).decode("utf-8")

    html = f'<div class="fixed-preview"><img src="data:image/{mime};base64,{encoded}" alt="Uploaded fabric image"></div>'
    st.markdown(html, unsafe_allow_html=True)


def result_gradient_html(confidence, predicted_class, probs):
    p = max(0, min(float(confidence), 100)) * 3.6
    return f"""
<div class="result-gradient-card">
    <div class="result-grid">
        <div class="css-donut" style="--p:{p}deg;">
            <div class="donut-center">
                <div class="donut-value">{confidence}%</div>
                <div class="donut-label">confidence</div>
            </div>
        </div>
        <div class="result-info">
            <div class="pred-title">Predicted class</div>
            <div class="pred-class">{predicted_class}%</div>
            <div class="pred-desc">Synthetic fiber content range</div>
            <div style="margin-top:22px;">
                {probability_html(probs)}
            </div>
        </div>
    </div>
</div>
"""


def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def clean_class(value):
    return str(value).replace("%", "").replace("–", "-").replace("—", "-").strip()


def get_prob(probs, class_name):
    if not probs:
        return 0

    class_name = clean_class(class_name)

    for key, value in probs.items():
        if clean_class(key) == class_name:
            try:
                return round(float(value), 2)
            except Exception:
                return 0

    return 0


def method_by_class(predicted_class):
    predicted_class = clean_class(predicted_class)

    if predicted_class == "30-50":
        return "Mechanical Recycling"
    if predicted_class == "50-70":
        return "Hybrid Process"
    if predicted_class == "70-100":
        return "Chemical Recycling"

    return "Not selected"


def recommendations_by_class(predicted_class):
    predicted_class = clean_class(predicted_class)

    if predicted_class == "30-50":
        return [
            ("↻", "Mechanical Recycling", "Recommended", "green", "Sorting, shredding and reuse of fibers are suitable for materials with moderate synthetic content."),
            ("△", "Hybrid Process", "Suitable", "yellow", "A combined approach can be used when the material contains both natural and synthetic fibers."),
            ("◌", "Chemical Recycling", "Possible", "pink", "Can be applied when deeper processing of synthetic components is required."),
        ]

    if predicted_class == "50-70":
        return [
            ("△", "Hybrid Process", "Recommended", "green", "Recommended for blended fabrics where one recycling method is not enough."),
            ("◌", "Chemical Recycling", "Suitable", "pink", "Useful for processing the synthetic part of the textile and recovering polymer-based raw materials."),
            ("↻", "Mechanical Recycling", "Limited", "yellow", "May be used after sorting, but recovered fiber quality can be lower."),
        ]

    if predicted_class == "70-100":
        return [
            ("◌", "Chemical Recycling", "Recommended", "green", "Best option for fabrics with high synthetic content because polymer fibers need deeper recovery."),
            ("△", "Thermal Recovery", "Controlled use", "yellow", "Can be considered only when material recovery is not possible and industrial control is available."),
            ("↻", "Mechanical Recycling", "Low priority", "pink", "Possible for downcycling, but high synthetic content can reduce recovered fiber quality."),
        ]

    return [
        ("↻", "Mechanical Recycling", "Waiting", "green", "Upload and analyze an image to receive a recycling recommendation."),
        ("△", "Hybrid Process", "Waiting", "yellow", "The system will choose the most relevant option after classification."),
        ("◌", "Chemical Recycling", "Waiting", "pink", "Recommendations depend on the predicted synthetic fiber range."),
    ]


def probability_html(probs):
    rows = ""

    for label in ["30-50", "50-70", "70-100"]:
        value = get_prob(probs, label)
        width = max(0, min(float(value), 100))

        rows += (
            f'<div class="prob-row">'
            f'<div>{label}%</div>'
            f'<div class="track"><div class="fill" style="width:{width}%;"></div></div>'
            f'<div>{value}%</div>'
            f'</div>'
        )

    return rows

def confidence_donut(confidence):
    fig = go.Figure(
        data=[
            go.Pie(
                values=[confidence, max(0, 100 - confidence)],
                hole=0.72,
                marker=dict(colors=[PINK, "rgba(255,255,255,0.10)"]),
                textinfo="none",
                sort=False,
                direction="clockwise",
            )
        ]
    )

    fig.update_layout(
        height=210,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        annotations=[
            dict(
                text=f"{confidence}%",
                x=0.5,
                y=0.53,
                font=dict(size=30, color=PINK, family="Space Grotesk"),
                showarrow=False,
            ),
            dict(
                text="confidence",
                x=0.5,
                y=0.37,
                font=dict(size=10, color="#DCE2DF", family="Inter"),
                showarrow=False,
            ),
        ],
    )
    return fig


def style_plot(fig, title=None, height=210):
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=TEXT,
        title_font_color=TEXT,
        margin=dict(l=8, r=8, t=32, b=8),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
    )
    return fig


def sidebar():
    st.sidebar.markdown(
        """
<div class="sidebar-logo">
    <div class="logo-symbol">◇</div>
    <div class="logo-title">EcoTextile<br><span>Studio</span></div>
</div>
""",
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "History", "Analytics", "Articles", "Recycling Guide"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown(
        """
<div class="sidebar-card">
    <div class="sidebar-card-title">Deep Learning<br>System</div>
    <div class="sidebar-card-text">
        AI-powered textile analysis for smarter recycling decisions.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    return page


def dashboard_page():
    saved = len(get_all_predictions())

    with st.container(border=True):
        hero_left, hero_right = st.columns([1.45, 0.55], gap="large")

        with hero_left:
            st.markdown(
                """
<div class="welcome">WELCOME BACK</div>
<div class="hero-title">Make better <span class="green">recycling</span> decisions <span class="pink">with AI</span></div>
<div class="hero-text">
    Upload a fabric microimage, estimate the synthetic fiber range and receive a recycling recommendation in one clean workspace.
</div>
""",
                unsafe_allow_html=True,
            )

        with hero_right:
            st.markdown(
                f"""
<div class="stat-card">
    <div class="stat-number">{saved}</div>
    <div class="stat-text">saved analyses in the<br>local history database</div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.write("")

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        with st.container(border=True):
            st.markdown(
                """
<div class="step-row">
    <div class="step">1</div>
    <div class="block-title">Upload fabric image</div>
</div>
<div class="muted">
    Use one microscope image of textile material.<br>
    Supported formats: <span class="format">JPG, PNG, WEBP, BMP, TIFF.</span>
</div>
""",
                unsafe_allow_html=True,
            )

            uploaded_file = st.file_uploader(
                "Upload fabric image",
                type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"],
                label_visibility="collapsed",
            )

            if uploaded_file:
                uploaded_image_preview(uploaded_file)
            else:
                st.markdown(
                    """
<div class="upload-zone">
    <div>
        <div class="upload-symbol">☁</div>
        <div class="upload-main">Drag and drop image here</div>
        <div class="upload-small">or click to browse</div>
    </div>
</div>
""",
                    unsafe_allow_html=True,
                )

        with st.container(border=True):
            result_for_rec = st.session_state.get("last_result")
            predicted_class_for_rec = result_for_rec.get("predicted_class") if result_for_rec else None

            st.markdown(
                """
<div class="step-row">
    <div class="step">3</div>
    <div>
        <div class="block-title">Recycling recommendations</div>
        <div class="muted">Based on the predicted class</div>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )

            r1, r2, r3 = st.columns(3, gap="medium")
            rec_columns = [r1, r2, r3]

            for col, (icon, title, badge, color, desc) in zip(rec_columns, recommendations_by_class(predicted_class_for_rec)):
                border_class = f"{color}-border"
                badge_class = f"badge-{color}"

                with col:
                    st.markdown(
                        f"""
<div class="rec-card {border_class}">
    <div class="rec-icon">{icon}</div>
    <div class="rec-title">{title}</div>
    <div class="rec-badge {badge_class}">{badge}</div>
    <div class="rec-desc">{desc}</div>
</div>
""",
                        unsafe_allow_html=True,
                    )

    with right:
        with st.container(border=True):
            st.markdown(
                """
<div class="step-row">
    <div class="step pink-step">2</div>
    <div class="block-title">Analysis result</div>
</div>
<div class="muted">
    The result appears here after classification.<br>
    There is only one analysis panel.
</div>
""",
                unsafe_allow_html=True,
            )

            clicked = st.button("Analyze fabric")

            if clicked:
                if uploaded_file is None:
                    st.warning("Please upload an image first.")
                else:
                    image_path = save_uploaded_file(uploaded_file)
                    result = predict_image(image_path)

                    if not result.get("recycling_method"):
                        result["recycling_method"] = method_by_class(result.get("predicted_class", ""))

                    if not result.get("recycling_description"):
                        result["recycling_description"] = "The recommendation is based on the predicted synthetic fiber content range."

                    save_prediction(result)
                    st.session_state["last_result"] = result
                    st.rerun()

            result = st.session_state.get("last_result")

            if result:
                confidence = round(float(result.get("confidence", 0)), 2)
                predicted_class = clean_class(result.get("predicted_class", "—"))
                probs = result.get("probabilities", {})

                st.markdown(
                    result_gradient_html(confidence, predicted_class, probs),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
<div class="result-gradient-card">
    <div class="empty-state">
        Run analysis to see confidence score, predicted class and probabilities.
    </div>
</div>
""",
                    unsafe_allow_html=True,
                )
                
    with right:
     with st.container(border=True):
        st.markdown(
            """
<div class="block-title">Analytics overview</div>
<div class="muted">Key insights from your data</div>
""",
            unsafe_allow_html=True,
        )

        class_dist = get_class_distribution()
        avg_conf = get_average_confidence_by_class()
        by_date = get_analysis_count_by_date()

        top1, top2 = st.columns(2, gap="large")

        with top1:
            if class_dist:
                fig = px.pie(
                    values=list(class_dist.values()),
                    names=list(class_dist.keys()),
                    hole=0.62,
                    color_discrete_sequence=[GREEN, PINK, YELLOW],
                )

                fig.update_layout(
                    height=170,
                    margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color=TEXT,
                    showlegend=False,
                )

                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with top2:
            if avg_conf:
                fig = px.bar(
                    x=list(avg_conf.keys()),
                    y=list(avg_conf.values()),
                    color=list(avg_conf.keys()),
                    color_discrete_sequence=[GREEN, PINK, YELLOW],
                )

                fig.update_layout(
                    height=170,
                    margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color=TEXT,
                    showlegend=False,
                    yaxis=dict(range=[0, 100], gridcolor="rgba(255,255,255,0.08)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                )

                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

        if by_date:
            fig = px.line(
                x=[str(k) for k in by_date.keys()],
                y=list(by_date.values()),
                markers=True,
                labels={"x": "Date", "y": "Analyses"},
            )

            fig.update_traces(line_color=GREEN, marker_color=PINK, line_width=3)

            fig.update_layout(
                height=150,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color=TEXT,
                xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            )

            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        """
<div class="footer-note">
    <b>Every analysis helps build a cleaner future.</b>
    &nbsp; Thank you for contributing to sustainable textile practices.
</div>
""",
        unsafe_allow_html=True,
    )



def history_page():

    st.markdown(
        '<div class="page-title">Analysis <span class="green">history</span></div>',
        unsafe_allow_html=True,
    )

    rows = get_all_predictions()

    if not rows:
        st.markdown(
            '<div class="empty-state">No saved analyses yet.</div>',
            unsafe_allow_html=True,
        )
        return

    def get_row_date(row):
        try:
            return datetime.strptime(str(row[1]), "%Y-%m-%d %H:%M:%S")
        except Exception:
            return datetime.min

    rows = sorted(rows, key=get_row_date, reverse=True)

    cols = st.columns(3, gap="large")

    for index, row in enumerate(rows):

        try:
            _, date, image_path, predicted_class, confidence, *_rest, recycling_method = row
        except Exception:
            continue

        col = cols[index % 3]

        with col:

            with st.container(border=True):

                if image_path and os.path.exists(image_path):

                    st.image(
                        image_path,
                        use_container_width=True,
                    )

                st.markdown(
                    f"""
<div style="
margin-top:14px;
display:flex;
justify-content:space-between;
align-items:center;
gap:10px;
">

<div style="
background:rgba(98,255,122,.12);
color:#62FF7A;
padding:6px 12px;
border-radius:999px;
font-size:12px;
font-weight:800;
">
{clean_class(predicted_class)}%
</div>

<div style="
background:rgba(255,122,182,.12);
color:#FF7AB6;
padding:6px 12px;
border-radius:999px;
font-size:12px;
font-weight:800;
">
{round(float(confidence), 2)}%
</div>

</div>
""",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
<div style="
font-size:22px;
font-weight:800;
margin-top:18px;
line-height:1.1;
">
{method_by_class(predicted_class)}
</div>
""",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
<div style="
color:#A7B0B8;
font-size:13px;
margin-top:10px;
">
Analysis date: {date}
</div>
""",
                    unsafe_allow_html=True,
                )

def analytics_page():
    st.markdown('<div class="page-title">Analytics</div>', unsafe_allow_html=True)

    class_dist = get_class_distribution()
    avg_conf = get_average_confidence_by_class()
    by_date = get_analysis_count_by_date()

    c1, c2 = st.columns(2, gap="large")

    with c1:
        with st.container(border=True):
            st.markdown('<div class="block-title">Class distribution</div>', unsafe_allow_html=True)

            if class_dist:
                fig = px.bar(
                    x=list(class_dist.keys()),
                    y=list(class_dist.values()),
                    color_discrete_sequence=[GREEN],
                    labels={"x": "Class", "y": "Count"},
                )
                fig = style_plot(fig, "Class distribution", height=330)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.markdown('<div class="empty-state">No data available.</div>', unsafe_allow_html=True)

    with c2:
        with st.container(border=True):
            st.markdown('<div class="block-title">Average confidence</div>', unsafe_allow_html=True)

            if avg_conf:
                fig = px.bar(
                    x=list(avg_conf.keys()),
                    y=list(avg_conf.values()),
                    color_discrete_sequence=[PINK],
                    labels={"x": "Class", "y": "Confidence"},
                )
                fig = style_plot(fig, "Average confidence", height=330)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.markdown('<div class="empty-state">No data available.</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="block-title">Analyses by date</div>', unsafe_allow_html=True)

        if by_date:
            fig = px.line(
                x=[str(k) for k in by_date.keys()],
                y=list(by_date.values()),
                markers=True,
                labels={"x": "Date", "y": "Count"},
            )
            fig.update_traces(line_color=GREEN, marker_color=PINK, line_width=3)
            fig = style_plot(fig, "Analyses by date", height=360)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown('<div class="empty-state">No data available.</div>', unsafe_allow_html=True)


def articles_page():
    st.markdown(
        '<div class="page-title">Articles</div>',
        unsafe_allow_html=True,
    )

    articles = [
        {
            "tag": "Technology",
            "title": "Mechanical vs chemical recycling",
            "preview": "Two major recycling approaches used in textile waste management.",
            "full": """
Mechanical recycling focuses on sorting, shredding and reusing textile fibers.
It is considered one of the most accessible and energy-efficient recycling methods
for fabrics containing natural fibers or lower synthetic content.

Chemical recycling is more advanced and is commonly used for textiles with high
synthetic fiber concentration. In this process, polymers are chemically broken down
and transformed into reusable raw materials that can later be used in new textile production.

Both methods are important for reducing textile waste and supporting circular fashion systems.
""",
        },

        {
            "tag": "Research",
            "title": "Why blended fabrics are difficult to recycle",
            "preview": "Mixed textiles require more complex recycling systems.",
            "full": """
Blended fabrics combine multiple fiber types such as cotton and polyester.
Although these materials improve durability and comfort, they also create
serious recycling challenges.

Different fibers react differently to heat, chemicals and mechanical processing.
Because of this, separating materials inside blended fabrics becomes difficult.

Modern AI systems and computer vision technologies can improve textile sorting
accuracy and help identify the most suitable recycling process automatically.
""",
        },

        {
            "tag": "AI system",
            "title": "Microimage classification with deep learning",
            "preview": "CNN models can analyze microscopic textile structures.",
            "full": """
Convolutional Neural Networks (CNNs) are widely used in computer vision tasks.
In textile analysis, CNN models can detect patterns, textures and fiber structures
inside microscope images.

The system learns visual differences between textile categories and predicts
synthetic fiber content ranges automatically.

Deep learning helps reduce manual sorting errors and can support automated
decision-making in sustainable textile recycling.
""",
        },

        {
            "tag": "Sustainability",
            "title": "AI for circular fashion",
            "preview": "Artificial intelligence can reduce textile waste.",
            "full": """
The fashion industry produces enormous amounts of textile waste every year.
Artificial intelligence technologies can improve sorting systems, classify materials
faster and increase recycling efficiency.

AI-powered fabric recognition systems help determine the composition of textiles
before recycling begins.

This supports circular economy principles where materials are reused instead of discarded.
""",
        },
    ]

    for article in articles:

        with st.container(border=True):

            st.markdown(
                f"""
<div style="
color:#FF7AB6;
font-size:12px;
font-weight:900;
letter-spacing:.18em;
text-transform:uppercase;
margin-bottom:10px;
">
{article['tag']}
</div>

<div style="
font-size:30px;
font-weight:800;
line-height:1;
margin-bottom:14px;
">
{article['title']}
</div>

<div style="
color:#A7B0B8;
font-size:15px;
line-height:1.7;
margin-bottom:10px;
">
{article['preview']}
</div>
""",
                unsafe_allow_html=True,
            )

            with st.expander("Read full article"):

                st.markdown(
                    f"""
<div style="
color:#D7DDD9;
font-size:15px;
line-height:1.9;
padding-top:10px;
">
{article['full']}
</div>
""",
                    unsafe_allow_html=True,
                )


def guide_page():
    st.markdown(
        '<div class="page-title">Recycling <span class="green">guide</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div style="
font-size:34px;
font-weight:850;
line-height:1.05;
margin-bottom:14px;
">
What do these percentages mean?
</div>

<div style="
color:#C8D0CC;
font-size:15px;
line-height:1.8;
max-width:980px;
margin-bottom:30px;
">
The percentage range shows how much synthetic fiber is estimated in the textile material.
For example, 30–50% means that the fabric contains a moderate amount of synthetic fibers,
while 70–100% means that the fabric is mostly synthetic. This value helps the system choose
the most suitable recycling method.
</div>
""",
        unsafe_allow_html=True,
    )

    guides = [
        {
            "range": "30–50%",
            "method": "Mechanical Recycling",
            "tag": "Moderate synthetic content",
            "color": "#62FF7A",
            "desc": """
This range means that the fabric contains a relatively balanced combination of natural and synthetic fibers.
Mechanical recycling is usually the first option because the material can often be sorted, shredded and reused
as secondary fiber without complex chemical treatment.
""",
            "steps": [
                "Sorting textile by material type and color.",
                "Cutting or shredding fabric into smaller fiber fragments.",
                "Using recovered fibers for insulation or secondary materials.",
            ],
            "best": "Best for fabrics where the structure can still be reused mechanically.",
            "warning": "Recovered fiber quality may be lower than the original material.",
        },

        {
            "range": "50–70%",
            "method": "Hybrid Recycling",
            "tag": "Mixed textile composition",
            "color": "#FFD84D",
            "desc": """
Synthetic fibers already dominate, but natural fibers may still be present.
A hybrid approach combines mechanical preparation with additional processing methods.
""",
            "steps": [
                "Pre-sorting textile waste.",
                "Mechanical fiber preparation.",
                "Additional chemical or thermal treatment.",
            ],
            "best": "Best for blended fabrics where one recycling method is not enough.",
            "warning": "Different fibers require different processing conditions.",
        },

        {
            "range": "70–100%",
            "method": "Chemical Recycling",
            "tag": "High synthetic content",
            "color": "#FF7AB6",
            "desc": """
This textile is mostly synthetic. Chemical recycling breaks polymer fibers into reusable raw materials.
""",
            "steps": [
                "Identifying polymer-based materials.",
                "Breaking down synthetic fibers chemically.",
                "Recovering reusable raw materials.",
            ],
            "best": "Best for highly synthetic textile materials.",
            "warning": "Requires industrial equipment and controlled conditions.",
        },
    ]

    cols = st.columns(3, gap="large")

    for col, item in zip(cols, guides):

        with col:
            with st.container(border=True):

                card_html = f"""
<div style="display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:18px;">

<div style="
background:{item['color']};
color:#071007;
padding:7px 13px;
border-radius:999px;
font-size:13px;
font-weight:900;
">
{item['range']}
</div>

<div style="
color:{item['color']};
font-size:12px;
font-weight:900;
text-transform:uppercase;
letter-spacing:.12em;
text-align:right;
">
{item['tag']}
</div>

</div>

<div style="
font-size:26px;
font-weight:850;
line-height:1.05;
margin-bottom:14px;
">
{item['method']}
</div>

<div style="
color:#C8D0CC;
font-size:14px;
line-height:1.75;
margin-bottom:18px;
">
{item['desc']}
</div>

<div style="
color:#62FF7A;
font-size:13px;
font-weight:900;
letter-spacing:.08em;
text-transform:uppercase;
margin-bottom:10px;
">
Main stages
</div>
"""

                st.markdown(card_html, unsafe_allow_html=True)

                for step in item["steps"]:
                    st.markdown(
                        f"""
<div style="
display:flex;
gap:10px;
align-items:flex-start;
margin-bottom:10px;
color:#DCE2DE;
font-size:13px;
line-height:1.55;
">
<span style="color:{item['color']};font-weight:900;">◇</span>
<span>{step}</span>
</div>
""",
                        unsafe_allow_html=True,
                    )

                bottom_html = f"""
<div style="
margin-top:18px;
padding:14px;
border-radius:16px;
background:rgba(98,255,122,.06);
border:1px solid rgba(98,255,122,.16);
color:#C8D0CC;
font-size:13px;
line-height:1.6;
">
<b style="color:{item['color']};">Best use:</b> {item['best']}
</div>

<div style="
margin-top:12px;
padding:14px;
border-radius:16px;
background:rgba(255,122,182,.06);
border:1px solid rgba(255,122,182,.16);
color:#C8D0CC;
font-size:13px;
line-height:1.6;
margin-bottom:20px;
">
<b style="color:#FF7AB6;">Note:</b> {item['warning']}
</div>
"""

                st.markdown(bottom_html, unsafe_allow_html=True)

page = sidebar()

if page == "Dashboard":
    dashboard_page()
elif page == "History":
    history_page()
elif page == "Analytics":
    analytics_page()
elif page == "Articles":
    articles_page()
elif page == "Recycling Guide":
    guide_page()
