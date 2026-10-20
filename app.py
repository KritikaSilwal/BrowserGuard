"""Three-page Streamlit interface for AI BrowserGuard."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.explain import (
    explain_with_shap_and_method,
    global_shap_importance,
    shap_waterfall_data,
)
from src.prediction import load_bundle, predict_batch, predict_url, predict_with_model
from src.display import stringify_value_column
from src.security_analysis import (
    compare_counterfactuals,
    parse_url_structure,
    structural_observations,
)
from src.reliability import threshold_metrics, unfamiliarity

MODEL_PATH = Path("models/url_model.joblib")

# Shared visual tokens keep the application consistent across pages and charts.
NAVY = "#16324F"
DEEP_NAVY = "#142B45"
BACKGROUND = "#F5F8FC"
WHITE = "#FFFFFF"
PALE_BLUE = "#E5F1FA"
TEAL = "#087F8C"
PINK = "#F4C7D9"
TEXT = "#263445"
MUTED = "#566579"
BORDER = "#D9E2EC"
SUCCESS = "#18794E"
WARNING = "#9A6700"
DANGER = "#B42318"

st.set_page_config(page_title="AI BrowserGuard", page_icon=":material/security:", layout="wide")
st.markdown(
    """
    <style>
    :root { --navy:#16324F; --deep-navy:#142B45; --bg:#F5F8FC; --white:#FFFFFF; --pale-blue:#E5F1FA; --teal:#087F8C; --pink:#F4C7D9; --text:#263445; --muted:#566579; --border:#D9E2EC; --danger:#B42318; }
    .stApp, [data-testid="stAppViewContainer"] > .main { background: var(--bg); color: var(--text); }
    [data-testid="stHeader"] { background: var(--bg); }
    [data-testid="stSidebar"] {
        background: var(--deep-navy);
        border-right: 1px solid #284763;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.4rem;
    }
    [data-testid="stSidebar"] .sidebar-brand {
        padding: 0.25rem 0.55rem 1.35rem;
        border-bottom: 1px solid rgba(199, 213, 229, 0.2);
        margin-bottom: 1.15rem;
    }
    [data-testid="stSidebar"] .sidebar-brand-title {
        color: #FFFFFF;
        font-size: 1.18rem;
        font-weight: 700;
        line-height: 1.25;
        letter-spacing: -0.02em;
    }
    [data-testid="stSidebar"] .sidebar-brand-icon {
        color: #FFFFFF;
        margin-right: 0.42rem;
    }
    [data-testid="stSidebar"] .sidebar-brand-subtitle {
        color: #C7D5E5;
        font-size: 0.84rem;
        line-height: 1.35;
        margin-top: 0.35rem;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] > label[data-testid="stWidgetLabel"] {
        color: #DCE7F3 !important;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin: 0 0 0.45rem 0.75rem;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] {
        align-items: center;
        border-radius: 10px;
        color: #F3F7FC !important;
        display: flex;
        min-height: 2.7rem;
        padding: 0.5rem 0.75rem;
        margin: 0.18rem 0;
        transition: background-color 160ms ease, color 160ms ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
        display: none;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] > div:last-child,
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] > div:last-child *,
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] > div:last-child p {
        color: #F3F7FC !important;
        font-size: 0.96rem;
        font-weight: 500;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:hover {
        background: #203D5D;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
        background: #087F8C;
        color: #FFFFFF !important;
        font-weight: 600;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) > div:last-child,
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) > div:last-child *,
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) > div:last-child p {
        color: #FFFFFF !important;
        font-weight: 600;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:focus-within {
        outline: 2px solid #FFFFFF;
        outline-offset: 2px;
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1440px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, h4, p, label, [data-testid="stMarkdownContainer"],
    [data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"] * {
        color: var(--text);
    }
    h1 { font-size: clamp(2rem, 4vw, 3.25rem); letter-spacing: -0.045em; }
    h2 { letter-spacing: -0.025em; }
    .hero { padding: 8px 0 22px; }
    .hero p, .muted, [data-testid="stCaptionContainer"] { color: var(--muted); }
    .hero-panel, .section-panel {
        background: var(--white);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 28px;
        box-shadow: 0 10px 26px rgba(20,43,69,.08);
    }
    .eyebrow { color: var(--teal); font-size: .78rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
    [data-testid="stMetric"] {
        background: var(--white);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 6px 18px rgba(20,43,69,.07);
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: var(--text);
        font-weight: 700;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--border);
        border-radius: 18px;
    }
    [data-testid="stTextInput"] input, [data-testid="stFileUploader"] section {
        background: var(--white);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 12px;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: var(--teal);
        box-shadow: 0 0 0 2px var(--pale-blue);
    }
    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span {
        color: var(--muted);
        opacity: 1;
    }
    [data-baseweb="select"] *,
    [data-baseweb="select"] input,
    [data-baseweb="popover"] *,
    [role="option"] {
        color: var(--text);
    }
    .stButton > button, [data-testid="stFormSubmitButton"] button,
    [data-testid="stDownloadButton"] button {
        background: var(--teal);
        color: #FFFFFF;
        border: 1px solid var(--teal);
        border-radius: 999px;
        font-weight: 700;
        transition: transform 160ms ease, background-color 160ms ease;
    }
    .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover,
    [data-testid="stDownloadButton"] button:hover {
        background: var(--navy);
        color: #FFFFFF;
        transform: translateY(-1px);
    }
    [data-testid="stAlert"] {
        background: var(--pale-blue);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 14px;
    }
    [data-testid="stAlert"] * { color: var(--text); }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 14px;
        overflow: hidden;
        background: var(--white);
    }
    [data-testid="stDataFrame"] * { color: var(--text); }
    button[role="tab"] { color: var(--text); }
    button[role="tab"][aria-selected="true"] {
        color: var(--teal);
        border-bottom-color: var(--teal);
    }
    @media (max-width: 720px) {
        [data-testid="stMainBlockContainer"] {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1.2rem;
        }
        .hero-panel, .section-panel { padding: 18px; }
        h1 { font-size: 2.2rem; }
    }
    hr { border-color: var(--border); opacity: 0.8; }
    </style>
    """,
    unsafe_allow_html=True,
)


def style_chart(figure: go.Figure, title: str | None = None) -> go.Figure:
    """Apply the shared warm theme without changing chart data."""
    if title:
        figure.update_layout(title=title)
    figure.update_layout(
        paper_bgcolor=WHITE,
        plot_bgcolor=BACKGROUND,
        font={"color": TEXT, "family": "Arial, sans-serif"},
        margin={"l": 68, "r": 28, "t": 64, "b": 64},
        legend={"bgcolor": WHITE, "font": {"color": TEXT}, "orientation": "h", "y": -0.18},
        colorway=[TEAL, NAVY, PINK, DANGER, DEEP_NAVY],
        autosize=True,
    )
    figure.update_xaxes(
        gridcolor=BORDER,
        zerolinecolor=NAVY,
        title_font={"color": "#000000"},
        tickfont={"color": "#000000"},
    )
    figure.update_yaxes(
        gridcolor=BORDER,
        zerolinecolor=NAVY,
        title_font={"color": "#000000"},
        tickfont={"color": "#000000"},
    )
    figure.update_layout(
        title_font={"color": "#000000"},
        annotationdefaults={"font": {"color": "#000000"}},
    )
    return figure


def init_state() -> None:
    st.session_state.setdefault("scan_history", [])
    st.session_state.setdefault("last_scan", None)
    st.session_state.setdefault("page", "Home")
    st.session_state.setdefault("navigate_to", None)


def request_navigation(page: str) -> None:
    """Queue navigation for the next script run before the page widget exists."""
    st.session_state["navigate_to"] = page


def metric_cards(items: list[tuple[str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items):
        column.metric(label, value)


def assessment_state(logistic_prediction: str, forest_prediction: str | None) -> tuple[str, str]:
    """Return a user-facing state from the saved model classifications."""
    if forest_prediction is not None and logistic_prediction != forest_prediction:
        return "Inconclusive", "The saved models disagree; review both outputs."
    if logistic_prediction == "Potentially phishing":
        if forest_prediction is None:
            return "Suspicious characteristics detected", "The available model identified phishing characteristics."
        return "High concern", "Both available models identify phishing characteristics."
    return "Low concern", "The available models did not identify a phishing classification."


def render_contribution_chart(frame: pd.DataFrame, title: str) -> None:
    if frame.empty:
        st.info("No contribution values are available for this direction.")
        return
    chart = px.bar(
        frame.sort_values("contribution"),
        x="contribution",
        y="feature",
        orientation="h",
        title=title,
        color="contribution",
        color_continuous_scale=[PALE_BLUE, WHITE, TEAL],
    )
    st.plotly_chart(style_chart(chart), width="stretch")


def render_findings(result: dict, explanation_frame: pd.DataFrame) -> None:
    """Show cautious evidence, separating observed structure from model signals."""
    st.subheader("Why was this URL flagged?")
    findings: list[dict[str, str]] = []
    features = result["raw_features"]
    if features["is_ipv4"]:
        findings.append({"source": "Observed structure", "finding": "IP-address hostname", "evidence": "The hostname is an IPv4 address.", "severity": "Review"})
    if features["has_punycode"]:
        findings.append({"source": "Observed structure", "finding": "Punycode hostname", "evidence": "The hostname includes an xn-- label.", "severity": "Review"})
    if features["has_at_symbol"]:
        findings.append({"source": "Observed structure", "finding": "At-symbol present", "evidence": "The URL contains @, which can obscure the destination in some URL formats.", "severity": "Review"})
    if features["has_percent_encoding"]:
        findings.append({"source": "Observed structure", "finding": "Encoded characters", "evidence": "The URL contains percent-encoded characters.", "severity": "Informational"})
    scheme_rows = explanation_frame[explanation_frame["feature"].isin(["http", "https"])]
    model_frame = explanation_frame[~explanation_frame["feature"].isin(["http", "https"])].copy()
    if not scheme_rows.empty:
        model_frame = pd.concat([
            model_frame,
            pd.DataFrame([{
                "feature": "scheme indicators (HTTP/HTTPS)",
                "contribution": float(scheme_rows["contribution"].sum()),
            }]),
        ], ignore_index=True)
    for _, row in model_frame[model_frame["contribution"] > 0].head(3).iterrows():
        findings.append({
            "source": "Logistic Regression signal",
            "finding": str(row["feature"]),
            "evidence": f"Centered phishing log-odds contribution: {row['contribution']:.4f}.",
            "severity": "Model signal",
        })
    if not findings:
        st.info("No implemented structural flags or positive displayed model signals were found.")
    else:
        st.dataframe(pd.DataFrame(findings), hide_index=True, width="stretch")
    st.caption(
        "Structural observations are evidence for review, not proof of phishing. "
        "Model signals are statistical and should be checked independently against the domain and context."
    )


def render_sidebar() -> str:
    requested_page = st.session_state.pop("navigate_to", None)
    if requested_page in {"Home", "URL scanner", "Analytics & reports"}:
        st.session_state["page"] = requested_page
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-title">
                    <span class="sidebar-brand-icon">◈</span>AI BrowserGuard
                </div>
                <div class="sidebar-brand-subtitle">
                    Explainable URL Threat Intelligence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return st.radio(
            "Navigate",
            ["Home", "URL scanner", "Analytics & reports"],
            format_func=lambda page: {
                "Home": "⌂  Home",
                "URL scanner": "⌕  URL Scanner",
                "Analytics & reports": "▤  Analytics & Reports",
            }[page],
            key="page",
            label_visibility="collapsed",
        )


def render_home(bundle: dict) -> None:
    hero_left, hero_right = st.columns([1.08, .92], vertical_alignment="center")
    with hero_left:
        st.markdown('<div class="eyebrow">AI-powered URL protection</div>', unsafe_allow_html=True)
        st.title("Browse smarter.\nDetect threats earlier.")
        st.markdown(
            "Analyze suspicious URLs with machine learning, understand the signals behind each prediction, and explore model reliability."
        )
        primary, secondary = st.columns(2)
        with primary:
            st.button(
                "Analyze a URL",
                type="primary",
                width="stretch",
                on_click=request_navigation,
                args=("URL scanner",),
            )
        with secondary:
            st.button(
                "Explore insights",
                width="stretch",
                on_click=request_navigation,
                args=("Analytics & reports",),
            )
    with hero_right:
        asset = Path("assets/browserguard_hero.svg")
        if asset.exists():
            st.image(str(asset), caption="Local, URL-only analysis", width="stretch")
        else:
            st.info("Local protection illustration unavailable.")
    history = st.session_state["scan_history"]
    phishing = sum(item["prediction"] == "Potentially phishing" for item in history)
    legitimate = len(history) - phishing
    metric_cards([
        ("URLs scanned this session", str(len(history))),
        ("Potential phishing detections", str(phishing)),
        ("Likely legitimate detections", str(legitimate)),
    ])
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Prediction distribution")
        if history:
            counts = pd.Series([item["prediction"] for item in history]).value_counts()
            distribution = px.pie(
                values=counts.values,
                names=counts.index,
                hole=0.55,
                title="URL Prediction Distribution",
                labels={"value": "Scans", "variable": "Prediction"},
                color_discrete_sequence=[DANGER, TEAL],
            )
            distribution.update_traces(
                textinfo="label+percent",
                hovertemplate="%{label}<br>Count: %{value}<br>Share: %{percent}<extra></extra>",
            )
            st.plotly_chart(style_chart(distribution), width="stretch")
        else:
            st.info("Run a scan to build session activity.")
    with right:
        st.subheader("Recent scan activity")
        if history:
            activity = pd.DataFrame(history[-8:][::-1]).copy()
            activity["URL"] = activity["URL"].map(
                lambda value: str(value) if len(str(value)) <= 56 else f"{str(value)[:53]}..."
            )
            st.dataframe(
                activity[["timestamp", "URL", "prediction"]],
                hide_index=True,
                width="stretch",
                height=292,
            )
            st.caption("Long URLs are shortened for display; the full URL remains in Scan history and exports.")
        else:
            st.info("No scans recorded.")
    st.subheader("Model performance summary")
    metrics = bundle.get("metrics", {})
    metric_cards([
        ("Accuracy", f"{metrics.get('accuracy', 0):.3f}"),
        ("Phishing precision", f"{metrics.get('precision', 0):.3f}"),
        ("Phishing recall", f"{metrics.get('recall', 0):.3f}"),
        ("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}"),
    ])
    with st.container(border=True):
        st.markdown("**Model status**")
        calibration = metrics.get("calibration")
        st.write(
            f"Active model: Logistic Regression · Evaluation: stratified row split · "
            f"Calibration: {'available' if calibration else 'not available'}"
        )
        st.caption("Predictions are URL-structure estimates. No submitted URL is opened or sent to an external service.")


def render_scanner(bundle: dict) -> None:
    st.markdown('<div class="eyebrow">Local threat analysis</div>', unsafe_allow_html=True)
    st.title("Analyze a suspicious URL")
    st.caption("Analyze URL structure locally. Submitted URLs are never opened, resolved, or sent to an external service.")
    with st.form("url-analysis"):
        raw_url = st.text_input("URL to analyze", placeholder="https://example.com/account")
        submitted = st.form_submit_button("Scan URL", type="primary", width="stretch")
    if submitted:
        try:
            result = predict_url(raw_url, bundle)
            st.session_state["last_scan"] = result
            st.session_state["scan_history"].append({
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "URL": result["url"], "prediction": result["classification"],
                "phishing_probability": result["phishing_probability"],
            })
        except (ValueError, TypeError) as exc:
            st.error(str(exc), title="Invalid URL")
    result = st.session_state.get("last_scan")
    if not result:
        st.info("Enter a URL above to begin a local risk analysis.")
        return
    forest_result = None
    if "random_forest" in bundle:
        forest_result = predict_with_model(result["url"], bundle["random_forest"], bundle)
    state, state_description = assessment_state(
        result["classification"],
        forest_result["classification"] if forest_result else None,
    )
    if state == "Inconclusive":
        st.warning(
            f"**{state}** — {state_description} No combined probability is reported.",
        )
        metric_cards([
            ("Assessment", state),
            ("Logistic Regression", result["classification"]),
            ("LR phishing probability", f"{result['phishing_probability']:.5%}"),
        ])
    else:
        metric_cards([
            ("Assessment", state),
            ("Classification", result["classification"]),
            ("Phishing probability", f"{result['phishing_probability']:.1%}"),
            ("Model agreement", "Both models agree" if forest_result else "Unavailable"),
        ])
    st.caption(state_description)
    st.subheader("Recommended actions")
    if state in {"High concern", "Inconclusive"}:
        st.markdown(
            "- Do not enter credentials or payment details on the destination.\n"
            "- Verify the domain through an independent, trusted channel.\n"
            "- Treat unexpected messages containing this URL as suspicious."
        )
    else:
        st.markdown(
            "- Continue to verify the domain and context before entering sensitive information.\n"
            "- Remember that URL-only analysis cannot establish that a site is safe."
        )
    early_explanations, explanation_method = explain_with_shap_and_method(
        bundle["model"], result["scaled_features"], bundle["feature_names"]
    )
    render_findings(result, pd.DataFrame(early_explanations))
    with st.expander("Prediction reliability and model consensus", expanded=True):
        reliability_reference = bundle.get("train_feature_reference")
        if reliability_reference:
            unfamiliar = unfamiliarity(
                result["scaled_features"],
                reliability_reference["mean"],
                reliability_reference["std"],
            )
            st.write(f"Input unfamiliarity: **{unfamiliar['level']}**")
            st.caption(
                f"{unfamiliar['unusual_feature_count']} standardized feature values exceed "
                f"the experimental z-score threshold of {unfamiliar['threshold']:.1f}. "
                "This is not a probability that the prediction is wrong."
            )
        calibration = bundle.get("metrics", {}).get("calibration")
        if calibration:
            st.write("Probability calibration: **evaluated with training-only cross-validation**")
            st.caption(
                f"Saved test-set Brier score: {calibration['brier_score']:.4f}. "
                "Calibration quality is separate from classification accuracy."
            )
        if forest_result is not None:
            logistic_prediction = predict_with_model(result["url"], bundle["model"], bundle)
            comparison = pd.DataFrame([
                {"model": "Logistic Regression", **logistic_prediction},
                {"model": "Random Forest", **forest_result},
            ])
            st.dataframe(comparison, hide_index=True, width="stretch")
            if logistic_prediction["prediction"] == forest_result["prediction"]:
                st.success("Both models agree on the classification. Agreement is not proof of correctness.")
            else:
                st.warning("The models disagree. This URL needs additional scrutiny; neither probability is automatically more reliable.")
        else:
            st.info("Random Forest comparison is unavailable in the saved artifact.")
    explanations = early_explanations
    explanation_frame = pd.DataFrame(explanations)
    explanation_frame["direction"] = explanation_frame["contribution"].map(
        lambda value: "toward phishing" if value > 0 else "toward legitimate"
    )
    scheme_rows = explanation_frame[explanation_frame["feature"].isin(["http", "https"])]
    display_explanation_frame = explanation_frame[~explanation_frame["feature"].isin(["http", "https"])].copy()
    if not scheme_rows.empty:
        display_explanation_frame = pd.concat([
            display_explanation_frame,
            pd.DataFrame([{
                "feature": "scheme indicators (HTTP/HTTPS)",
                "contribution": float(scheme_rows["contribution"].sum()),
                "direction": "toward phishing" if scheme_rows["contribution"].sum() > 0 else "toward legitimate",
            }]),
        ], ignore_index=True)
    st.subheader("Local explanation")
    st.caption(
        f"Method: {explanation_method}. Contributions are in phishing log-odds space, "
        "not percentage points of probability. Logistic Regression only; these terms are relative to "
        "the standardized training baseline."
    )
    toward_phishing, toward_legitimate = st.columns(2)
    with toward_phishing:
        st.markdown("#### Features pushing toward phishing")
        phishing_frame = display_explanation_frame[display_explanation_frame["contribution"] > 0].copy()
        if phishing_frame.empty:
            st.info("No displayed features push this prediction toward phishing.")
        else:
            phishing_frame["contribution"] = phishing_frame["contribution"].map(lambda value: f"{value:.4f}")
            st.dataframe(phishing_frame, hide_index=True, width="stretch")
    with toward_legitimate:
        st.markdown("#### Features pushing toward legitimate")
        legitimate_frame = display_explanation_frame[display_explanation_frame["contribution"] <= 0].copy()
        if legitimate_frame.empty:
            st.info("No displayed features push this prediction toward legitimate.")
        else:
            legitimate_frame["contribution"] = legitimate_frame["contribution"].map(lambda value: f"{value:.4f}")
            st.dataframe(legitimate_frame, hide_index=True, width="stretch")
    chart_frame = display_explanation_frame[["feature", "contribution"]]
    render_contribution_chart(chart_frame, "Local phishing log-odds contributions")
    try:
        waterfall_values, waterfall_method = shap_waterfall_data(
            bundle["model"], result["scaled_features"], bundle["feature_names"]
        )
        waterfall_frame = pd.DataFrame(waterfall_values)
        waterfall_chart = px.bar(
            waterfall_frame.sort_values("value"),
            x="value",
            y="feature",
            orientation="h",
            title=f"SHAP waterfall (top features; {waterfall_method})",
            color="value",
            color_continuous_scale=[PALE_BLUE, WHITE, TEAL],
        )
        st.plotly_chart(style_chart(waterfall_chart), width="stretch")
    except (ImportError, AttributeError, ValueError):
        st.info("A SHAP waterfall is unavailable in this environment; the exact linear contribution chart is shown above.")
    st.caption("Positive contributions increase the model's phishing estimate; negative contributions decrease it. These are statistical signals, not proof of intent.")
    st.subheader("Extracted URL features")
    st.dataframe(pd.DataFrame([result["raw_features"]]).T.rename(columns={0: "value"}), width="stretch")
    with st.expander("URL structure inspector", expanded=True):
        structure = parse_url_structure(result["url"])
        structure_display = pd.DataFrame([
            {"component": key.replace("_", " ").title(), "value": value}
            for key, value in structure.items()
        ])
        # The inspector intentionally contains strings, booleans, integers,
        # and floats. Normalize only this display copy for Arrow conversion;
        # prediction features remain numeric in result["raw_features"].
        structure_display["value"] = structure_display["value"].map(str)
        st.dataframe(structure_display, hide_index=True, width="stretch")
        st.markdown("**Structural observations**")
        for observation in structural_observations(structure):
            st.write(f"- {observation}")
        st.caption("These are URL-structure observations only and are not proof that a URL is malicious.")
    with st.expander("Hypothetical counterfactual analysis"):
        st.caption("Each scenario changes the URL, re-extracts features, and runs the saved model. It does not establish safety.")
        try:
            counterfactuals = compare_counterfactuals(result["url"], bundle)
        except (ValueError, TypeError):
            counterfactuals = []
        if counterfactuals:
            st.dataframe(pd.DataFrame(counterfactuals), hide_index=True, width="stretch")
        else:
            st.info("No constrained URL transformations are available for this input.")


def render_performance(bundle: dict) -> None:
    metrics = bundle.get("metrics", {})
    st.subheader("Model performance")
    report = metrics.get("classification_report", {})
    metric_cards([
        ("Accuracy", f"{metrics.get('accuracy', 0):.3f}"),
        ("Precision", f"{metrics.get('precision', 0):.3f}"),
        ("Recall", f"{metrics.get('recall', 0):.3f}"),
        ("F1-score", f"{metrics.get('f1', 0):.3f}"),
    ])
    left, right = st.columns(2)
    with left:
        matrix = metrics.get("confusion_matrix", [])
        matrix_chart = px.imshow(
            matrix,
            text_auto=True,
            color_continuous_scale=[[0, WHITE], [0.55, PALE_BLUE], [1, TEAL]],
            labels={"x": "Predicted", "y": "Actual"},
            title="Confusion matrix",
        )
        st.plotly_chart(style_chart(matrix_chart), width="stretch")
    with right:
        curve = metrics.get("roc_curve", {})
        figure = go.Figure(go.Scatter(x=curve.get("fpr", []), y=curve.get("tpr", []), mode="lines", name="Logistic Regression"))
        figure.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash"})
        figure.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(style_chart(figure), width="stretch")
    pr = metrics.get("precision_recall_curve", {})
    precision_recall = px.line(
        x=pr.get("recall", []),
        y=pr.get("precision", []),
        labels={"x": "Recall", "y": "Precision"},
        title="Precision-recall curve",
        color_discrete_sequence=[TEAL],
    )
    st.plotly_chart(style_chart(precision_recall), width="stretch")
    forest = metrics.get("random_forest")
    if forest:
        st.dataframe(pd.DataFrame({"Model": ["Logistic Regression", "Random Forest"], "Accuracy": [metrics["accuracy"], forest["accuracy"]], "Precision": [metrics["precision"], forest["precision"]], "Recall": [metrics["recall"], forest["recall"]], "F1": [metrics["f1"], forest["f1"]], "ROC-AUC": [metrics["roc_auc"], forest["roc_auc"]]}), hide_index=True, width="stretch")
        with st.expander("Compare models on the current scanned URL"):
            current = st.session_state.get("last_scan")
            if current:
                logistic = predict_with_model(current["url"], bundle["model"], bundle)
                forest_prediction = predict_with_model(current["url"], bundle["random_forest"], bundle)
                st.dataframe(pd.DataFrame([
                    {"model": "Logistic Regression", **logistic},
                    {"model": "Random Forest", **forest_prediction},
                ]), hide_index=True, width="stretch")
                if logistic["prediction"] != forest_prediction["prediction"]:
                    st.info("The models disagree on this URL. Neither model is automatically more reliable from probability alone.")
                forest_matrix = forest.get("confusion_matrix")
                if forest_matrix:
                    st.plotly_chart(
                        style_chart(px.imshow(
                            forest_matrix,
                            text_auto=True,
                            color_continuous_scale=[[0, WHITE], [0.55, PALE_BLUE], [1, TEAL]],
                            title="Random Forest confusion matrix",
                        )),
                        width="stretch",
                    )
            else:
                st.info("Scan a URL first to compare both saved models on the same input.")
    else:
        st.info("Random Forest comparison is unavailable in this saved artifact. Retrain with the current train.py to generate it.")

def render_explainability(bundle: dict) -> None:
    metrics = bundle.get("metrics", {})
    names = bundle["feature_names"]
    importance = pd.DataFrame({"feature": names, "importance": metrics.get("coefficient_importance", [0] * len(names))})
    importance["absolute"] = importance["importance"].abs()
    st.subheader("Global feature importance")
    coefficient_chart = px.bar(
        importance.nlargest(12, "absolute").sort_values("absolute"),
        x="absolute",
        y="feature",
        orientation="h",
        title="Logistic Regression feature influence",
        color_discrete_sequence=[NAVY],
    )
    st.plotly_chart(style_chart(coefficient_chart), width="stretch")
    permutation = pd.DataFrame({"feature": names, "importance": metrics.get("permutation_importance", [0] * len(names))})
    st.subheader("Permutation importance")
    permutation_chart = px.bar(
        permutation.nlargest(12, "importance").sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title="Permutation importance on the held-out test set",
        color_discrete_sequence=[TEAL],
    )
    st.plotly_chart(style_chart(permutation_chart), width="stretch")
    if "x_test" in bundle:
        shap_values, method = global_shap_importance(bundle["model"], bundle["x_test"], names)
        shap_frame = pd.DataFrame({"feature": names, "mean_absolute_value": shap_values})
        st.subheader("SHAP summary")
        shap_chart = px.bar(
            shap_frame.nlargest(12, "mean_absolute_value").sort_values("mean_absolute_value"),
            x="mean_absolute_value",
            y="feature",
            orientation="h",
            title=f"Global SHAP importance ({method})",
            color_discrete_sequence=[PINK],
        )
        st.plotly_chart(style_chart(shap_chart), width="stretch")
    st.info("The local scanner uses SHAP's LinearExplainer when available. All global plots use actual saved model outputs and the held-out evaluation set.")


def render_generalization() -> None:
    st.subheader("Unseen-domain generalization lab")
    report_path = Path("reports/generalization_metrics.json")
    if not report_path.exists():
        st.info("No domain-held-out experiment is saved yet. Run `python run_generalization.py` to create it.")
        return
    raw = __import__("json").loads(report_path.read_text(encoding="utf-8"))
    st.caption(
        f"{raw['method']}; registrable domains are extracted with "
        f"{raw['registrable_domain_method']}."
    )
    st.dataframe(pd.DataFrame([
        {"metric": "Training rows", "value": raw["train_rows"]},
        {"metric": "Test rows", "value": raw["test_rows"]},
        {"metric": "Training domains", "value": raw["train_domains"]},
        {"metric": "Test domains", "value": raw["test_domains"]},
        {"metric": "Overlapping domains", "value": raw["overlap_domains"]},
        {"metric": "Test domains absent from training (%)", "value": raw["test_domains_absent_from_training_pct"]},
    ]), hide_index=True, width="stretch")
    rows = []
    for model_name, values in raw["models"].items():
        rows.append({"model": model_name.replace("_", " ").title(), **values})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    chart = px.bar(
        pd.DataFrame(rows).melt(id_vars="model", value_vars=["accuracy", "precision", "recall", "f1"]),
        x="model", y="value", color="variable", barmode="group",
        title="Domain-held-out metrics",
        color_discrete_sequence=[NAVY, TEAL, PINK, DANGER],
    )
    st.plotly_chart(style_chart(chart), width="stretch")
    st.caption("This experiment keeps each registrable domain in one partition. It is distinct from the production row-split metrics.")


def render_threshold_analysis(bundle: dict) -> None:
    metrics = bundle.get("metrics", {})
    if "y_test" not in bundle or "test_probabilities" not in bundle:
        st.info("Threshold analysis requires saved test predictions. Retrain with the current train.py.")
        return
    threshold = st.slider("Demonstration threshold", 0.05, 0.95, 0.50, 0.05)
    result = threshold_metrics(bundle["y_test"], bundle["test_probabilities"], threshold)
    metric_cards([
        ("Phishing predictions", str(result["phishing_predictions"])),
        ("Phishing precision", f"{result['precision']:.3f}"),
        ("Phishing recall", f"{result['recall']:.3f}"),
        ("False-negative rate", f"{result['false_negative_rate']:.3f}"),
    ])
    st.caption(
        "This is an interactive threshold demonstration using saved test predictions; "
        "it does not select or tune a production threshold."
    )
    st.markdown(
        "**How to read this:** Lower thresholds flag more URLs as potentially phishing, "
        "which can improve recall but also create more false alarms. Higher thresholds "
        "flag fewer URLs, which can improve precision but may miss more phishing URLs."
    )


def render_reliability(bundle: dict) -> None:
    """Render user-facing reliability metrics from saved evaluation artifacts."""
    metrics = bundle.get("metrics", {})
    report = metrics.get("classification_report", {})
    matrix = metrics.get("confusion_matrix")
    sample_count = len(bundle.get("y_test", []))

    st.subheader("Detection performance")
    if matrix and len(matrix) == 2 and all(len(row) == 2 for row in matrix):
        true_phishing, missed_phishing = matrix[0]
        false_alarms, true_legitimate = matrix[1]
        phishing_precision = true_phishing / max(true_phishing + false_alarms, 1)
        phishing_recall = true_phishing / max(true_phishing + missed_phishing, 1)
        false_negative_rate = missed_phishing / max(true_phishing + missed_phishing, 1)
    elif report:
        phishing_precision = float(report.get("0", {}).get("precision", 0))
        phishing_recall = float(report.get("0", {}).get("recall", 0))
        false_negative_rate = 1 - phishing_recall
    else:
        st.info("Detection performance data is unavailable in the saved artifact.")
        phishing_precision = phishing_recall = false_negative_rate = 0.0
    metric_cards([
        ("Phishing precision", f"{phishing_precision:.3f}"),
        ("Phishing recall", f"{phishing_recall:.3f}"),
        ("False-negative rate", f"{false_negative_rate:.3f}"),
        ("Evaluation samples", f"{sample_count:,}" if sample_count else "Unavailable"),
    ])
    st.caption(
        "These cards describe the saved Logistic Regression classification results at "
        "the training-time default threshold on the stratified test partition."
    )

    calibration = metrics.get("calibration")
    st.subheader("Probability calibration")
    if not calibration:
        st.info("Calibration results are unavailable in this artifact.")
    else:
        metric_cards([("Brier score", f"{calibration['brier_score']:.4f}")])
        st.write("**Model:** Logistic Regression (calibrated output)")
        st.write("**Evaluation dataset:** PhiUSIIL URL dataset, stratified 20% test partition")
        st.write(f"**Evaluated samples:** {sample_count:,}" if sample_count else "**Evaluated samples:** unavailable")
        st.write(f"**Method:** {calibration['method']}")
        st.markdown(
            "The Brier score measures the average squared difference between the "
            "predicted phishing probability and the observed phishing outcome. Lower "
            "is better for probability quality, but it does not guarantee that every "
            "individual prediction is correct."
        )
        predicted = calibration.get("mean_predicted_probability", [])
        observed = calibration.get("fraction_phishing", [])
        if predicted and observed and len(predicted) == len(observed):
            reliability = go.Figure()
            reliability.add_trace(go.Scatter(
                x=predicted,
                y=observed,
                mode="lines+markers",
                name="Observed calibration bins",
            ))
            reliability.add_shape(
                type="line", x0=0, y0=0, x1=1, y1=1,
                line={"dash": "dash", "color": NAVY},
            )
            reliability.update_layout(
                title="Predicted probability versus observed phishing frequency",
                xaxis_title="Mean predicted phishing probability",
                yaxis_title="Observed phishing frequency",
            )
            st.plotly_chart(style_chart(reliability), width="stretch")
            st.caption(
                f"{len(predicted)} quantile calibration bins are available. "
                "The artifact does not include the sample count for each individual bin."
            )
        else:
            st.info("A calibration chart cannot be displayed because valid paired bin statistics are unavailable.")

    st.subheader("Threshold analysis")
    render_threshold_analysis(bundle)
    with st.expander("Advanced Reliability Data"):
        st.caption(
            "Technical diagnostic data from the saved artifact. These values support "
            "the summaries above and are not additional confidence scores."
        )
        if calibration:
            st.json(calibration)
        else:
            st.info("No calibration JSON is available.")


def render_audit(bundle: dict) -> None:
    st.subheader("Dataset and model audit")
    summary = bundle.get("summary", {})
    counts = summary.get("class_counts", {})
    metrics = bundle.get("metrics", {})
    phishing_count = int(counts.get(0, counts.get("0", 0)))
    legitimate_count = int(counts.get(1, counts.get("1", 0)))
    total_records = phishing_count + legitimate_count
    st.caption(
        "Label mapping used by the training code: label 0 = phishing, label 1 = legitimate. "
        "All values below come from the saved dataset and evaluation artifacts."
    )
    if total_records:
        phishing_pct = phishing_count / total_records
        legitimate_pct = legitimate_count / total_records
        metric_cards([
            ("Total usable records", f"{total_records:,}"),
            ("Phishing records", f"{phishing_count:,} ({phishing_pct:.1%})"),
            ("Legitimate records", f"{legitimate_count:,} ({legitimate_pct:.1%})"),
        ])
        composition = pd.DataFrame({
            "class": ["Phishing (label 0)", "Legitimate (label 1)"],
            "records": [phishing_count, legitimate_count],
        })
        composition_chart = px.bar(
            composition,
            x="class",
            y="records",
            color="class",
            title="Dataset class distribution",
            color_discrete_sequence=[DANGER, TEAL],
        )
        st.plotly_chart(style_chart(composition_chart), width="stretch")

    st.subheader("Evaluation methodology")
    st.write(
        "The primary production result uses a stratified random row split: 80% of usable "
        "URLs for training and 20% for testing. Stratification preserves class proportions, "
        "but related URLs or domains can occur in both partitions, so this is not a "
        "domain-independent test."
    )
    domain_split = metrics.get("domain_split")
    if domain_split:
        train_domains = int(domain_split.get("train_unique_domains", 0))
        test_domains = int(domain_split.get("test_unique_domains", 0))
        unseen_domains = int(domain_split.get("unseen_test_domains", 0))
        shared_domains = test_domains - unseen_domains
        domain_rows = [
            {"audit level": "Stored primary audit", "measure": "Training domains", "value": train_domains},
            {"audit level": "Stored primary audit", "measure": "Test domains", "value": test_domains},
            {"audit level": "Stored primary audit", "measure": "Shared test domains", "value": shared_domains},
            {"audit level": "Stored primary audit", "measure": "Unseen test domains", "value": unseen_domains},
            {"audit level": "Stored primary audit", "measure": "Test domains absent from training", "value": f"{unseen_domains / max(test_domains, 1):.1%}"},
        ]
        st.dataframe(
            stringify_value_column(pd.DataFrame(domain_rows)),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "These stored production counts are hostname-level, as recorded by train.py; "
            "they are not a registrable-domain-held-out evaluation."
        )

    generalization_path = Path("reports/generalization_metrics.json")
    if generalization_path.exists():
        generalization = json.loads(generalization_path.read_text(encoding="utf-8"))
        st.subheader("Registrable-domain-held-out audit")
        st.write(
            "A separate experiment groups URLs by registrable domain using the Public Suffix "
            "List and keeps each group in one partition."
        )
        grouped_train = int(generalization["train_domains"])
        grouped_test = int(generalization["test_domains"])
        grouped_overlap = int(generalization["overlap_domains"])
        grouped_unseen = grouped_test - grouped_overlap
        grouped_rows = [
            {"measure": "Training rows", "value": generalization["train_rows"]},
            {"measure": "Test rows", "value": generalization["test_rows"]},
            {"measure": "Training registrable domains", "value": grouped_train},
            {"measure": "Test registrable domains", "value": grouped_test},
            {"measure": "Shared registrable domains", "value": grouped_overlap},
            {"measure": "Unseen test registrable domains", "value": grouped_unseen},
            {"measure": "Domain overlap", "value": f"{grouped_overlap / max(grouped_test, 1):.1%}"},
        ]
        st.dataframe(
            stringify_value_column(pd.DataFrame(grouped_rows)),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            f"Verified relationship: unseen test domains = test domains − shared domains "
            f"({grouped_test} − {grouped_overlap} = {grouped_unseen}). "
            f"Method: {generalization.get('registrable_domain_method', 'saved experiment metadata')}."
        )
    else:
        st.info("The registrable-domain generalization report is not available.")

    st.subheader("Error analysis")
    matrix = metrics.get("confusion_matrix")
    sample_count = len(bundle.get("y_test", []))
    if matrix and len(matrix) == 2 and all(len(row) == 2 for row in matrix):
        true_phishing, false_negatives = matrix[0]
        false_positives, true_legitimate = matrix[1]
        phishing_fpr = false_positives / max(false_positives + true_legitimate, 1)
        phishing_fnr = false_negatives / max(true_phishing + false_negatives, 1)
        error_rows = [
            {"measure": "Phishing false-positive rate", "value": f"{phishing_fpr:.3%}", "count": false_positives, "meaning": "Legitimate URLs incorrectly flagged as phishing"},
            {"measure": "Phishing false-negative rate", "value": f"{phishing_fnr:.3%}", "count": false_negatives, "meaning": "Phishing URLs missed by the model"},
            {"measure": "Evaluation samples", "value": f"{sample_count:,}", "count": sample_count, "meaning": "Rows in the saved primary test partition"},
        ]
        st.dataframe(
            stringify_value_column(pd.DataFrame(error_rows)),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Confusion-matrix order is [label 0 phishing, label 1 legitimate]. "
            "Rows are actual labels and columns are predicted labels."
        )
    else:
        st.info("Confusion-matrix error data is unavailable in the saved artifact.")

    st.subheader("Known limitations")
    st.markdown(
        """
        - The model uses URL structure only; it does not inspect webpage content.
        - Redirect inspection and live reputation checking are not used.
        - The primary production evaluation is row-based, so related domains may overlap.
        - Phishing patterns and domain distributions can change over time.
        - False positives and false negatives are possible.
        - A legitimate classification is not a guarantee that a URL is safe.
        """
    )
    with st.expander("Advanced Audit Data"):
        st.caption("Raw saved audit objects for technical review; the summaries above are the user-facing interpretation.")
        st.json({
            "dataset_summary": summary,
            "primary_metrics": metrics,
            "registrable_domain_experiment": generalization if generalization_path.exists() else None,
        })


def render_batch(bundle: dict) -> None:
    st.subheader("Batch analysis")
    upload = st.file_uploader("Upload a CSV containing URLs", type="csv")
    if upload:
        frame = pd.read_csv(upload)
        column = st.selectbox("URL column", frame.columns)
        if st.button("Analyze batch", type="primary"):
            results, errors = predict_batch(frame[column].fillna("").tolist(), bundle)
            result_frame = pd.DataFrame(results)
            st.metric("Valid URLs analyzed", len(result_frame))
            if errors:
                st.warning(f"{len(errors)} rows could not be analyzed.")
                st.dataframe(pd.DataFrame(errors), hide_index=True, width="stretch")
            st.dataframe(result_frame, hide_index=True, width="stretch")
            st.download_button("Download batch results", result_frame.to_csv(index=False), "browserguard_batch_results.csv", "text/csv")


def render_history() -> None:
    st.subheader("Scan history")
    history = st.session_state["scan_history"]
    if history:
        st.dataframe(pd.DataFrame(history), hide_index=True, width="stretch")
        st.download_button("Export scan history", pd.DataFrame(history).to_csv(index=False), "browserguard_scan_history.csv", "text/csv")
        if st.button("Clear history"):
            st.session_state["scan_history"] = []
            st.session_state["last_scan"] = None
            st.rerun()
    else:
        st.info("No scans have been recorded in this session.")


def render_analytics(bundle: dict) -> None:
    st.title("Analytics & reports")
    performance, generalization, reliability_tab, explainability, batch, history, audit = st.tabs(["Model performance", "Generalization lab", "Reliability", "Explainable AI", "Batch analysis", "Scan history", "Model audit"])
    with performance:
        render_performance(bundle)
    with explainability:
        render_explainability(bundle)
    with generalization:
        render_generalization()
    with reliability_tab:
        render_reliability(bundle)
    with batch:
        render_batch(bundle)
    with history:
        render_history()
    with audit:
        render_audit(bundle)


init_state()
try:
    bundle = load_bundle(MODEL_PATH)
except FileNotFoundError as exc:
    st.error(str(exc), title="Model unavailable")
    st.stop()

page = render_sidebar()
if page == "Home":
    render_home(bundle)
elif page == "URL scanner":
    render_scanner(bundle)
else:
    render_analytics(bundle)
