"""Streamlit front-end for the Phishing URL Detector.

Run with:  streamlit run app.py

The app only performs static URL-string classification. It never sends a
request to the submitted URL, never executes page scripts and never calls a
reputation API.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from config.settings import (
    APP_SUBTITLE,
    APP_TITLE,
    DATASET_LICENSE,
    DATASET_SOURCE,
    DATASET_URL,
    DISCLAIMER,
    RESULTS_DIR,
    ensure_directories,
)
from src.feature_extraction import (
    FEATURE_NAMES,
    SUSPICIOUS_TLDS,
    URL_SHORTENER_DOMAINS,
    get_feature_descriptions,
    normalize_url,
)
from src.prediction import ModelNotAvailableError, load_model_bundle, predict_url
from src.utils import load_json

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_directories()


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_model_bundle():
    """Load the trained model once per server process."""
    return load_model_bundle()


@st.cache_data(show_spinner=False)
def get_evaluation() -> dict | None:
    path = RESULTS_DIR / "evaluation_metrics.json"
    if not path.exists():
        return None
    try:
        return load_json(path)
    except (OSError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def get_dataset_summary() -> dict | None:
    path = RESULTS_DIR / "dataset_summary.json"
    if not path.exists():
        return None
    try:
        return load_json(path)
    except (OSError, ValueError):
        return None


def read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.6rem; max-width: 1150px; }
      .hero {
          background: linear-gradient(120deg, #0f2027 0%, #203a43 55%, #2c5364 100%);
          padding: 1.6rem 1.8rem; border-radius: 14px; color: #f8f9fa;
          margin-bottom: 1.2rem;
      }
      .hero h1 { color: #ffffff; font-size: 2.05rem; margin: 0 0 .25rem 0; }
      .hero p  { color: #cfe3ea; margin: 0; font-size: 1.02rem; }
      .banner-phish {
          background: #fdecea; border-left: 6px solid #c0392b;
          padding: 1rem 1.2rem; border-radius: 8px; color: #7b241c;
      }
      .banner-safe {
          background: #eafaf1; border-left: 6px solid #1e8449;
          padding: 1rem 1.2rem; border-radius: 8px; color: #145a32;
      }
      .banner-phish h3, .banner-safe h3 { margin: 0; }
      div[data-testid="stMetric"] {
          background: #f7f9fa; border: 1px solid #e3e8eb;
          padding: .7rem .9rem; border-radius: 10px;
      }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Shared components
# --------------------------------------------------------------------------- #
def metric_row(metrics: dict, keys: tuple[tuple[str, str], ...]) -> None:
    columns = st.columns(len(keys))
    for column, (label, key) in zip(columns, keys):
        value = metrics.get(key)
        text = "n/a" if value is None else f"{float(value):.4f}"
        column.metric(label, text)


def render_sidebar() -> str:
    st.sidebar.markdown("## 🛡️ Phishing URL Detector")
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "URL Prediction", "Model Performance", "About"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()

    try:
        _, metadata = get_model_bundle()
        st.sidebar.success(f"Model ready: {metadata.get('model_name', 'unknown')}")
        metrics = metadata.get("metrics", {})
        if metrics:
            st.sidebar.caption(
                f"F1 {metrics.get('f1_score', 0):.3f} · "
                f"Recall {metrics.get('recall', 0):.3f} · "
                f"Precision {metrics.get('precision', 0):.3f}"
            )
    except ModelNotAvailableError:
        st.sidebar.warning("No trained model found.")
        st.sidebar.caption("Run `python -m src.model_training` first.")

    st.sidebar.divider()
    st.sidebar.caption(
        "Static URL analysis only. The submitted URL is never opened, fetched "
        "or executed."
    )
    return page


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_dashboard() -> None:
    hero(APP_TITLE, APP_SUBTITLE)
    st.markdown(
        "Paste a URL and this application extracts **38 lexical features** and "
        "classifies it with a Scikit-learn model trained on labelled phishing data. "
        "Everything runs locally — no URL is ever visited."
    )

    evaluation = get_evaluation()
    summary = get_dataset_summary()

    try:
        _, metadata = get_model_bundle()
        metrics = metadata.get("metrics", {})
        model_name = metadata.get("model_name", "n/a")
        model_ready = True
    except ModelNotAvailableError as exc:
        metadata = {}
        metrics = {}
        model_name = "not trained"
        model_ready = False
        st.error(str(exc))

    st.markdown("### Model at a glance")
    metric_row(
        metrics,
        (
            ("Accuracy", "accuracy"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1-score", "f1_score"),
            ("ROC-AUC", "roc_auc"),
        ),
    )

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### What is a phishing URL?")
        st.markdown(
            "A phishing URL is crafted to look trustworthy while leading to a fake "
            "login page, malware download or payment scam. Typical tricks include "
            "brand names on unofficial domains, misspellings, excessive hyphens or "
            "subdomains, unusual top-level domains, IP-address hosts and "
            "credential-related keywords such as `verify` or `login`."
        )
        st.markdown("#### How this project works")
        st.markdown(
            "1. **Dataset** — labelled URLs from a public research dataset.\n"
            "2. **Feature engineering** — 38 numeric signals extracted from the URL string.\n"
            "3. **Training** — Logistic Regression, Decision Tree and Random Forest.\n"
            "4. **Evaluation** — held-out test set, stratified 80/20 split.\n"
            "5. **Prediction** — the best model scores a new URL locally."
        )
    with right:
        st.markdown("#### Project facts")
        rows = {
            "Selected model": model_name,
            "Features": f"{len(FEATURE_NAMES)}",
            "Training URLs": (
                f"{metadata.get('training_rows', 0):,}" if model_ready else "n/a"
            ),
            "Test URLs": f"{metadata.get('test_rows', 0):,}" if model_ready else "n/a",
            "Dataset": "PhiUSIIL (UCI ML Repository)",
            "Total labelled URLs": (
                f"{summary.get('rows_final', 0):,}" if summary else "n/a"
            ),
        }
        st.table(pd.DataFrame(rows.items(), columns=["Item", "Value"]).set_index("Item"))

    if evaluation:
        st.caption(
            "Metrics above are measured on the held-out test split and are not "
            "guaranteed to generalise to new, unseen URLs."
        )

    st.info(DISCLAIMER)


def page_predict() -> None:
    hero("URL Prediction", "Analyse a single URL with the trained classifier")

    st.markdown(
        "Enter a full URL (for example `https://www.example.com/login`). "
        "A scheme is added automatically if you omit it. The URL is validated "
        "and parsed locally — **no network request is made**."
    )

    with st.form("predict_form"):
        url_input = st.text_input(
            "URL to analyse",
            placeholder="https://www.example.com/login",
            help="Only http:// and https:// URLs are supported.",
        )
        submitted = st.form_submit_button("Analyze URL", type="primary")

    if submitted:
        st.session_state.pop("prediction", None)
        st.session_state.pop("prediction_error", None)
        raw = (url_input or "").strip()

        if not raw:
            st.session_state["prediction_error"] = "Please enter a URL."
        else:
            try:
                normalize_url(raw)  # validation only
                with st.spinner("Extracting features and scoring the URL ..."):
                    st.session_state["prediction"] = predict_url(raw)
            except ValueError as exc:
                st.session_state["prediction_error"] = f"Invalid URL: {exc}"
            except ModelNotAvailableError as exc:
                st.session_state["prediction_error"] = str(exc)
            except Exception as exc:  # defensive: never crash the UI
                st.session_state["prediction_error"] = f"Prediction failed: {exc}"

    error = st.session_state.get("prediction_error")
    if error:
        st.error(error)

    result = st.session_state.get("prediction")
    if result:
        render_prediction(result)

    st.divider()
    st.caption(DISCLAIMER)


def render_prediction(result: dict) -> None:
    st.markdown("### Result")

    if result["is_phishing"]:
        st.markdown(
            '<div class="banner-phish"><h3>Potential Phishing URL Detected</h3>'
            "This URL shares patterns with phishing URLs in the training data. "
            "Do not enter credentials or payment details.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="banner-safe"><h3>Predicted as Legitimate URL</h3>'
            "This URL shares patterns with legitimate URLs in the training data. "
            "This is not proof that the site is safe.</div>",
            unsafe_allow_html=True,
        )

    st.write("")
    cols = st.columns(3)
    probability = result.get("phishing_probability")
    cols[0].metric(
        "Phishing probability",
        "n/a" if probability is None else f"{probability * 100:.2f}%",
    )
    cols[1].metric("Predicted class", result["label_name"])
    cols[2].metric("Model", result["model_name"])

    if probability is not None:
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(
            "The score is the model's estimated probability for the phishing class. "
            "It is not a guarantee and should not be used as the sole security check."
        )

    with st.expander("Extracted URL features", expanded=True):
        descriptions = get_feature_descriptions()
        table = pd.DataFrame(
            {
                "Feature": list(result["features"].keys()),
                "Value": list(result["features"].values()),
                "Description": [descriptions.get(name, "") for name in result["features"]],
            }
        )
        st.dataframe(table, width="stretch", hide_index=True)
        st.download_button(
            "Download features as CSV",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="url_features.csv",
            mime="text/csv",
        )

    st.caption(
        "Analysed URL: " + result["url"] + " — the URL was not opened or requested."
    )


def page_performance() -> None:
    hero("Model Performance", "Evaluation on the held-out test set")

    evaluation = get_evaluation()
    if not evaluation:
        st.warning(
            "No evaluation results found. Train and evaluate the models first:\n\n"
            "```\npython -m src.model_training\npython -m src.model_evaluation\n```"
        )
        return

    comparison = pd.DataFrame(evaluation.get("comparison", []))
    if comparison.empty:
        st.warning("Evaluation file is present but contains no comparison rows.")
        return

    st.markdown("### Selected model")
    selected = evaluation.get("selected_model", "n/a")
    st.markdown(f"**{selected}** — {evaluation.get('selection_rule', '')}")
    detailed = evaluation.get("detailed_metrics", {}).get(selected, {})
    metric_row(
        detailed,
        (
            ("Accuracy", "accuracy"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1-score", "f1_score"),
            ("ROC-AUC", "roc_auc"),
        ),
    )
    st.caption(
        f"Evaluated on {evaluation.get('test_rows', 0):,} held-out URLs: "
        f"{evaluation.get('test_class_distribution', {}).get('legitimate', 0):,} legitimate, "
        f"{evaluation.get('test_class_distribution', {}).get('phishing', 0):,} phishing."
    )

    st.markdown("### Model comparison")
    display = comparison.copy()
    display = display[
        ["rank", "model", "accuracy", "precision", "recall", "f1_score", "roc_auc",
         "train_time_seconds"]
    ]
    for column in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    display["train_time_seconds"] = display["train_time_seconds"].map(
        lambda value: f"{value:.2f}s"
    )
    display.columns = [
        "Rank", "Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Train time"
    ]
    st.dataframe(display, width="stretch", hide_index=True)

    st.markdown("### Confusion matrix")
    st.caption(
        "Rows are the true class, columns are the predicted class. The top-right "
        "cell counts legitimate URLs flagged as phishing (false alarms); the "
        "bottom-left cell counts missed phishing URLs."
    )
    matrix = detailed.get("confusion_matrix")
    if matrix:
        matrix_df = pd.DataFrame(
            matrix,
            index=["Actual: Legitimate", "Actual: Phishing"],
            columns=["Predicted: Legitimate", "Predicted: Phishing"],
        )
        st.dataframe(matrix_df, width="content")

    fig_cols = st.columns(2)
    cm_path = RESULTS_DIR / f"confusion_matrix_{selected.lower().replace(' ', '_')}.png"
    roc_path = RESULTS_DIR / "roc_curves.png"
    if cm_path.exists():
        fig_cols[0].image(str(cm_path), caption=f"Confusion matrix — {selected}")
    if roc_path.exists():
        fig_cols[1].image(str(roc_path), caption="ROC curves for all models")

    metrics_path = RESULTS_DIR / "metrics_comparison.png"
    if metrics_path.exists():
        st.image(str(metrics_path), caption="Metric comparison across models")

    st.markdown("### Classification report")
    report_path = RESULTS_DIR / f"classification_report_{selected.lower().replace(' ', '_')}.txt"
    report = read_text(report_path)
    if report:
        st.code(report, language="text")
    else:
        st.caption("Classification report file not found.")

    st.markdown("### How to read these metrics")
    st.markdown(
        "- **Accuracy** — share of all test URLs classified correctly. Misleading "
        "when classes are imbalanced.\n"
        "- **Precision** — of the URLs flagged as phishing, how many really were "
        "phishing? Low precision means many false alarms.\n"
        "- **Recall** — of all real phishing URLs, how many were caught? Low "
        "recall means dangerous misses.\n"
        "- **F1-score** — harmonic mean of precision and recall.\n"
        "- **ROC-AUC** — how well the score separates the classes across all "
        "thresholds (1.0 is perfect, 0.5 is random)."
    )


def page_about() -> None:
    hero("About This Project", "Objective, workflow, data and limitations")

    st.markdown("### Objective")
    st.markdown(
        "Build an educational, end-to-end machine learning system that classifies "
        "a URL as **phishing/malicious** or **legitimate/benign** using only "
        "lexical features extracted from the URL string."
    )

    st.markdown("### Machine learning workflow")
    st.markdown(
        "```\nDataset -> cleaning -> feature extraction -> EDA -> stratified split\n"
        "        -> model training -> evaluation -> model selection\n"
        "        -> model persistence -> Streamlit prediction\n```"
    )

    st.markdown("### Dataset")
    st.markdown(
        f"**Source:** {DATASET_SOURCE}\n\n"
        f"**Link:** {DATASET_URL}\n\n"
        f"**License / usage:** {DATASET_LICENSE}\n\n"
        "The raw dataset provides real URL strings, so every feature used by the "
        "model is computed by this project rather than copied from a precomputed "
        "feature table. Labels were converted to `1 = phishing`, `0 = legitimate`."
    )

    st.markdown("### Feature engineering")
    st.markdown(
        f"All **{len(FEATURE_NAMES)}** features are derived from the URL text:"
    )
    st.markdown(
        "- **Length and structure** — URL, hostname, path and query length; dots, "
        "slashes, hyphens, underscores, digits and special characters.\n"
        "- **Host analysis** — subdomain count, IP-address host, punycode, "
        "explicit/unusual port, long hostname.\n"
        "- **Lexical risk signals** — credential keywords, brand names on "
        "unofficial domains, URL shorteners, suspicious TLDs, percent-encoding "
        "and obfuscation patterns."
    )
    st.markdown(
        f"Suspicious TLDs ({len(SUSPICIOUS_TLDS)}) and URL shorteners "
        f"({len(URL_SHORTENER_DOMAINS)}) are fixed lookup lists in "
        "`src/feature_extraction.py`."
    )

    st.markdown("### Models")
    st.markdown(
        "- **Logistic Regression** — linear baseline with `StandardScaler` in a Pipeline.\n"
        "- **Decision Tree** — interpretable non-linear model.\n"
        "- **Random Forest** — ensemble of trees; usually the strongest of the three.\n\n"
        "All models use balanced class weights and are compared on the same test set."
    )

    st.markdown("### Limitations")
    st.markdown(
        "1. **URL-only analysis.** Page content, JavaScript, certificates, redirects, "
        "WHOIS data and hosting reputation are not inspected.\n"
        "2. **Dataset bias.** In PhiUSIIL, all legitimate URLs use HTTPS while about "
        "half of phishing URLs use HTTP. The model may rely heavily on this shortcut. "
        "HTTPS does not make a site trustworthy.\n"
        "3. **Path bias.** Legitimate examples are mostly homepages, so URLs with "
        "paths such as `/login` can be over-flagged.\n"
        "4. **Temporal drift.** Phishing techniques and URL patterns change; the model "
        "must be retrained on fresh data.\n"
        "5. **Not a security product.** False positives and false negatives are "
        "inevitable. Never use this tool as your only protection.\n"
        "6. **No guarantee.** A 'legitimate' prediction is not a statement that the "
        "website is safe."
    )

    st.markdown("### Privacy and safety")
    st.markdown(
        "- Submitted URLs are never opened, fetched or executed.\n"
        "- No HTTP requests, reputation APIs or third-party services are used at "
        "prediction time.\n"
        "- Input is validated and length-limited before parsing."
    )

    st.info(DISCLAIMER)


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
page = render_sidebar()

if page == "Dashboard":
    page_dashboard()
elif page == "URL Prediction":
    page_predict()
elif page == "Model Performance":
    page_performance()
else:
    page_about()
