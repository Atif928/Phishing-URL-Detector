"""Central project configuration.

Every module imports paths and constants from here so that training and
prediction always agree on where files live and how the data is interpreted.

Environment variables (optionally loaded from a local ``.env`` file):

    PHISHING_DATASET_PATH   Path to the raw CSV containing URLs + labels.
    PHISHING_MODEL_PATH     Path to the trained model bundle (.joblib).
    PHISHING_TEST_SIZE      Test split fraction (default 0.2).
    PHISHING_RANDOM_STATE   Random seed (default 42).
"""

from __future__ import annotations

import os
from pathlib import Path

try:  # python-dotenv is optional; the project still works without it.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - only hit when dotenv is not installed
    pass

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
EDA_DIR = RESULTS_DIR / "eda"
DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# The standardised dataset that the pipeline consumes. It must contain a URL
# column and a label column (see docs/dataset_guide.md).
DATASET_PATH = Path(
    os.getenv("PHISHING_DATASET_PATH", RAW_DATA_DIR / "urls.csv")
)

# Where the feature matrix produced by the preprocessing step is cached.
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "url_features.csv"

# Model artefacts.
MODEL_PATH = Path(os.getenv("PHISHING_MODEL_PATH", MODELS_DIR / "phishing_detector.joblib"))
MODEL_METADATA_PATH = MODELS_DIR / "phishing_detector_metadata.json"

# Evaluation artefacts.
MODEL_COMPARISON_CSV = RESULTS_DIR / "model_comparison.csv"
MODEL_COMPARISON_JSON = RESULTS_DIR / "model_comparison.json"
EVALUATION_METRICS_JSON = RESULTS_DIR / "evaluation_metrics.json"
DATASET_SUMMARY_JSON = RESULTS_DIR / "dataset_summary.json"
TRAINING_SUMMARY_JSON = RESULTS_DIR / "training_summary.json"

# --------------------------------------------------------------------------- #
# Machine-learning settings
# --------------------------------------------------------------------------- #
RANDOM_STATE = int(os.getenv("PHISHING_RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("PHISHING_TEST_SIZE", "0.2"))
MAX_URL_LENGTH = 2083  # Common upper bound for browser-compatible URLs.

# Label convention used *inside this project*:
#   1 -> phishing / malicious (the positive class)
#   0 -> legitimate / benign
PHISHING_LABEL = 1
LEGITIMATE_LABEL = 0
LABEL_NAMES = {PHISHING_LABEL: "Phishing", LEGITIMATE_LABEL: "Legitimate"}

# Column names accepted in a user-supplied CSV (case-insensitive matching).
URL_COLUMN_CANDIDATES = ("url", "urls", "link", "uri", "address")
LABEL_COLUMN_CANDIDATES = ("label", "labels", "class", "target", "status", "type", "is_phishing")

# Strings that map to each class when normalising label values.
POSITIVE_LABEL_STRINGS = {
    "1",
    "1.0",
    "true",
    "yes",
    "y",
    "phishing",
    "phish",
    "malicious",
    "malware",
    "bad",
    "unsafe",
    "fraud",
    "defacement",
}
NEGATIVE_LABEL_STRINGS = {
    "0",
    "0.0",
    "false",
    "no",
    "n",
    "legitimate",
    "legit",
    "benign",
    "good",
    "safe",
    "ham",
    "clean",
}

# --------------------------------------------------------------------------- #
# Application text
# --------------------------------------------------------------------------- #
APP_TITLE = "Phishing URL Detector"
APP_SUBTITLE = "Machine Learning-Based URL Security Analysis"
DISCLAIMER = (
    "This tool performs a static, URL-string-only classification using a "
    "machine-learning model trained on a public dataset. It does not visit the "
    "URL, does not analyse page content, and cannot guarantee that a website is "
    "safe or malicious. Always use additional security practices."
)

DATASET_SOURCE = (
    "PhiUSIIL Phishing URL Dataset (UCI Machine Learning Repository, ID 967), "
    "Prasad & Chandra (2024). 235,795 labelled URLs."
)
DATASET_URL = "https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset"
DATASET_LICENSE = "CC BY 4.0 (see docs/dataset_guide.md for attribution)"


def ensure_directories() -> None:
    """Create every directory the project writes to."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        EDA_DIR,
        DOCS_DIR,
        NOTEBOOKS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
