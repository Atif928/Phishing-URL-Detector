"""Load the trained model and classify URL strings.

The prediction layer validates the input, extracts features with the exact same
function used during training, checks the saved feature order and returns a
small, UI-friendly dictionary.

It never contacts the URL being analysed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from config.settings import (
    DISCLAIMER,
    LEGITIMATE_LABEL,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PHISHING_LABEL,
)
from src.feature_extraction import FEATURE_NAMES, extract_url_features
from src.model_training import feature_schema_hash
from src.utils import get_logger, load_json

logger = get_logger(__name__)

_BUNDLE_CACHE: dict[str, tuple[tuple[float, float], object, dict]] = {}


class ModelNotAvailableError(Exception):
    """Raised when the model artefact is missing, unreadable or incompatible."""


def load_model_bundle(
    model_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
):
    """Load ``(model, metadata)`` with validation and caching.

    Raises
    ------
    ModelNotAvailableError
        If the file is missing, corrupted, or was trained with a different
        feature schema.
    """
    model_path = Path(model_path) if model_path else MODEL_PATH
    metadata_path = Path(metadata_path) if metadata_path else MODEL_METADATA_PATH

    if not model_path.exists():
        raise ModelNotAvailableError(
            f"No trained model found at {model_path}. "
            "Train one first with: python -m src.model_training"
        )

    try:
        stat = model_path.stat()
        meta_stat = metadata_path.stat() if metadata_path.exists() else None
    except OSError as exc:
        raise ModelNotAvailableError(f"Could not access the model file: {exc}") from exc

    cache_key = str(model_path.resolve())
    signature = (stat.st_mtime, meta_stat.st_mtime if meta_stat else 0.0)
    cached = _BUNDLE_CACHE.get(cache_key)
    if cached and cached[0] == signature:
        return cached[1], cached[2]

    try:
        model = joblib.load(model_path)
    except Exception as exc:  # corrupted / wrong scikit-learn version
        raise ModelNotAvailableError(
            f"The model file could not be loaded and may be corrupted or incompatible "
            f"with this environment: {model_path} ({exc}). "
            "Retrain it with: python -m src.model_training --force"
        ) from exc

    if not hasattr(model, "predict"):
        raise ModelNotAvailableError(
            f"The file at {model_path} is not a scikit-learn model."
        )

    metadata: dict = {}
    if metadata_path.exists():
        try:
            metadata = load_json(metadata_path)
        except (ValueError, OSError) as exc:
            logger.warning("Could not read model metadata (%s); continuing without it.", exc)
            metadata = {}

    saved_features = metadata.get("feature_names")
    if saved_features:
        if list(saved_features) != list(FEATURE_NAMES):
            raise ModelNotAvailableError(
                "The saved model was trained with a different feature order than the "
                "current code. Retrain it with: python -m src.model_training --force"
            )
        saved_hash = metadata.get("feature_schema_hash")
        if saved_hash and saved_hash != feature_schema_hash(list(saved_features)):
            raise ModelNotAvailableError(
                "Model metadata failed its feature-schema integrity check. "
                "Retrain with: python -m src.model_training --force"
            )
    else:
        logger.warning(
            "Model metadata is missing feature names; assuming the current FEATURE_NAMES order."
        )

    metadata.setdefault("model_name", type(model).__name__)
    metadata.setdefault("feature_names", list(FEATURE_NAMES))
    metadata.setdefault("label_mapping", {"0": "Legitimate", "1": "Phishing"})
    _BUNDLE_CACHE[cache_key] = (signature, model, metadata)
    return model, metadata


def predict_url(url: str, threshold: float = 0.5) -> dict:
    """Classify a single URL.

    Returns a dictionary with the class, probability, model name, extracted
    features and a safety disclaimer.

    Raises
    ------
    ValueError
        For empty or malformed URLs.
    ModelNotAvailableError
        If the trained model cannot be loaded.
    RuntimeError
        If the model returns an unexpected result.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    features = extract_url_features(url)
    model, metadata = load_model_bundle()

    feature_frame = pd.DataFrame([features], columns=list(FEATURE_NAMES))

    try:
        prediction = int(model.predict(feature_frame)[0])
    except Exception as exc:
        raise RuntimeError(f"Model prediction failed: {exc}") from exc

    phishing_probability = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(feature_frame)[0]
            classes = list(model.classes_)
            phishing_probability = float(probabilities[classes.index(PHISHING_LABEL)])
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("predict_proba failed (%s); falling back to predict().", exc)

    if phishing_probability is not None:
        prediction = PHISHING_LABEL if phishing_probability >= threshold else LEGITIMATE_LABEL

    is_phishing = prediction == PHISHING_LABEL
    label_name = "Phishing" if is_phishing else "Legitimate"
    if phishing_probability is not None:
        confidence = phishing_probability if is_phishing else 1.0 - phishing_probability
    else:
        confidence = None

    return {
        "url": url,
        "is_phishing": is_phishing,
        "label": prediction,
        "label_name": label_name,
        "phishing_probability": phishing_probability,
        "legitimate_probability": (
            None if phishing_probability is None else 1.0 - phishing_probability
        ),
        "confidence": confidence,
        "threshold": threshold,
        "model_name": metadata.get("model_name", type(model).__name__),
        "features": features,
        "disclaimer": DISCLAIMER,
    }


def predict_urls(urls: list[str], threshold: float = 0.5) -> list[dict]:
    """Classify several URLs, collecting per-URL errors instead of raising."""
    results: list[dict] = []
    for url in urls:
        try:
            results.append(predict_url(url, threshold=threshold))
        except (ValueError, RuntimeError) as exc:
            results.append({"url": url, "error": str(exc)})
    return results


def get_model_info() -> dict:
    """Return metadata for the UI (raises ModelNotAvailableError if missing)."""
    _, metadata = load_model_bundle()
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a URL with the trained model.")
    parser.add_argument("url", help="URL to analyse (it is never fetched)")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    try:
        result = predict_url(args.url, threshold=args.threshold)
    except (ValueError, ModelNotAvailableError, RuntimeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    print(f"URL:        {result['url']}")
    print(f"Prediction: {result['label_name']}")
    if result["phishing_probability"] is not None:
        print(f"Phishing probability: {result['phishing_probability']:.4f}")
    print(f"Model:      {result['model_name']}")
    print(f"\n{result['disclaimer']}")


if __name__ == "__main__":
    main()
