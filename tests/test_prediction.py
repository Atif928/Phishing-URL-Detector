"""Tests for model loading and URL prediction."""

from __future__ import annotations

import joblib
import pytest

from config.settings import MODEL_METADATA_PATH, MODEL_PATH
from src.feature_extraction import FEATURE_NAMES
from src.prediction import (
    ModelNotAvailableError,
    load_model_bundle,
    predict_url,
    predict_urls,
)

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Trained model not found; run 'python -m src.model_training' first.",
)


@pytest.fixture(scope="module")
def bundle():
    return load_model_bundle()


def test_model_bundle_loads_with_matching_schema(bundle):
    model, metadata = bundle
    assert hasattr(model, "predict")
    assert list(metadata["feature_names"]) == list(FEATURE_NAMES)
    assert metadata["feature_count"] == len(FEATURE_NAMES)
    assert metadata["model_name"]


def test_predict_returns_expected_structure():
    result = predict_url("https://www.example.com")
    expected_keys = {
        "url",
        "is_phishing",
        "label",
        "label_name",
        "phishing_probability",
        "legitimate_probability",
        "confidence",
        "threshold",
        "model_name",
        "features",
        "disclaimer",
    }
    assert expected_keys.issubset(result.keys())
    assert isinstance(result["is_phishing"], bool)
    assert result["label"] in (0, 1)
    assert result["label_name"] in ("Phishing", "Legitimate")
    assert tuple(result["features"].keys()) == FEATURE_NAMES
    assert "not proof" in result["disclaimer"].lower() or "cannot guarantee" in result["disclaimer"].lower()


def test_probabilities_are_valid():
    result = predict_url("https://www.example.com/login")
    probability = result["phishing_probability"]
    assert probability is not None
    assert 0.0 <= probability <= 1.0
    assert result["legitimate_probability"] == pytest.approx(1.0 - probability)
    assert result["confidence"] == pytest.approx(
        probability if result["is_phishing"] else 1.0 - probability
    )
    assert 0.0 <= result["confidence"] <= 1.0


def test_label_matches_probability_and_threshold():
    result = predict_url("https://www.example.com", threshold=0.5)
    assert result["is_phishing"] == (result["phishing_probability"] >= 0.5)
    assert result["label"] == (1 if result["is_phishing"] else 0)


def test_known_legitimate_url_is_not_flagged():
    result = predict_url("https://www.uni-mainz.de")
    assert result["is_phishing"] is False


def test_known_phishing_url_is_flagged():
    result = predict_url("http://www.f0519141.xsph.ru")
    assert result["is_phishing"] is True


def test_obviously_suspicious_url_scores_high():
    result = predict_url("http://secure-paypal-login.verify-account.tk/update.php?id=99")
    assert result["phishing_probability"] > 0.5


@pytest.mark.parametrize("url", ["", "   ", "javascript:alert(1)", "https://exa mple.com"])
def test_invalid_urls_raise_value_error(url):
    with pytest.raises(ValueError):
        predict_url(url)


def test_invalid_threshold_raises():
    with pytest.raises(ValueError, match="threshold"):
        predict_url("https://www.example.com", threshold=1.5)


def test_prediction_is_deterministic():
    first = predict_url("https://www.example.com/path?q=1")
    second = predict_url("https://www.example.com/path?q=1")
    assert first["phishing_probability"] == second["phishing_probability"]


def test_predict_urls_collects_errors():
    results = predict_urls(["https://www.example.com", "bad url"])
    assert len(results) == 2
    assert "error" in results[1]
    assert results[1]["url"] == "bad url"


def test_missing_model_file_raises(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="No trained model"):
        load_model_bundle(model_path=tmp_path / "missing.joblib")


def test_corrupted_model_file_raises(tmp_path):
    broken = tmp_path / "broken.joblib"
    broken.write_bytes(b"this is not a joblib file")
    with pytest.raises(ModelNotAvailableError, match="corrupted|compatible"):
        load_model_bundle(model_path=broken, metadata_path=tmp_path / "none.json")


def test_feature_schema_mismatch_is_rejected(tmp_path):
    import json

    model_copy = tmp_path / "model.joblib"
    joblib.dump(joblib.load(MODEL_PATH), model_copy)

    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    metadata["feature_names"] = list(reversed(FEATURE_NAMES))
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ModelNotAvailableError, match="feature order"):
        load_model_bundle(model_path=model_copy, metadata_path=metadata_path)
