"""Tests for dataset loading, cleaning and splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DatasetError, normalize_label, standardize_frame
from src.feature_extraction import FEATURE_NAMES
from src.model_training import build_models
from src.preprocessing import (
    build_feature_frame,
    class_distribution,
    clean_url_frame,
    load_processed_dataset,
    split_dataset,
)


def make_raw_frame(rows):
    return pd.DataFrame(rows, columns=["url", "label"])


def test_normalize_label_values():
    assert normalize_label(1) == 1
    assert normalize_label(0) == 0
    assert normalize_label("1") == 1
    assert normalize_label("0") == 0
    assert normalize_label("phishing") == 1
    assert normalize_label("Phishing") == 1
    assert normalize_label("legitimate") == 0
    assert normalize_label("benign") == 0
    assert normalize_label(True) == 1
    assert normalize_label(False) == 0
    assert normalize_label("unknown-class") is None
    assert normalize_label(None) is None


def test_standardize_frame_accepts_link_as_url_column():
    frame = pd.DataFrame({"link": ["https://example.com"], "label": [1]})
    # "link" is accepted as a URL column name.
    standard, _ = standardize_frame(frame)
    assert list(standard.columns) == ["url", "label"]


def test_standardize_frame_rejects_missing_label_column():
    frame = pd.DataFrame({"url": ["https://example.com"], "something": [1]})
    with pytest.raises(DatasetError, match="label"):
        standardize_frame(frame)


def test_standardize_frame_rejects_empty_dataset():
    with pytest.raises(DatasetError):
        standardize_frame(pd.DataFrame(columns=["url", "label"]))


def test_phiusiil_label_convention_is_inverted():
    """PhiUSIIL uses 1=legitimate, 0=phishing; the project uses the opposite."""
    frame = pd.DataFrame(
        {
            "FILENAME": ["a.txt", "b.txt"],
            "URL": ["https://www.uni-mainz.de", "http://www.f0519141.xsph.ru"],
            "URLLength": [24, 30],
            "URLSimilarityIndex": [100, 3],
            "label": [1, 0],
        }
    )
    standard, report = standardize_frame(frame, source="test")
    labels = dict(zip(standard["url"], standard["label"]))
    assert labels["https://www.uni-mainz.de"] == 0
    assert labels["http://www.f0519141.xsph.ru"] == 1
    assert report.class_counts == {"legitimate": 1, "phishing": 1}
    assert any("PhiUSIIL" in note for note in report.notes)


def test_clean_url_frame_removes_missing_duplicate_and_invalid_rows():
    frame = make_raw_frame(
        [
            ("https://www.example.com", 0),
            ("https://www.example.com", 1),
            ("HTTPS://WWW.EXAMPLE.COM", 0),  # duplicate ignoring case
            ("not a url", 0),  # whitespace -> invalid
            ("", 1),  # missing
            (None, 0),  # missing
            ("http://192.168.1.1/login", 1),
        ]
    )
    cleaned, stats = clean_url_frame(frame)
    assert len(cleaned) == 2
    assert stats["duplicate_urls"] == 2
    assert stats["invalid_urls"] == 1
    # None and "" are both counted as missing URLs.
    assert stats["missing_urls"] == 2
    assert set(cleaned["url"]) == {"https://www.example.com", "http://192.168.1.1/login"}
    assert cleaned["label"].isin([0, 1]).all()


def test_build_feature_frame_returns_expected_columns():
    frame = make_raw_frame(
        [
            ("https://www.example.com", 0),
            ("http://192.168.1.1/login", 1),
        ]
    )
    featured, invalid = build_feature_frame(frame)
    assert invalid == []
    assert len(featured) == 2
    assert set(FEATURE_NAMES).issubset(featured.columns)
    assert {"url", "label"}.issubset(featured.columns)
    assert featured.shape[1] == len(FEATURE_NAMES) + 2


def test_build_feature_frame_keeps_urls_aligned_with_features():
    frame = make_raw_frame(
        [
            ("https://www.example.com", 0),
            ("http://example.com/login", 1),
            ("https://bit.ly/abc", 0),
        ]
    )
    featured, invalid = build_feature_frame(frame)
    assert invalid == []
    for _, row in featured.iterrows():
        assert row["url_length"] == len(row["url"])


def test_split_dataset_is_stratified_and_reproducible():
    rng = np.random.default_rng(42)
    n = 200
    frame = pd.DataFrame(
        {
            "url": [f"https://example{i}.com" for i in range(n)],
            "label": [1] * 120 + [0] * 80,
        }
    )
    for feature in FEATURE_NAMES:
        frame[feature] = rng.random(n)

    X_train, X_test, y_train, y_test = split_dataset(frame, test_size=0.2, random_state=42)
    assert len(X_train) == 160
    assert len(X_test) == 40
    assert list(X_train.columns) == list(FEATURE_NAMES)
    assert list(X_test.columns) == list(FEATURE_NAMES)
    assert abs(y_test.mean() - y_train.mean()) < 0.02  # stratification
    assert len(set(X_train.index) & set(X_test.index)) == 0
    assert len(set(X_train.index) | set(X_test.index)) == n

    X_train2, X_test2, _, _ = split_dataset(frame, test_size=0.2, random_state=42)
    pd.testing.assert_frame_equal(X_train, X_train2)
    pd.testing.assert_frame_equal(X_test, X_test2)


def test_split_dataset_requires_both_classes():
    frame = pd.DataFrame(
        {
            "url": ["https://a.com", "https://b.com"],
            "label": [1, 1],
            **{feature: [0, 0] for feature in FEATURE_NAMES},
        }
    )
    with pytest.raises(ValueError, match="both"):
        split_dataset(frame)


def test_class_distribution():
    target = pd.Series([1, 1, 1, 0, 0])
    distribution = class_distribution(target)
    assert distribution["total"] == 5
    assert distribution["phishing"] == 3
    assert distribution["legitimate"] == 2
    assert distribution["phishing_ratio"] == pytest.approx(0.6)


def test_logistic_regression_scales_inside_the_pipeline():
    """Scaling must be fitted with the model, never on the full dataset."""
    pipeline = build_models()["Logistic Regression"]
    assert "scaler" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_processed_dataset_matches_feature_schema():
    """Integration check against the real artefacts, when they exist."""
    try:
        frame = load_processed_dataset()
    except FileNotFoundError:
        pytest.skip("Processed dataset not built yet")
    assert len(frame) > 0
    assert set(FEATURE_NAMES).issubset(frame.columns)
    assert set(frame["label"].unique()) <= {0, 1}
