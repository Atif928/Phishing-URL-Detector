"""Tests for URL validation and feature engineering."""

from __future__ import annotations

import math

import pytest

from src.feature_extraction import (
    FEATURE_NAMES,
    extract_features_batch,
    extract_url_features,
    is_valid_url,
    normalize_url,
)


def test_feature_schema_order_and_count():
    features = extract_url_features("https://www.example.com/login")
    assert tuple(features.keys()) == FEATURE_NAMES
    assert len(features) == 38


def test_all_feature_values_are_numeric_and_finite():
    features = extract_url_features("https://sub.example.co.uk/a/b?x=1&y=2")
    for name, value in features.items():
        assert isinstance(value, (int, float)), f"{name} is not numeric"
        assert not (isinstance(value, float) and math.isnan(value))


def test_counts_on_a_known_url():
    url = "https://www.example.com/login?user=alice&next=/home"
    features = extract_url_features(url)
    assert features["url_length"] == len(url)
    assert features["num_dots"] == 2
    assert features["num_question_marks"] == 1
    assert features["num_equal_signs"] == 2
    assert features["num_ampersands"] == 1
    assert features["num_subdomains"] == 1
    assert features["path_depth"] == 1
    assert features["num_params"] == 2
    assert features["has_https"] == 1


def test_https_detection():
    assert extract_url_features("https://example.com")["has_https"] == 1
    assert extract_url_features("http://example.com")["has_https"] == 0
    # A scheme-less URL is treated as http:// for analysis.
    assert extract_url_features("example.com")["has_https"] == 0


def test_ip_address_detection():
    assert extract_url_features("http://192.168.1.1/admin")["has_ip_address"] == 1
    assert extract_url_features("https://example.com/admin")["has_ip_address"] == 0
    # IPv6 hosts are also recognised.
    assert extract_url_features("http://[2001:db8::1]/")["has_ip_address"] == 1


def test_url_length_is_exactly_the_normalized_length():
    assert extract_url_features("https://example.com/a")["url_length"] == len(
        "https://example.com/a"
    )
    # Missing scheme -> http:// is prepended.
    assert extract_url_features("example.com/a")["url_length"] == len("http://example.com/a")


def test_suspicious_keyword_detection():
    assert extract_url_features("http://example.com/verify-account")["has_suspicious_keyword"] == 1
    assert extract_url_features("http://example.com/accountant")["has_suspicious_keyword"] == 0
    assert extract_url_features("https://example.com")["has_suspicious_keyword"] == 0


def test_url_shortener_detection():
    assert extract_url_features("https://bit.ly/abc123")["has_url_shortener"] == 1
    assert extract_url_features("https://www.bit.ly/abc123")["has_url_shortener"] == 1
    assert extract_url_features("https://example.com")["has_url_shortener"] == 0


def test_brand_impostor_detection():
    assert extract_url_features("https://www.paypal.com/signin")["has_brand_term"] == 0
    impostor = extract_url_features("https://paypal-account-verify.tk/login")
    assert impostor["has_brand_term"] == 1
    assert impostor["suspicious_tld"] == 1


def test_port_features():
    features = extract_url_features("http://example.com:8080/admin")
    assert features["has_port"] == 1
    assert features["has_unusual_port"] == 1
    standard = extract_url_features("https://example.com:443/")
    assert standard["has_port"] == 1
    assert standard["has_unusual_port"] == 0


def test_subdomain_counting():
    assert extract_url_features("https://example.com")["num_subdomains"] == 0
    assert extract_url_features("https://www.example.com")["num_subdomains"] == 1
    assert extract_url_features("https://a.b.example.com")["num_subdomains"] == 2


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "not a url",
        "https://exa mple.com",
        "javascript:alert(1)",
        "data:text/html;base64,PHNjcmlwdD4=",
        "file:///C:/Windows/System32/drivers/etc/hosts",
        "http://",
        "http://-bad.example.com",
        "https://example.com/" + "a" * 3000,
    ],
)
def test_invalid_urls_raise_value_error(url):
    with pytest.raises(ValueError):
        extract_url_features(url)


def test_empty_url_raises_specific_message():
    with pytest.raises(ValueError, match="empty"):
        extract_url_features("   ")


def test_is_valid_url_matches_validation():
    assert is_valid_url("https://www.example.com") is True
    assert is_valid_url("example.com") is True
    assert is_valid_url("") is False
    assert is_valid_url(None) is False


def test_normalize_url_adds_scheme_and_strips_whitespace():
    assert normalize_url("  example.com/path  ") == "http://example.com/path"
    assert normalize_url("https://example.com") == "https://example.com"


def test_feature_extraction_is_deterministic():
    url = "https://secure-paypal-login.verify-account.tk/update?id=1"
    assert extract_url_features(url) == extract_url_features(url)


def test_batch_extraction_reports_invalid_rows():
    urls = ["https://example.com", "not a url", "http://192.168.0.1/x"]
    frame, invalid = extract_features_batch(urls)
    assert len(frame) == 2
    assert list(frame.columns) == list(FEATURE_NAMES)
    assert len(invalid) == 1
    assert invalid[0][0] == "not a url"


def test_no_network_access_is_performed(monkeypatch):
    """Feature extraction must be purely lexical."""

    def forbidden(*args, **kwargs):  # pragma: no cover - only runs on failure
        raise AssertionError("feature extraction attempted a network call")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    features = extract_url_features("https://suspicious-looking.example.tk/login")
    assert features["has_https"] == 1
