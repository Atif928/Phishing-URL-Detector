"""URL feature engineering.

This module converts a raw URL *string* into a fixed-order dictionary of
numeric features. It never performs network requests: parsing is done entirely
with :mod:`urllib.parse`.

The same function is used during training and at prediction time, which is what
prevents "train/serve skew".

Security notes
--------------
* Input is treated as untrusted text. It is length-limited and validated
  before parsing.
* ``javascript:``, ``data:``, ``file:`` and other non-HTTP schemes are
  rejected.
* The URL is never opened, fetched or executed.
"""

from __future__ import annotations

import ipaddress
import re
from functools import lru_cache
from typing import Iterable
from urllib.parse import urlsplit

import pandas as pd

from config.settings import MAX_URL_LENGTH
from src.utils import safe_divide

# --------------------------------------------------------------------------- #
# Feature schema -- the order here is the canonical model input order.
# Changing it invalidates saved models; the metadata file stores this list and
# prediction refuses to run if it does not match.
# --------------------------------------------------------------------------- #
FEATURE_NAMES: tuple[str, ...] = (
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_question_marks",
    "num_equal_signs",
    "num_at_symbols",
    "num_ampersands",
    "num_special_chars",
    "num_digits",
    "digit_ratio",
    "num_letters",
    "letter_ratio",
    "num_subdomains",
    "num_params",
    "path_depth",
    "tld_length",
    "has_https",
    "has_ip_address",
    "has_suspicious_keyword",
    "num_suspicious_keywords",
    "has_url_shortener",
    "has_brand_term",
    "hostname_is_long",
    "has_unusual_port",
    "has_port",
    "has_punycode",
    "suspicious_tld",
    "num_percent_encoded",
    "has_hex_encoding",
    "has_double_slash_in_path",
    "has_obfuscation",
    "is_long_url",
    "hostname_hyphen_count",
)

# --------------------------------------------------------------------------- #
# Lookup tables
# --------------------------------------------------------------------------- #
SUSPICIOUS_KEYWORDS = frozenset(
    {
        "account",
        "alert",
        "authenticate",
        "banking",
        "bonus",
        "confirm",
        "credential",
        "free",
        "gift",
        "login",
        "logon",
        "password",
        "pay",
        "prize",
        "recover",
        "recovery",
        "redirect",
        "secure",
        "security",
        "signin",
        "submit",
        "support",
        "suspend",
        "unlock",
        "update",
        "validate",
        "verify",
        "wallet",
        "webscr",
    }
)

URL_SHORTENER_DOMAINS = frozenset(
    {
        "adf.ly",
        "bit.do",
        "bit.ly",
        "buff.ly",
        "cutt.ly",
        "goo.gl",
        "is.gd",
        "lnkd.in",
        "ow.ly",
        "rb.gy",
        "rebrand.ly",
        "s.id",
        "shorturl.at",
        "t.co",
        "t.ly",
        "tiny.cc",
        "tinyurl.com",
        "v.gd",
        "rebrandly.com",
    }
)

# brand term -> official registrable domain (used to avoid flagging the real site)
BRAND_TERMS: dict[str, str] = {
    "adobe": "adobe.com",
    "amazon": "amazon.com",
    "apple": "apple.com",
    "bankofamerica": "bankofamerica.com",
    "binance": "binance.com",
    "chase": "chase.com",
    "coinbase": "coinbase.com",
    "dhl": "dhl.com",
    "dropbox": "dropbox.com",
    "facebook": "facebook.com",
    "fedex": "fedex.com",
    "google": "google.com",
    "icloud": "icloud.com",
    "instagram": "instagram.com",
    "linkedin": "linkedin.com",
    "microsoft": "microsoft.com",
    "netflix": "netflix.com",
    "office365": "office.com",
    "outlook": "outlook.com",
    "paypal": "paypal.com",
    "steam": "steampowered.com",
    "usps": "usps.com",
    "whatsapp": "whatsapp.com",
    "wellsfargo": "wellsfargo.com",
}

# TLDs that appear disproportionately often in phishing campaigns.
SUSPICIOUS_TLDS = frozenset(
    {
        "accountant",
        "bid",
        "buzz",
        "cam",
        "cfd",
        "click",
        "club",
        "country",
        "cyou",
        "date",
        "download",
        "faith",
        "fit",
        "ga",
        "gdn",
        "gq",
        "icu",
        "link",
        "live",
        "loan",
        "men",
        "ml",
        "monster",
        "online",
        "party",
        "quest",
        "racing",
        "rest",
        "review",
        "sbs",
        "science",
        "site",
        "stream",
        "surl",
        "tk",
        "top",
        "trade",
        "win",
        "work",
        "xyz",
    }
)

# Characters counted as "special" (letters, digits, dots, slashes and hyphens
# are intentionally excluded because they have dedicated feature columns).
SPECIAL_CHARS = frozenset("!$%^&*()_+|~=`{}[]:;'<>,?=@#\\\"")

IPV4_PATTERN = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
HEX_ESCAPE_PATTERN = re.compile(r"%[0-9a-fA-F]{2}")
HOSTNAME_PATTERN = re.compile(r"^[\w.\-\[\]:]+$", re.UNICODE)
DIGIT_PATTERN = re.compile(r"\d")
LETTER_PATTERN = re.compile(r"[A-Za-z]")

DEFAULT_PORTS = {None, 80, 443}


# --------------------------------------------------------------------------- #
# Parsing and validation
# --------------------------------------------------------------------------- #
def normalize_url(url: str) -> str:
    """Clean and validate a URL.

    A missing scheme is assumed to be ``http://`` (a common, conservative
    assumption for pasted URLs). Raises ``ValueError`` for anything that is not
    a plausible HTTP(S) URL.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string.")
    cleaned = url.strip()
    if not cleaned:
        raise ValueError("URL is empty.")
    if len(cleaned) > MAX_URL_LENGTH:
        raise ValueError(f"URL is longer than the {MAX_URL_LENGTH} character limit.")
    if any(char.isspace() for char in cleaned):
        raise ValueError("URL must not contain whitespace.")
    if any(ord(char) < 32 or ord(char) == 127 for char in cleaned):
        raise ValueError("URL must not contain control characters.")

    lowered = cleaned.lower()
    if lowered.startswith(("javascript:", "data:", "file:", "vbscript:", "about:", "chrome:")):
        raise ValueError("Only http:// and https:// URLs are supported.")

    if "://" not in cleaned:
        cleaned = "http://" + cleaned

    try:
        parsed = urlsplit(cleaned)
        hostname = parsed.hostname
        # Accessing .port validates that a port (if present) is numeric/valid.
        parsed.port
    except ValueError as exc:
        raise ValueError(f"URL could not be parsed: {exc}") from exc

    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only http:// and https:// URLs are supported.")
    if not parsed.netloc or not hostname:
        raise ValueError("URL has no hostname.")
    if not HOSTNAME_PATTERN.match(hostname):
        raise ValueError("URL hostname contains invalid characters.")
    if hostname.startswith(".") or hostname.endswith("."):
        raise ValueError("URL hostname is malformed.")

    if not _is_ip(hostname):
        _validate_hostname(hostname)

    return cleaned


def _validate_hostname(hostname: str) -> None:
    """Validate DNS-style hostname labels (approximate, no public suffix list)."""
    labels = hostname.split(".")
    if len(labels) < 2:
        raise ValueError("URL hostname is not a valid domain name.")
    for label in labels:
        if not label:
            raise ValueError("URL hostname contains an empty label.")
        if label.startswith("-") or label.endswith("-"):
            raise ValueError("URL hostname label starts or ends with a hyphen.")
        if len(label) > 63:
            raise ValueError("URL hostname label is longer than 63 characters.")

    tld = labels[-1]
    if len(tld) < 2 or not re.fullmatch(r"[a-z0-9\-\u0080-\uffff]+", tld):
        raise ValueError("URL has an invalid top-level domain.")


def _is_ip(hostname: str) -> bool:
    """True when the hostname is an IPv4/IPv6 address."""
    candidate = hostname.strip("[]")
    try:
        ipaddress.ip_address(candidate)
        return True
    except ValueError:
        return bool(IPV4_PATTERN.match(candidate))


def is_valid_url(url: str) -> bool:
    """Non-raising wrapper around :func:`normalize_url`."""
    try:
        normalize_url(url)
        return True
    except (ValueError, TypeError):
        return False


@lru_cache(maxsize=2048)
def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(part for part in re.split(r"[^a-z0-9]+", text.lower()) if part)


def _contains_keyword(text: str, keywords: Iterable[str]) -> bool:
    """Match suspicious keywords as whole URL tokens.

    URLs are split on non-alphanumeric characters, so ``verify-account`` and
    ``/login/`` match while ``accountant`` does not. Concatenated words
    (``paypalverify``) are intentionally not matched by this feature; other
    signals such as brand terms and URL length still apply.
    """
    keyword_set = keywords if isinstance(keywords, (set, frozenset)) else frozenset(keywords)
    return any(token in keyword_set for token in _tokenize(text))


def _count_keywords(text: str) -> int:
    tokens = set(_tokenize(text))
    return len(tokens & SUSPICIOUS_KEYWORDS)


def _extract_tld(hostname: str) -> str:
    if _is_ip(hostname) or "." not in hostname:
        return ""
    return hostname.rsplit(".", 1)[-1].lower()


def _count_subdomains(hostname: str) -> int:
    if _is_ip(hostname) or "." not in hostname:
        return 0
    labels = hostname.split(".")
    # The last two labels are the registrable domain (approximation that does
    # not depend on the Public Suffix List).
    return max(len(labels) - 2, 0)


def parse_url(url: str):
    """Return the parsed URL object after validation."""
    normalized = normalize_url(url)
    return normalized, urlsplit(normalized)


# --------------------------------------------------------------------------- #
# Main public API
# --------------------------------------------------------------------------- #
def extract_url_features(url: str) -> dict[str, float | int]:
    """Extract all engineered features from a single URL.

    Parameters
    ----------
    url:
        Raw URL text supplied by the user or dataset.

    Returns
    -------
    dict
        Feature name -> numeric value, in :data:`FEATURE_NAMES` order.

    Raises
    ------
    ValueError
        If the URL is empty, malformed or uses a non-HTTP(S) scheme.
    """
    normalized, parsed = parse_url(url)
    hostname = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    scheme = parsed.scheme.lower()

    url_length = len(normalized)
    hostname_length = len(hostname)
    num_digits = len(DIGIT_PATTERN.findall(normalized))
    num_letters = len(LETTER_PATTERN.findall(normalized))

    tld = _extract_tld(hostname)
    is_ip = _is_ip(hostname)

    # Query parameters: "a=1&b=2" -> 2
    num_params = len([pair for pair in query.split("&") if pair]) if query else 0

    # Path depth: number of non-empty path segments.
    path_depth = len([segment for segment in path.split("/") if segment])

    has_percent = "%" in normalized
    percent_count = normalized.count("%")
    has_hex_encoding = int(percent_count >= 3 and bool(HEX_ESCAPE_PATTERN.search(normalized)))

    double_slash_path = int("//" in path)

    features: dict[str, float | int] = {
        "url_length": url_length,
        "hostname_length": hostname_length,
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": normalized.count("."),
        "num_hyphens": normalized.count("-"),
        "num_underscores": normalized.count("_"),
        "num_slashes": normalized.count("/"),
        "num_question_marks": normalized.count("?"),
        "num_equal_signs": normalized.count("="),
        "num_at_symbols": normalized.count("@"),
        "num_ampersands": normalized.count("&"),
        "num_special_chars": sum(1 for char in normalized if char in SPECIAL_CHARS),
        "num_digits": num_digits,
        "digit_ratio": round(safe_divide(num_digits, url_length), 4),
        "num_letters": num_letters,
        "letter_ratio": round(safe_divide(num_letters, url_length), 4),
        "num_subdomains": _count_subdomains(hostname),
        "num_params": num_params,
        "path_depth": path_depth,
        "tld_length": len(tld),
        "has_https": int(scheme == "https"),
        "has_ip_address": int(is_ip),
        "has_suspicious_keyword": int(_contains_keyword(normalized, SUSPICIOUS_KEYWORDS)),
        "num_suspicious_keywords": _count_keywords(normalized),
        "has_url_shortener": int(_is_shortener(hostname)),
        "has_brand_term": int(_has_brand_impostor(hostname, normalized)),
        "hostname_is_long": int(hostname_length > 30),
        "has_unusual_port": int(_has_unusual_port(parsed)),
        "has_port": int(parsed.port is not None),
        "has_punycode": int("xn--" in hostname),
        "suspicious_tld": int(bool(tld) and tld in SUSPICIOUS_TLDS),
        "num_percent_encoded": percent_count if has_percent else 0,
        "has_hex_encoding": has_hex_encoding,
        "has_double_slash_in_path": double_slash_path,
        "has_obfuscation": int(
            "@" in normalized or has_hex_encoding or bool(double_slash_path)
        ),
        "is_long_url": int(url_length > 75),
        "hostname_hyphen_count": hostname.count("-"),
    }

    # Defensive check: guarantees training and prediction use the same schema.
    if tuple(features.keys()) != FEATURE_NAMES:
        raise RuntimeError("Feature schema mismatch in extract_url_features().")
    return features


def _is_shortener(hostname: str) -> bool:
    return any(hostname == domain or hostname.endswith("." + domain) for domain in URL_SHORTENER_DOMAINS)


def _has_brand_impostor(hostname: str, url: str) -> bool:
    """Flag brand names used on a domain that is not the brand's own domain."""
    for brand, official in BRAND_TERMS.items():
        pattern = rf"(^|[.\-_/]){re.escape(brand)}([.\-_/]|$)"
        if not re.search(pattern, url.lower()):
            continue
        if hostname == official or hostname.endswith("." + official):
            continue
        return True
    return False


def _has_unusual_port(parsed) -> bool:
    try:
        port = parsed.port
    except ValueError:
        return True
    return port not in DEFAULT_PORTS


def extract_features_batch(urls: Iterable[str]) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Extract features for many URLs.

    Invalid URLs are skipped and returned as ``(url, error_message)`` pairs so
    callers can report how many rows were dropped and why.
    """
    rows: list[dict[str, float | int]] = []
    invalid: list[tuple[str, str]] = []
    for url in urls:
        try:
            rows.append(extract_url_features(url))
        except (ValueError, TypeError) as exc:
            invalid.append((str(url), str(exc)))
    frame = pd.DataFrame(rows, columns=list(FEATURE_NAMES))
    return frame, invalid


def get_feature_descriptions() -> dict[str, str]:
    """Human-readable descriptions shown in the Streamlit UI."""
    return {
        "url_length": "Total number of characters in the URL",
        "hostname_length": "Length of the hostname (domain + subdomains)",
        "path_length": "Length of the URL path",
        "query_length": "Length of the query string",
        "num_dots": "Number of '.' characters",
        "num_hyphens": "Number of '-' characters",
        "num_underscores": "Number of '_' characters",
        "num_slashes": "Number of '/' characters",
        "num_question_marks": "Number of '?' characters",
        "num_equal_signs": "Number of '=' characters",
        "num_at_symbols": "Number of '@' characters",
        "num_ampersands": "Number of '&' characters",
        "num_special_chars": "Count of unusual special characters",
        "num_digits": "Number of digit characters",
        "digit_ratio": "Digits divided by URL length",
        "num_letters": "Number of Latin letters",
        "letter_ratio": "Letters divided by URL length",
        "num_subdomains": "Number of subdomain labels",
        "num_params": "Number of query parameters",
        "path_depth": "Number of path segments",
        "tld_length": "Length of the top-level domain",
        "has_https": "1 if the scheme is HTTPS",
        "has_ip_address": "1 if an IP address is used instead of a domain",
        "has_suspicious_keyword": "1 if a phishing-style keyword is present",
        "num_suspicious_keywords": "Number of distinct phishing-style keywords",
        "has_url_shortener": "1 if the host is a known URL shortener",
        "has_brand_term": "1 if a brand name appears on an unofficial domain",
        "hostname_is_long": "1 if the hostname is longer than 30 characters",
        "has_unusual_port": "1 if a non-standard network port is used",
        "has_port": "1 if an explicit port is present",
        "has_punycode": "1 if the domain uses punycode ('xn--')",
        "suspicious_tld": "1 if the TLD is frequently abused",
        "num_percent_encoded": "Number of percent-encoded characters",
        "has_hex_encoding": "1 if the URL contains several hex escapes",
        "has_double_slash_in_path": "1 if '//' appears in the path",
        "has_obfuscation": "1 if '@', hex escapes or '//' obfuscation is present",
        "is_long_url": "1 if the URL is longer than 75 characters",
        "hostname_hyphen_count": "Number of hyphens in the hostname",
    }
