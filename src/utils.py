"""Small shared helpers used across the project."""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER_NAME = "phishing_detector"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a configured module logger (safe to call repeatedly)."""
    logger = logging.getLogger(name or LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)-8s %(name)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def utc_timestamp() -> str:
    """ISO-8601 UTC timestamp, used in metadata files."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalars and nested containers to JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if hasattr(value, "item"):  # numpy scalar
        return to_jsonable(value.item())
    if hasattr(value, "tolist"):  # numpy array / pandas series
        return to_jsonable(value.tolist())
    return str(value)


def save_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    """Write ``data`` as UTF-8 JSON, creating parent folders as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=indent, ensure_ascii=False)
    return path


def load_json(path: str | Path) -> Any:
    """Read a JSON file. Raises FileNotFoundError / ValueError on problems."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as exc:  # pragma: no cover - corrupt artefact
            raise ValueError(f"Could not parse JSON file {path}: {exc}") from exc


def slugify(value: str) -> str:
    """Turn a model name into a filesystem-friendly slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "model"


def format_number(value: float | int, decimals: int = 0) -> str:
    """Format large counts with thousands separators."""
    if decimals:
        return f"{value:,.{decimals}f}"
    return f"{int(value):,}"


def percent(value: float, decimals: int = 2) -> str:
    """Format a 0-1 ratio as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that never raises on a zero denominator."""
    if denominator in (0, 0.0, None):
        return default
    return numerator / denominator
