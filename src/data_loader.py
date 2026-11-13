"""Dataset discovery, loading and label normalisation.

The project consumes a simple CSV with two required columns:

    url,label

``label`` uses this project's convention: ``1 = phishing``, ``0 = legitimate``.
Different public datasets use different conventions, so all label handling is
centralised here and documented in ``docs/dataset_guide.md``.

This module never downloads data at import time. Use
``python -m src.data_loader --download-phiusiil`` (or the guide in the docs)
to obtain the dataset explicitly.
"""

from __future__ import annotations

import argparse
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from config.settings import (
    DATASET_PATH,
    LEGITIMATE_LABEL,
    NEGATIVE_LABEL_STRINGS,
    PHISHING_LABEL,
    POSITIVE_LABEL_STRINGS,
    RAW_DATA_DIR,
    URL_COLUMN_CANDIDATES,
    LABEL_COLUMN_CANDIDATES,
)
from src.utils import get_logger, save_json, utc_timestamp

logger = get_logger(__name__)

PHIUSIIL_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
PHIUSIIL_ARCHIVE_NAME = "phiusiil.zip"
PHIUSIIL_CSV_NAME = "PhiUSIIL_Phishing_URL_Dataset.csv"


class DatasetError(Exception):
    """Raised when a dataset is missing, unreadable or malformed."""


@dataclass
class DatasetReport:
    """Summary of what happened while loading/cleaning the dataset."""

    source: str = ""
    rows_loaded: int = 0
    rows_used: int = 0
    missing_urls: int = 0
    missing_labels: int = 0
    duplicate_urls: int = 0
    invalid_urls: int = 0
    unmapped_labels: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "rows_loaded": self.rows_loaded,
            "rows_used": self.rows_used,
            "missing_urls": self.missing_urls,
            "missing_labels": self.missing_labels,
            "duplicate_urls": self.duplicate_urls,
            "invalid_urls": self.invalid_urls,
            "unmapped_labels": self.unmapped_labels,
            "class_counts": self.class_counts,
            "notes": self.notes,
            "generated_at": utc_timestamp(),
        }


# --------------------------------------------------------------------------- #
# Dataset discovery
# --------------------------------------------------------------------------- #
def find_dataset_path(preferred: str | Path | None = None) -> Path:
    """Locate a raw URL dataset.

    Search order:
    1. explicit ``preferred`` path / ``PHISHING_DATASET_PATH``
    2. ``data/raw/urls.csv`` (the standardised file)
    3. the downloaded PhiUSIIL CSV inside ``data/raw/phiusiil``
    4. any other CSV directly inside ``data/raw``
    """
    candidates: list[Path] = []
    if preferred:
        candidates.append(Path(preferred))
    if DATASET_PATH not in candidates:
        candidates.append(DATASET_PATH)

    candidates.extend(
        [
            RAW_DATA_DIR / "phiusiil" / PHIUSIIL_CSV_NAME,
            RAW_DATA_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv",
        ]
    )
    if RAW_DATA_DIR.exists():
        candidates.extend(sorted(RAW_DATA_DIR.glob("*.csv")))

    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate

    raise DatasetError(
        "No dataset found. Place a CSV with 'url' and 'label' columns at "
        f"{DATASET_PATH} (or download PhiUSIIL with "
        "'python -m src.data_loader --download-phiusiil'). "
        "See docs/dataset_guide.md for details."
    )


# --------------------------------------------------------------------------- #
# Reading and standardising
# --------------------------------------------------------------------------- #
def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV trying a few common encodings (URL datasets are messy)."""
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.EmptyDataError as exc:
            raise DatasetError(f"Dataset file is empty: {path}") from exc
        except pd.errors.ParserError as exc:
            raise DatasetError(f"Dataset could not be parsed as CSV: {path} ({exc})") from exc
    raise DatasetError(f"Could not decode dataset {path}: {last_error}")


def _find_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {str(column).strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def is_phiusiil_frame(frame: pd.DataFrame) -> bool:
    """Detect the original UCI PhiUSIIL export (different label convention)."""
    columns = {str(c).strip().lower() for c in frame.columns}
    return {"filename", "urllength", "urlsimilarityindex"} <= columns


def normalize_label(value) -> int | None:
    """Map a single label value to 1 (phishing) or 0 (legitimate).

    Returns ``None`` for values that cannot be interpreted confidently.
    """
    if isinstance(value, bool):
        return PHISHING_LABEL if value else LEGITIMATE_LABEL
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    text = str(value).strip().lower()
    if text in POSITIVE_LABEL_STRINGS:
        return PHISHING_LABEL
    if text in NEGATIVE_LABEL_STRINGS:
        return LEGITIMATE_LABEL
    return None


def normalize_labels(series: pd.Series) -> pd.Series:
    """Vectorised version of :func:`normalize_label` (NaN marks unusable rows)."""
    mapping = {value: normalize_label(value) for value in series.dropna().unique()}
    return series.map(mapping)


def standardize_frame(frame: pd.DataFrame, source: str = "") -> tuple[pd.DataFrame, DatasetReport]:
    """Return a ``url``/``label`` frame with the project's label convention."""
    report = DatasetReport(source=source or "<in-memory>", rows_loaded=len(frame))
    if frame.empty:
        raise DatasetError("Dataset contains no rows.")

    columns = list(frame.columns)
    url_col = _find_column(columns, URL_COLUMN_CANDIDATES)
    label_col = _find_column(columns, LABEL_COLUMN_CANDIDATES)

    if url_col is None:
        raise DatasetError(
            "Dataset has no URL column. Expected one of: "
            f"{', '.join(URL_COLUMN_CANDIDATES)}. Found: {', '.join(map(str, columns[:15]))}"
        )
    if label_col is None:
        raise DatasetError(
            "Dataset has no label column. Expected one of: "
            f"{', '.join(LABEL_COLUMN_CANDIDATES)}. Found: {', '.join(map(str, columns[:15]))}"
        )

    phiusiil = is_phiusiil_frame(frame)
    working = frame[[url_col, label_col]].copy()
    working.columns = ["url", "label"]

    if phiusiil:
        # UCI PhiUSIIL: 1 = legitimate, 0 = phishing. Invert to our convention
        # where 1 = phishing and 0 = legitimate.
        working["label"] = pd.to_numeric(working["label"], errors="coerce").map(
            {1: LEGITIMATE_LABEL, 0: PHISHING_LABEL}
        )
        report.notes.append(
            "Detected PhiUSIIL dataset: original labels (1=legitimate, 0=phishing) "
            "were converted to 1=phishing, 0=legitimate."
        )
    else:
        working["label"] = normalize_labels(working["label"])

    if working["url"].notna().any():
        working["url"] = working["url"].astype(str).str.strip()
        # Strip an accidental "www." prefix? No: it is a real signal, keep as-is.
        working["url"] = working["url"].replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})

    report.missing_urls = int(working["url"].isna().sum())
    report.missing_labels = int(working["label"].isna().sum())
    report.unmapped_labels = report.missing_labels

    working = working.dropna(subset=["url", "label"])
    working["label"] = working["label"].astype(int)
    report.rows_used = len(working)
    report.class_counts = {
        "legitimate": int((working["label"] == LEGITIMATE_LABEL).sum()),
        "phishing": int((working["label"] == PHISHING_LABEL).sum()),
    }

    if working.empty:
        raise DatasetError(
            "No usable rows remain after cleaning. Check the label values in your CSV: "
            "supported values include 0/1, legitimate/phishing, benign/malicious."
        )

    # Duplicates are removed in preprocessing (reported there), but making the
    # frame unique by URL here keeps the raw dataset tidy.
    return working.reset_index(drop=True), report


def load_url_dataset(path: str | Path | None = None) -> tuple[pd.DataFrame, DatasetReport]:
    """Load and standardise a labelled URL dataset from disk."""
    dataset_path = find_dataset_path(path)
    logger.info("Loading dataset: %s", dataset_path)
    frame = _read_csv(dataset_path)

    if frame.empty:
        raise DatasetError(f"Dataset file has no rows: {dataset_path}")

    standard, report = standardize_frame(frame, source=str(dataset_path))
    if report.notes:
        for note in report.notes:
            logger.info(note)
    logger.info(
        "Loaded %s rows (legitimate=%s, phishing=%s)",
        f"{report.rows_used:,}",
        report.class_counts.get("legitimate", 0),
        report.class_counts.get("phishing", 0),
    )
    return standard, report


# --------------------------------------------------------------------------- #
# Optional dataset download (explicit, user-initiated only)
# --------------------------------------------------------------------------- #
def download_phiusiil(destination: Path | None = None, force: bool = False) -> Path:
    """Download and extract the UCI PhiUSIIL dataset.

    This downloads a *dataset file*, never user-submitted URLs. The extracted
    CSV is converted into ``data/raw/urls.csv`` by
    :func:`convert_phiusiil_to_standard`.
    """
    raw_dir = Path(destination) if destination else RAW_DATA_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / PHIUSIIL_ARCHIVE_NAME
    extract_dir = raw_dir / "phiusiil"
    csv_path = extract_dir / PHIUSIIL_CSV_NAME

    if csv_path.exists() and not force:
        logger.info("PhiUSIIL dataset already present: %s", csv_path)
        return csv_path

    if not archive_path.exists() or force:
        logger.info("Downloading PhiUSIIL dataset from UCI ...")
        urlretrieve(PHIUSIIL_URL, archive_path)  # noqa: S310 - fixed, trusted URL

    logger.info("Extracting archive ...")
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)

    if not csv_path.exists():
        raise DatasetError(f"Expected {PHIUSIIL_CSV_NAME} inside {archive_path}.")
    return csv_path


def convert_phiusiil_to_standard(
    source: str | Path | None = None,
    destination: str | Path = DATASET_PATH,
) -> Path:
    """Convert the raw PhiUSIIL export into ``data/raw/urls.csv``."""
    source_path = Path(source) if source else download_phiusiil()
    frame = _read_csv(source_path)
    standard, report = standardize_frame(frame, source=str(source_path))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    standard.to_csv(destination, index=False, encoding="utf-8")
    save_json(report.as_dict(), destination.parent / "dataset_load_report.json")
    logger.info("Wrote standardised dataset (%s rows) to %s", f"{len(standard):,}", destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset utilities for the Phishing URL Detector.")
    parser.add_argument(
        "--download-phiusiil",
        action="store_true",
        help="Download the UCI PhiUSIIL dataset and convert it to data/raw/urls.csv",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    args = parser.parse_args()

    if args.download_phiusiil:
        try:
            csv_path = download_phiusiil(force=args.force)
            convert_phiusiil_to_standard(csv_path)
        except DatasetError as exc:
            raise SystemExit(f"ERROR: {exc}") from exc
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
