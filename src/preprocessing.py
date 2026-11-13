"""Data cleaning, feature extraction and train/test splitting.

The pipeline is:

    raw CSV -> validate columns -> clean URLs -> extract features
            -> save data/processed/url_features.csv
            -> stratified train/test split (80/20, random_state=42)

No scaling, feature selection or resampling is applied to the full dataset.
Those steps live inside the scikit-learn pipelines so they are fitted on the
training split only (no data leakage).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config.settings import (
    DATASET_PATH,
    DATASET_SUMMARY_JSON,
    LEGITIMATE_LABEL,
    PHISHING_LABEL,
    PROCESSED_DATASET_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data_loader import DatasetError, load_url_dataset
from src.feature_extraction import FEATURE_NAMES, extract_url_features, is_valid_url
from src.utils import get_logger, save_json, utc_timestamp

logger = get_logger(__name__)


def clean_url_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop missing/duplicate/invalid URLs and return cleaning statistics."""
    stats: dict[str, object] = {
        "rows_input": int(len(frame)),
        "missing_urls": 0,
        "duplicate_urls": 0,
        "invalid_urls": 0,
        "invalid_examples": [],
    }

    working = frame.copy()
    working["url"] = working["url"].astype("string").str.strip()
    working["label"] = pd.to_numeric(working["label"], errors="coerce")

    missing_mask = working["url"].isna() | (working["url"] == "") | working["label"].isna()
    stats["missing_urls"] = int(missing_mask.sum())
    working = working.loc[~missing_mask].copy()

    # Duplicate detection is case-insensitive because hostnames are
    # case-insensitive ("Example.com" == "example.com").
    working["_url_key"] = working["url"].str.lower()
    duplicate_mask = working["_url_key"].duplicated(keep="first")
    stats["duplicate_urls"] = int(duplicate_mask.sum())
    working = working.loc[~duplicate_mask].drop(columns="_url_key")

    validity = working["url"].map(is_valid_url)
    invalid_rows = working.loc[~validity, "url"].head(5).tolist()
    stats["invalid_urls"] = int((~validity).sum())
    stats["invalid_examples"] = [str(url) for url in invalid_rows]
    working = working.loc[validity].copy()

    working["label"] = working["label"].astype(int)
    stats["rows_output"] = int(len(working))
    return working.reset_index(drop=True), stats


def build_feature_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Extract engineered features and attach the original URL and label.

    Row-by-row extraction keeps URLs aligned with their features even if an
    individual URL fails parsing.
    """
    rows: list[dict] = []
    kept_positions: list[int] = []
    invalid: list[tuple[str, str]] = []

    total = len(frame)
    for position, (url, _label) in enumerate(
        zip(frame["url"].tolist(), frame["label"].tolist())
    ):
        try:
            rows.append(extract_url_features(url))
            kept_positions.append(position)
        except (ValueError, TypeError) as exc:
            invalid.append((str(url), str(exc)))
        if total >= 50_000 and (position + 1) % 50_000 == 0:
            logger.info("  ... %s / %s URLs processed", f"{position + 1:,}", f"{total:,}")

    if invalid:
        logger.warning("Skipped %s URLs that failed feature extraction.", len(invalid))

    features = pd.DataFrame(rows, columns=list(FEATURE_NAMES))
    realised = frame.iloc[kept_positions].reset_index(drop=True)
    output = pd.concat(
        [realised[["url", "label"]].reset_index(drop=True), features.reset_index(drop=True)],
        axis=1,
    )
    return output, invalid


def prepare_dataset(
    dataset_path: str | Path | None = None,
    output_path: str | Path | None = None,
    force: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run the full cleaning + feature-extraction step and cache the result.

    Parameters
    ----------
    dataset_path:
        Optional path to a raw CSV. Defaults to the configured dataset.
    output_path:
        Where to write the processed feature CSV.
    force:
        Rebuild even if the processed CSV already exists.
    limit:
        Optional row limit for quick experiments (applied before cleaning).
    """
    output_path = Path(output_path) if output_path else PROCESSED_DATASET_PATH
    if output_path.exists() and not force:
        logger.info("Processed dataset already exists: %s (use force=True to rebuild)", output_path)
        return load_processed_dataset(output_path)

    frame, load_report = load_url_dataset(dataset_path)
    if limit:
        frame = frame.head(limit).copy()
        logger.info("Limiting dataset to the first %s rows.", f"{limit:,}")

    cleaned, clean_stats = clean_url_frame(frame)
    if cleaned.empty:
        raise DatasetError("All rows were removed during cleaning; the dataset is unusable.")

    logger.info("Extracting URL features for %s URLs ...", f"{len(cleaned):,}")
    featured, failed = build_feature_frame(cleaned)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(output_path, index=False, encoding="utf-8")

    class_counts = {
        "legitimate": int((featured["label"] == LEGITIMATE_LABEL).sum()),
        "phishing": int((featured["label"] == PHISHING_LABEL).sum()),
    }
    total = len(featured)
    summary = {
        "generated_at": utc_timestamp(),
        "raw_dataset": load_report.as_dict(),
        "cleaning": clean_stats,
        "feature_extraction_failures": len(failed),
        "rows_final": total,
        "class_counts": class_counts,
        "class_balance": {
            label: round(count / total, 4) if total else 0.0
            for label, count in class_counts.items()
        },
        "imbalance_ratio": (
            round(max(class_counts.values()) / min(class_counts.values()), 3)
            if min(class_counts.values()) > 0
            else None
        ),
        "num_features": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "processed_path": str(output_path),
    }
    save_json(summary, DATASET_SUMMARY_JSON)
    logger.info(
        "Saved %s engineered rows to %s (legitimate=%s, phishing=%s)",
        f"{total:,}",
        output_path,
        f"{class_counts['legitimate']:,}",
        f"{class_counts['phishing']:,}",
    )
    return featured


def load_processed_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the cached engineered feature matrix and validate its schema."""
    path = Path(path) if path else PROCESSED_DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. Run 'python -m src.preprocessing' first."
        )

    frame = pd.read_csv(path)
    missing = [column for column in ("url", "label", *FEATURE_NAMES) if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Processed dataset is missing {len(missing)} expected column(s): "
            f"{', '.join(missing[:8])}. Rebuild it with 'python -m src.preprocessing --force'."
        )
    if frame.empty:
        raise ValueError("Processed dataset is empty.")
    return frame


def split_dataset(
    frame: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Stratified 80/20 split returning ``X_train, X_test, y_train, y_test``."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if frame.empty:
        raise ValueError("Cannot split an empty dataset.")
    if frame["label"].nunique() < 2:
        raise ValueError("Dataset must contain both phishing and legitimate examples.")

    features = frame.loc[:, list(FEATURE_NAMES)]
    target = frame["label"].astype(int)
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def class_distribution(target: pd.Series) -> dict[str, float | int]:
    """Return counts and proportions for a label series."""
    counts = target.value_counts().to_dict()
    total = len(target)
    return {
        "total": total,
        "legitimate": int(counts.get(LEGITIMATE_LABEL, 0)),
        "phishing": int(counts.get(PHISHING_LABEL, 0)),
        "phishing_ratio": round(counts.get(PHISHING_LABEL, 0) / total, 4) if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean URLs and build the feature matrix.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to a raw url,label CSV")
    parser.add_argument("--output", type=str, default=None, help="Output feature CSV path")
    parser.add_argument("--force", action="store_true", help="Rebuild even if a cache exists")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N rows")
    args = parser.parse_args()

    try:
        prepare_dataset(
            dataset_path=args.dataset,
            output_path=args.output,
            force=args.force,
            limit=args.limit,
        )
    except (DatasetError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
