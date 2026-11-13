"""Train, compare and persist the phishing URL classifiers.

Three scikit-learn models are trained on identical data:

* Logistic Regression (with StandardScaler inside a Pipeline)
* Decision Tree
* Random Forest

The best model is selected primarily by F1-score, then recall, then ROC-AUC.
Recall is emphasised because failing to flag a phishing URL is usually more
costly than a false alarm, while precision still matters because false alarms
erode trust. All metrics come from the held-out test set.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import time
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config.settings import (
    MODELS_DIR,
    MODEL_COMPARISON_CSV,
    MODEL_COMPARISON_JSON,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PROCESSED_DATASET_PATH,
    PROJECT_ROOT,
    RANDOM_STATE,
    TEST_SIZE,
    TRAINING_SUMMARY_JSON,
    ensure_directories,
)
from src.data_loader import DatasetError
from src.feature_extraction import FEATURE_NAMES
from src.model_evaluation import compute_metrics, format_metrics_table
from src.preprocessing import (
    class_distribution,
    load_processed_dataset,
    prepare_dataset,
    split_dataset,
)
from src.utils import get_logger, save_json, slugify, utc_timestamp

logger = get_logger(__name__)


def build_models() -> dict:
    """Return fresh, untrained estimators with reproducible settings."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=14,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def feature_schema_hash(feature_names: list[str] | tuple[str, ...] = FEATURE_NAMES) -> str:
    """Stable fingerprint of the feature order (used to detect train/serve drift)."""
    return hashlib.sha256("|".join(feature_names).encode("utf-8")).hexdigest()[:16]


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def train_models(
    dataset_path: str | Path | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    force_rebuild: bool = False,
    save: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Run the full training workflow and return ``(comparison, metadata)``."""
    ensure_directories()

    if force_rebuild or not PROCESSED_DATASET_PATH.exists():
        # prepare_dataset reuses the cached feature CSV unless it is missing.
        prepare_dataset(dataset_path=dataset_path, force=force_rebuild)

    frame = load_processed_dataset()
    logger.info("Loaded %s engineered rows with %s features.", f"{len(frame):,}", len(FEATURE_NAMES))

    X_train, X_test, y_train, y_test = split_dataset(
        frame, test_size=test_size, random_state=random_state
    )
    logger.info(
        "Train rows: %s | Test rows: %s | test_size=%.2f | random_state=%s",
        f"{len(X_train):,}",
        f"{len(X_test):,}",
        test_size,
        random_state,
    )
    logger.info("Training class distribution: %s", class_distribution(y_train))

    models = build_models()
    results: list[dict] = []
    trained_models: dict[str, object] = {}

    for name, estimator in models.items():
        logger.info("Training %s ...", name)
        start = time.perf_counter()
        estimator.fit(X_train, y_train)
        train_time = time.perf_counter() - start

        start = time.perf_counter()
        y_pred = estimator.predict(X_test)
        predict_time = time.perf_counter() - start

        y_prob = (
            estimator.predict_proba(X_test)[:, 1] if hasattr(estimator, "predict_proba") else None
        )
        metrics = compute_metrics(y_test, y_pred, y_prob, include_report=True)
        train_accuracy = float(estimator.score(X_train, y_train))

        model_file = MODELS_DIR / f"{slugify(name)}.joblib"
        if save:
            joblib.dump(estimator, model_file, compress=3)

        trained_models[name] = estimator
        results.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"],
                "train_accuracy": round(train_accuracy, 6),
                "train_time_seconds": round(train_time, 3),
                "predict_time_seconds": round(predict_time, 3),
                "true_negatives": metrics["true_negatives"],
                "false_positives": metrics["false_positives"],
                "false_negatives": metrics["false_negatives"],
                "true_positives": metrics["true_positives"],
                "confusion_matrix": metrics["confusion_matrix"],
                "file": _relative(model_file),
            }
        )
        logger.info(
            "%s -> accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f (%.2fs)",
            name,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"],
            metrics["roc_auc"] or 0.0,
            train_time,
        )

    comparison = pd.DataFrame(results).sort_values(
        ["f1_score", "recall", "roc_auc"], ascending=False
    ).reset_index(drop=True)
    comparison.insert(0, "rank", range(1, len(comparison) + 1))

    best_row = comparison.iloc[0]
    best_name = str(best_row["model"])
    best_model = trained_models[best_name]

    if save:
        joblib.dump(best_model, MODEL_PATH, compress=3)
        comparison.to_csv(MODEL_COMPARISON_CSV, index=False)
        save_json(comparison.to_dict(orient="records"), MODEL_COMPARISON_JSON)

    selection_rule = (
        "Ranked by F1-score (primary), then recall (false negatives are costly in "
        "security), then ROC-AUC. Accuracy alone is not used."
    )

    metadata = {
        "model_name": best_name,
        "model_file": _relative(MODEL_PATH),
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "feature_schema_hash": feature_schema_hash(),
        "label_mapping": {"0": "Legitimate", "1": "Phishing"},
        "positive_class": "Phishing",
        "trained_at": utc_timestamp(),
        "random_state": random_state,
        "test_size": test_size,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_class_distribution": class_distribution(y_train),
        "test_class_distribution": class_distribution(y_test),
        "metrics": {
            key: best_row[key]
            for key in (
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "true_negatives",
                "false_positives",
                "false_negatives",
                "true_positives",
            )
        },
        "confusion_matrix": best_row["confusion_matrix"],
        "selection_rule": selection_rule,
        "comparison": comparison.to_dict(orient="records"),
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
        "dataset": {
            "path": "data/raw/urls.csv",
            "source": "PhiUSIIL Phishing URL Dataset (UCI ML Repository, ID 967)",
        },
    }

    summary = {
        "trained_at": metadata["trained_at"],
        "rows": int(len(frame)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": len(FEATURE_NAMES),
        "selection_rule": selection_rule,
        "selected_model": {
            "name": best_name,
            "file": _relative(MODEL_PATH),
            "metrics": metadata["metrics"],
        },
        "models": results,
        "comparison": comparison.to_dict(orient="records"),
    }

    if save:
        save_json(metadata, MODEL_METADATA_PATH)
        save_json(summary, TRAINING_SUMMARY_JSON)

    print("\n" + format_metrics_table(comparison) + "\n")
    logger.info("Selected model: %s (saved to %s)", best_name, MODEL_PATH)

    return comparison, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and compare phishing URL classification models."
    )
    parser.add_argument("--dataset", type=str, default=None, help="Optional raw url,label CSV")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the processed feature matrix before training",
    )
    args = parser.parse_args()

    try:
        train_models(
            dataset_path=args.dataset,
            test_size=args.test_size,
            random_state=args.random_state,
            force_rebuild=args.force,
        )
    except (DatasetError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
