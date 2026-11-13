"""Model evaluation: metrics, reports and charts.

Every number produced here comes from predictions on the held-out test split
created by :func:`src.preprocessing.split_dataset`. Nothing is hard-coded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless-safe backend for scripts / servers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config.settings import PROJECT_ROOT, RESULTS_DIR
from src.feature_extraction import FEATURE_NAMES
from src.preprocessing import class_distribution, load_processed_dataset, split_dataset
from src.utils import get_logger, load_json, save_json, slugify, to_jsonable, utc_timestamp

logger = get_logger(__name__)
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

MODEL_COLORS = {
    "Logistic Regression": "#2E86AB",
    "Decision Tree": "#F18F01",
    "Random Forest": "#2A9D8F",
}
DEFAULT_COLOR = "#6C5B7B"


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def compute_metrics(
    y_true,
    y_pred,
    y_prob=None,
    *,
    include_report: bool = False,
) -> dict:
    """Compute the evaluation metrics used throughout the project.

    All values are computed from real predictions; nothing is hard-coded.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "roc_auc": None,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": cm.tolist(),
        "support": {
            "legitimate": int((y_true == 0).sum()),
            "phishing": int((y_true == 1).sum()),
        },
    }
    if y_prob is not None:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 6)
    if include_report:
        metrics["classification_report"] = classification_report(
            y_true,
            y_pred,
            target_names=["Legitimate", "Phishing"],
            output_dict=True,
            zero_division=0,
        )
    return to_jsonable(metrics)


def format_metrics_table(comparison: pd.DataFrame) -> str:
    """Markdown table for the console / README."""
    display = comparison.copy()
    for column in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        if column in display:
            display[column] = display[column].map(lambda value: f"{value:.4f}")
    if "train_time_seconds" in display:
        display["train_time_seconds"] = display["train_time_seconds"].map(lambda v: f"{v:.2f}s")
    columns = [
        c
        for c in ("model", "accuracy", "precision", "recall", "f1_score", "roc_auc", "train_time_seconds")
        if c in display.columns
    ]
    return display[columns].to_markdown(index=False)


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def _color_for(name: str) -> str:
    return MODEL_COLORS.get(name, DEFAULT_COLOR)


def plot_metric_bars(comparison: pd.DataFrame, results_dir: Path) -> list[Path]:
    """Save a grouped bar chart plus one chart per headline metric."""
    paths: list[Path] = []
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = [m for m in ("accuracy", "precision", "recall", "f1_score", "roc_auc") if m in comparison]
    melted = comparison.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="score")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=melted, x="metric", y="score", hue="model", ax=ax, palette="deep")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison on the held-out test set")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.legend(title="Model", loc="lower right", fontsize=9)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, padding=1)
    path = results_dir / "metrics_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    titles = {
        "precision": "Precision comparison",
        "recall": "Recall comparison",
        "f1_score": "F1-score comparison",
        "accuracy": "Accuracy comparison",
    }
    for metric, title in titles.items():
        if metric not in comparison:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        sns.barplot(
            data=comparison,
            x="model",
            y=metric,
            hue="model",
            legend=False,
            palette=[_color_for(m) for m in comparison["model"]],
            ax=ax,
        )
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{title} (test set)")
        ax.set_xlabel("")
        ax.set_ylabel(metric.replace("_", " ").title())
        for container in ax.containers:
            ax.bar_label(container, fmt="%.4f", fontsize=9)
        fig.savefig(results_dir / f"{metric}_comparison.png")
        plt.close(fig)
        paths.append(results_dir / f"{metric}_comparison.png")

    return paths


def plot_roc_curves(y_true, probabilities: dict[str, np.ndarray], results_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, proba in probabilities.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=_color_for(name), linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="Random guessing")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves - phishing class")
    ax.legend(loc="lower right", fontsize=9)
    path = results_dir / "roc_curves.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_confusion_matrix(cm, model_name: str, results_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"],
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(f"Confusion matrix - {model_name}")
    path = results_dir / f"confusion_matrix_{slugify(model_name)}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_feature_importance(model, feature_names, results_dir: Path, top_n: int = 15) -> Path | None:
    """Bar chart of the most important features (tree models only)."""
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = list(model.named_steps.values())[-1]
    if not hasattr(estimator, "feature_importances_"):
        return None

    importances = pd.Series(estimator.feature_importances_, index=feature_names).sort_values(
        ascending=False
    )
    top = importances.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    top.plot(kind="barh", ax=ax, color="#2A9D8F")
    ax.set_title(f"Top {top_n} features - Random Forest")
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("")
    path = results_dir / "feature_importance.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Full evaluation
# --------------------------------------------------------------------------- #
def evaluate_saved_models(results_dir: Path | None = None) -> pd.DataFrame:
    """Load every trained model, evaluate on the shared test split, save artefacts."""
    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(results_dir / "training_summary.json")
    models_info = summary["models"]
    if not models_info:
        raise ValueError("training_summary.json lists no models. Train models first.")

    frame = load_processed_dataset()
    _, X_test, _, y_test = split_dataset(frame)

    rows = []
    probabilities: dict[str, np.ndarray] = {}

    for entry in models_info:
        name = entry.get("name") or entry.get("model")
        if not name:
            logger.warning("Skipping malformed model entry: %s", entry)
            continue
        model_path = PROJECT_ROOT / entry["file"]
        if not model_path.exists():
            logger.warning("Skipping missing model file: %s", model_path)
            continue

        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        metrics = compute_metrics(y_test, y_pred, y_prob, include_report=True)

        rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"],
                "train_time_seconds": entry.get("train_time_seconds"),
                "predict_time_seconds": entry.get("predict_time_seconds"),
            }
        )
        if y_prob is not None:
            probabilities[name] = y_prob

        plot_confusion_matrix(np.array(metrics["confusion_matrix"]), name, results_dir)
        report_text = classification_report(
            y_test, y_pred, target_names=["Legitimate", "Phishing"], digits=4, zero_division=0
        )
        (results_dir / f"classification_report_{slugify(name)}.txt").write_text(
            f"Model: {name}\n"
            f"Evaluated on the held-out test set: {len(y_test):,} URLs\n"
            f"{'=' * 62}\n{report_text}",
            encoding="utf-8",
        )
        (results_dir / f"metrics_{slugify(name)}.json").write_text(
            __import__("json").dumps(metrics, indent=2), encoding="utf-8"
        )

    if not rows:
        raise RuntimeError("No saved models could be evaluated.")

    comparison = pd.DataFrame(rows).sort_values(
        ["f1_score", "recall", "roc_auc"], ascending=False
    ).reset_index(drop=True)
    comparison.insert(0, "rank", range(1, len(comparison) + 1))
    comparison.to_csv(results_dir / "model_comparison.csv", index=False)

    plot_metric_bars(comparison, results_dir)
    if probabilities:
        plot_roc_curves(np.asarray(y_test), probabilities, results_dir)

    best_name = summary.get("selected_model", {}).get("name")
    if best_name:
        best_entry = next(
            (m for m in models_info if (m.get("name") or m.get("model")) == best_name), None
        )
        if best_entry:
            best_path = results_dir.parent / best_entry["file"]
            if best_path.exists():
                plot_feature_importance(joblib.load(best_path), list(FEATURE_NAMES), results_dir)

    evaluation = {
        "generated_at": utc_timestamp(),
        "test_rows": int(len(y_test)),
        "test_class_distribution": class_distribution(y_test),
        "selected_model": best_name,
        "selection_rule": summary.get("selection_rule"),
        "models": rows,
        "comparison": comparison.to_dict(orient="records"),
        "detailed_metrics": {
            row["model"]: load_json(results_dir / f"metrics_{slugify(row['model'])}.json")
            for row in rows
        },
    }
    save_json(evaluation, results_dir / "evaluation_metrics.json")

    logger.info("Evaluation written to %s", results_dir)
    print("\n" + format_metrics_table(comparison) + "\n")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved phishing-detection models.")
    parser.add_argument("--results-dir", type=str, default=None)
    args = parser.parse_args()
    evaluate_saved_models(Path(args.results_dir) if args.results_dir else None)


if __name__ == "__main__":
    main()
