"""Train and evaluate customer churn models with no test-set tuning.

From project root:
    python -m ml.train
"""
import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.common import (
    CATEGORICAL_FEATURES,
    FEATURES,
    MODEL_FILE,
    MODEL_METADATA,
    NUMERIC_FEATURES,
    ROOT,
    TARGET,
    TEST_CSV,
    TRAIN_CSV,
    load_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)
SEED = 42


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def make_candidates() -> dict[str, Pipeline]:
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=120,
            max_leaf_nodes=25,
            learning_rate=0.08,
            l2_regularization=1.0,
            random_state=SEED,
        ),
    }
    return {
        name: Pipeline([("preprocess", make_preprocessor()), ("classifier", estimator)])
        for name, estimator in estimators.items()
    }


def score_model(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Scoring uses P(Churn=1), not a fixed class column assumption."""
    labels = list(model.classes_)
    pos_col = labels.index(1)
    positive_prob = model.predict_proba(x)[:, pos_col]
    y_pred = (positive_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, positive_prob)),
        "brier_score": float(brier_score_loss(y, positive_prob)),
    }


def main() -> None:
    reports = ROOT / "reports"
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Reading training dataset and untouched external test file")
    train_df, train_profile = load_dataset(TRAIN_CSV)
    test_df, test_profile = load_dataset(TEST_CSV)
    x_all, y_all = train_df[FEATURES], train_df[TARGET]
    x_test, y_test = test_df[FEATURES], test_df[TARGET]
    x_train, x_val, y_train, y_val = train_test_split(
        x_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all
    )

    candidates = make_candidates()
    comparison = []
    for name, model in candidates.items():
        LOGGER.info("Fitting %s using %s rows", name, len(x_train))
        model.fit(x_train, y_train)
        scores = score_model(model, x_val, y_val)
        comparison.append({"model": name, **scores})
        LOGGER.info("Validation %s: ROC-AUC %.4f, F1 %.4f", name, scores["roc_auc"], scores["f1"])

    # All model selection is based on validation data, never on test labels.
    best = max(comparison, key=lambda item: (item["roc_auc"], item["f1"]))
    best_name = best["model"]
    LOGGER.info("Selected %s using validation ROC-AUC", best_name)
    final_model = copy.deepcopy(candidates[best_name])
    final_model.fit(x_all, y_all)
    test_scores = score_model(final_model, x_test, y_test)
    probabilities = final_model.predict_proba(x_test)[:, list(final_model.classes_).index(1)]
    predictions = (probabilities >= 0.5).astype(int)
    LOGGER.info("Test: ROC-AUC %.4f, F1 %.4f", test_scores["roc_auc"], test_scores["f1"])

    joblib.dump(final_model, MODEL_FILE, compress=3)
    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_model": best_name,
        "selection_metric": "validation_roc_auc",
        "validation_fraction": 0.2,
        "random_state": SEED,
        "decision_threshold": 0.5,
        "risk_thresholds": {"low_below": 0.30, "high_at_or_above": 0.70},
        "feature_names": FEATURES,
        "positive_class": 1,
        "training": train_profile,
        "testing": test_profile,
        "validation_results": comparison,
        "selected_validation_result": best,
        "test_metrics": test_scores,
        "notes": [
            "CustomerID excluded because it is an identifier.",
            "Test set was never used to tune or choose a model.",
            "Predicted probabilities are model estimates, not guaranteed calibrated probabilities.",
            "Training and supplied test datasets have differing churn prevalence; monitor dataset shift.",
            "Provided dataset has no stated provenance or representative-population guarantee.",
        ],
    }
    MODEL_METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (reports / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame(comparison).round(6).to_csv(reports / "model_comparison.csv", index=False)
    (reports / "classification_report.txt").write_text(
        classification_report(y_test, predictions, digits=4, target_names=["Retained", "Churned"]),
        encoding="utf-8",
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, predictions, labels=[0, 1]),
        display_labels=["Retained", "Churned"],
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Test-set confusion matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(figures / "confusion_matrix.png", dpi=155)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {test_scores['roc_auc']:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random guessing")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve - held-out test data")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figures / "roc_curve.png", dpi=155)
    plt.close(fig)

    # Reliability curve on held-out test set; use as diagnostic, NOT as a fitted calibrator.
    from sklearn.calibration import CalibrationDisplay
    fig, ax = plt.subplots(figsize=(6.5, 5))
    CalibrationDisplay.from_predictions(
        y_test, probabilities, n_bins=10, strategy="quantile", ax=ax, name=best_name
    )
    ax.set_title("Probability calibration diagnostic - test set")
    fig.tight_layout()
    fig.savefig(figures / "calibration_diagnostic.png", dpi=155)
    plt.close(fig)

    # Explain the selected fitted model with permutation importance on a
    # sample of validation records (never use the external test for selection).
    val_indices, _ = train_test_split(
        np.arange(len(x_val)), train_size=2_500, random_state=SEED, stratify=y_val
    )
    importance = permutation_importance(
        candidates[best_name], x_val.iloc[val_indices], y_val.iloc[val_indices],
        scoring="roc_auc", n_repeats=3, random_state=SEED, n_jobs=1,
    )
    importance_table = pd.DataFrame({
        "feature": FEATURES,
        "importance_mean_roc_auc_drop": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean_roc_auc_drop", ascending=False)
    importance_table.to_csv(reports / "feature_importance.csv", index=False)
    metadata["feature_importance"] = importance_table.round(6).to_dict(orient="records")
    MODEL_METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (reports / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    top = importance_table.sort_values("importance_mean_roc_auc_drop", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5.1))
    ax.barh(top["feature"], top["importance_mean_roc_auc_drop"],
            xerr=top["importance_std"], color="#6b61d9", capsize=2)
    ax.axvline(0, color="#a9acc2", linewidth=1)
    ax.set(xlabel="Mean decrease in validation ROC-AUC", title="Permutation importance of original input features")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "feature_importance.png", dpi=165)
    plt.close(fig)

    LOGGER.info("Saved pipeline to %s", MODEL_FILE)
    LOGGER.info("Saved evaluation results to %s", reports)


if __name__ == "__main__":
    main()
