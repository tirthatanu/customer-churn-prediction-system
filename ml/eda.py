"""Reproducible exploratory data analysis and exported figures.

From project root:
    python -m ml.eda
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ml.common import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    ROOT,
    TARGET,
    TEST_CSV,
    TRAIN_CSV,
    load_dataset,
)


def main() -> None:
    train, training_profile = load_dataset(TRAIN_CSV)
    test, test_profile = load_dataset(TEST_CSV)
    output = ROOT / "reports" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", palette="deep")

    summary = {
        "train": training_profile,
        "test": test_profile,
        "train_dimensions": list(train.shape),
        "test_dimensions": list(test.shape),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "training_numeric_summary": train[NUMERIC_FEATURES].describe().round(3).to_dict(),
        "train_categorical_counts": {
            col: {str(k): int(v) for k, v in train[col].value_counts(dropna=False).items()}
            for col in CATEGORICAL_FEATURES
        },
    }
    (ROOT / "reports" / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 5))
    counts = train[TARGET].value_counts().reindex([0, 1])
    sns.barplot(x=["Retained (0)", "Churned (1)"], y=counts.values, ax=ax, color="#4f62ba")
    ax.set(title="Training label distribution", ylabel="Number of customers", xlabel="")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():,.0f}", (p.get_x() + p.get_width()/2, p.get_height()),
                    ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output / "class_distribution.png", dpi=160)
    plt.close(fig)

    numeric_for_hist = ["Age", "Tenure", "Usage Frequency", "Support Calls", "Payment Delay", "Total Spend"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    # Sample for quick rendering; exact counts and metrics always use full data.
    sampled = train.sample(n=min(45_000, len(train)), random_state=42)
    for ax, col in zip(axes.flat, numeric_for_hist):
        sns.histplot(sampled[col], bins=24, ax=ax, color="#4f62ba", edgecolor=None)
        ax.set(title=col, ylabel="Sample count")
    fig.suptitle("Training feature distributions (sampled for plotting)")
    fig.tight_layout()
    fig.savefig(output / "numeric_distributions.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, col in zip(axes, CATEGORICAL_FEATURES):
        rates = train.groupby(col)[TARGET].agg(["mean", "size"]).sort_values("mean", ascending=False)
        sns.barplot(x=rates.index, y=rates["mean"].values, ax=ax, color="#4f62ba")
        ax.set(xlabel=col, ylabel="Churn rate", ylim=(0, 1))
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Churn rate by category (training data)")
    fig.tight_layout()
    fig.savefig(output / "churn_by_category.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    correlations = train[NUMERIC_FEATURES + [TARGET]].corr()
    sns.heatmap(correlations, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Numeric correlations, training data")
    fig.tight_layout()
    fig.savefig(output / "correlation_heatmap.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    rates = [training_profile["churn_rate"], test_profile["churn_rate"]]
    sns.barplot(x=["Training", "Testing"], y=rates, ax=ax, color="#4f62ba")
    ax.set(ylabel="Churn fraction", ylim=(0, 1), title="Label prevalence across provided splits")
    for i, rate in enumerate(rates):
        ax.annotate(f"{rate:.1%}", (i, rate + 0.02), ha="center")
    fig.tight_layout()
    fig.savefig(output / "train_test_target_shift.png", dpi=160)
    plt.close(fig)

    print("Training rows:", len(train), "Test rows:", len(test))
    print("Training churn rate:", f"{training_profile['churn_rate']:.2%}")
    print("Testing churn rate:", f"{test_profile['churn_rate']:.2%}")
    print("Figures and EDA summary saved under reports/")


if __name__ == "__main__":
    main()
