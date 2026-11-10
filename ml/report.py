"""Generate a readable evaluation report from the exported model and dataset."""
import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

from ml.common import FEATURES, MODEL_FILE, MODEL_METADATA, NUMERIC_FEATURES, ROOT, TEST_CSV, TRAIN_CSV, TARGET, load_dataset


def main() -> None:
    meta = json.loads(MODEL_METADATA.read_text(encoding="utf-8"))
    model = joblib.load(MODEL_FILE)
    train, _ = load_dataset(TRAIN_CSV)
    test, _ = load_dataset(TEST_CSV)
    prediction = model.predict(test[FEATURES])
    tn, fp, fn, tp = confusion_matrix(test[TARGET], prediction, labels=[0, 1]).ravel()
    test_metrics = meta["test_metrics"]
    comparison = pd.DataFrame(meta["validation_results"])
    percent = lambda value: f"{value:.2%}"
    comparison_table = "| Candidate | Validation ROC-AUC | Validation F1 | Validation accuracy |\n|---|---:|---:|---:|\n"
    for _, row in comparison.iterrows():
        comparison_table += f"| {row['model']} | {row['roc_auc']:.4f} | {row['f1']:.4f} | {percent(row['accuracy'])} |\n"
    means_table = "| Input feature | Training mean | Test mean |\n|---|---:|---:|\n"
    for col in NUMERIC_FEATURES:
        means_table += f"| {col} | {train[col].mean():.2f} | {test[col].mean():.2f} |\n"
    baseline = max(float(test[TARGET].mean()), 1 - float(test[TARGET].mean()))
    top_feats = meta["feature_importance"][:5]
    top_table = "| Feature | Validation ROC-AUC decrease |\n|---|---:|\n"
    for item in top_feats:
        top_table += f"| {item['feature']} | {item['importance_mean_roc_auc_drop']:.4f} |\n"
    content = f"""# Customer Churn Prediction System — Model Evaluation Report

Generated: {datetime.now(timezone.utc).date().isoformat()} (UTC)

## 1. Dataset and task

Predict **Churn** (0 = retained, 1 = churned) from 10 customer demographic, usage, and subscription attributes. User-provided CSVs originate from a publicly shared Customer Churn Dataset (see README for Kaggle link). CustomerID is excluded. Missing target records are dropped; one blank training row was removed. Missing feature values, if present, are imputed using statistics fitted on training folds only.

- Clean training records: **{len(train):,}**. Supplied test records: **{len(test):,}**.
- Training churn prevalence: **{percent(train[TARGET].mean())}**. Test churn prevalence: **{percent(test[TARGET].mean())}**.
- The supplied test set was **not used in model selection or parameter tuning**.

## 2. Preprocessing and experimental design

Numerical features: median imputation followed by standard scaling. Categorical features: most-frequent imputation followed by one-hot encoding (`handle_unknown='ignore'`). These transforms and the classifier are stored in the same Scikit-learn `Pipeline`. A stratified 80/20 split of the **provided training CSV** was used to compare models. Validation ROC-AUC, with F1 as a tie-breaker, selected the classifier. The winner was retrained on the full supplied training file, then evaluated once on the separate provided test file using a fixed 0.50 decision threshold. Random seed: 42.

## 3. Validation model comparison

{comparison_table}
Selected: **{meta['best_model']}** by validation ROC-AUC.

## 4. Held-out test performance (selected model)

| Metric | Test result |
|---|---:|
| Accuracy | **{percent(test_metrics['accuracy'])}** |
| Precision (churn class) | **{percent(test_metrics['precision'])}** |
| Recall (churn class) | **{percent(test_metrics['recall'])}** |
| F1-score (churn class) | **{percent(test_metrics['f1'])}** |
| ROC-AUC | **{test_metrics['roc_auc']:.4f}** |
| Brier score (lower better) | **{test_metrics['brier_score']:.4f}** |
| Always-predict-majority baseline accuracy | **{percent(baseline)}** |

Confusion matrix (actual rows × predicted columns):

| Actual / Predicted | Retained (0) | Churned (1) |
|---|---:|---:|
| Retained (0) | {tn:,} | {fp:,} |
| Churned (1) | {fn:,} | {tp:,} |

![Confusion matrix](figures/confusion_matrix.png)

![ROC curve](figures/roc_curve.png)

![Probability calibration diagnostic](figures/calibration_diagnostic.png)

## 5. Distribution shift warning

**Critical:** The selected model has approximately **{percent(meta['selected_validation_result']['accuracy'])}** validation accuracy but only **{percent(test_metrics['accuracy'])}** on the supplied separate test data. Its test accuracy is below the simple majority-class baseline of **{percent(baseline)}**. This large difference indicates that validation results do not carry over to the provided test distribution. The model labels nearly all supplied test records as churned. This is not acceptable evidence for live retention decisions.

The data themselves have different distributions:

{means_table}
Differences in feature distributions, churn prevalence, and/or label-generating relationships could contribute. The original generating process and real-world representativeness have not been verified; no specific root cause can be proven from these CSVs alone.

## 6. Feature importance

Permutation importance was computed on a **2,500-record sample of the validation partition**, using 3 repeats and the decrease in ROC-AUC. Results describe model dependence on input features, **not causality**. The validation-split context matters because external generalization is poor.

{top_table}
![Feature importance](figures/feature_importance.png)

## 7. Risk categories and limits

The API returns `P(Churn=1)` from `predict_proba` and a class using the **0.50** threshold. Risk level is a separate rule: **Low <0.30; Medium 0.30 to <0.70; High >=0.70**. These boundaries are illustrative configuration choices, not business-optimized cutoffs. Because external test calibration is poor, displaying a very high probability **does not guarantee churn**.

## 8. Recommended next steps

Before any practical retention decision, investigate the provenance and time period of both files; check for duplicate entities and label inconsistencies; collect a representative deployment sample; use a time- or group-aware holdout if relevant; evaluate more stable training data and external validation; calibrate and monitor probabilities; define thresholds using a real intervention-cost framework; and restrict access to prediction history.

**Intended use:** academic demonstration of an end-to-end FastAPI + Scikit-learn workflow, not a production-ready decision engine.
"""
    (ROOT / "reports" / "model_evaluation.md").write_text(content, encoding="utf-8")
    print("Wrote reports/model_evaluation.md")


if __name__ == "__main__":
    main()
