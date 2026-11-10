"""Shared dataset column definitions and loading utilities."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data" / "raw" / "train.csv"
TEST_CSV = ROOT / "data" / "raw" / "test.csv"
MODEL_FILE = ROOT / "models" / "churn_pipeline.joblib"
MODEL_METADATA = ROOT / "models" / "model_metadata.json"

TARGET = "Churn"
IDENTIFIER = "CustomerID"
NUMERIC_FEATURES = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
]
CATEGORICAL_FEATURES = ["Gender", "Subscription Type", "Contract Length"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

API_TO_DATASET = {
    "age": "Age",
    "gender": "Gender",
    "tenure": "Tenure",
    "usage_frequency": "Usage Frequency",
    "support_calls": "Support Calls",
    "payment_delay": "Payment Delay",
    "subscription_type": "Subscription Type",
    "contract_length": "Contract Length",
    "total_spend": "Total Spend",
    "last_interaction": "Last Interaction",
}


def load_dataset(path: Path) -> tuple[pd.DataFrame, dict]:
    """Load CSV, remove unlabeled records, and clean feature types."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_csv(path)
    expected = set(FEATURES + [TARGET])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset {path.name} missing columns: {sorted(missing)}")

    original_rows = len(df)
    # The supplied training CSV contains one completely blank row.
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise").astype("int64")
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Churn must contain only the labels 0 and 1")
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].str.strip()
    profile = {
        "file": path.name,
        "original_rows": original_rows,
        "usable_rows": int(len(df)),
        "rows_removed_missing_target": original_rows - len(df),
        "missing_feature_values": {c: int(df[c].isna().sum()) for c in FEATURES},
        "churn_counts": {str(int(k)): int(v) for k, v in df[TARGET].value_counts().items()},
        "churn_rate": float(df[TARGET].mean()),
    }
    return df, profile
