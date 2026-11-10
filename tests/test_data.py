import pandas as pd

from ml.common import FEATURES, TRAIN_CSV, load_dataset
from ml.train import make_preprocessor


def test_clean_dataset():
    data, profile = load_dataset(TRAIN_CSV)
    assert len(data) == 440_832
    assert profile["rows_removed_missing_target"] == 1
    assert set(data["Churn"].unique()) == {0, 1}
    assert set(FEATURES).issubset(data.columns)


def test_preprocessor_handles_missing_and_unseen():
    a = {"Age": 30, "Tenure": 12, "Usage Frequency": 4, "Support Calls": 3,
         "Payment Delay": 5, "Total Spend": 700, "Last Interaction": 15,
         "Gender": "Female", "Subscription Type": "Basic", "Contract Length": "Monthly"}
    b = dict(a, Age=None, Gender="Male", Subscription_Type="unused")
    b.pop("Subscription_Type")
    preprocessor = make_preprocessor()
    out = preprocessor.fit_transform(pd.DataFrame([a, b]))
    assert out.shape[0] == 2
    unseen = dict(a, Subscription_Type="unused")
    unseen.pop("Subscription_Type")
    unseen["Subscription Type"] = "NewPlan"
    transformed = preprocessor.transform(pd.DataFrame([unseen]))
    assert transformed.shape[1] == out.shape[1]
