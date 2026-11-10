from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from ml.common import MODEL_FILE, MODEL_METADATA


@pytest.fixture
def client(tmp_path: Path):
    test_app = create_app(model_path=MODEL_FILE, metadata_path=MODEL_METADATA,
                          database_path=tmp_path / "predictions.sqlite3")
    with TestClient(test_app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def sample_customer():
    return {
        "age": 30,
        "gender": "Female",
        "tenure": 12,
        "usage_frequency": 15,
        "support_calls": 4,
        "payment_delay": 10,
        "subscription_type": "Standard",
        "contract_length": "Monthly",
        "total_spend": 906.0,
        "last_interaction": 9,
    }
