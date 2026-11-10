import pytest

from app.main import risk_level


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    info = response.json()
    assert "test_metrics" in info
    assert len(info["features"]) == 10
    assert info["target"] == "Churn"


def test_predict_and_history(client, sample_customer):
    assert client.get("/predictions").json()["count"] == 0
    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 200, response.text
    result = response.json()
    assert set(result) == {"prediction", "probability", "risk_level"}
    assert result["prediction"] in {"Churn", "No Churn"}
    assert 0 <= result["probability"] <= 1
    assert result["risk_level"] in {"Low", "Medium", "High"}
    history = client.get("/predictions", params={"limit": 5}).json()
    assert history["count"] == 1
    assert history["predictions"][0]["prediction"] == result["prediction"]
    assert "age" not in history["predictions"][0]
    assert "gender" not in history["predictions"][0]


def test_validation_error(client, sample_customer):
    data = dict(sample_customer, age=-1, gender="Unknown")
    response = client.post("/predict", json=data)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert client.get("/predictions").json()["count"] == 0


def test_rejects_unavailable_features(client, sample_customer):
    data = dict(sample_customer, monthly_charges=75.5)
    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_incomplete_payload(client):
    response = client.post("/predict", json={"tenure": 12})
    assert response.status_code == 422


def test_history_pagination(client, sample_customer):
    for _ in range(3):
        assert client.post("/predict", json=sample_customer).status_code == 200
    page = client.get("/predictions?limit=2&offset=1").json()
    assert page["count"] == 3
    assert page["limit"] == 2
    assert page["offset"] == 1
    assert len(page["predictions"]) == 2
    assert client.get("/predictions?limit=101").status_code == 422


@pytest.mark.parametrize("probability, expected", [(0.0, "Low"), (0.299, "Low"),
                                                        (0.30, "Medium"), (0.699, "Medium"),
                                                        (0.70, "High"), (1.0, "High")])
def test_risk_thresholds(probability, expected):
    assert risk_level(probability) == expected


def test_frontend_served(client):
    result = client.get("/")
    assert result.status_code == 200
    assert "<title>Customer Churn Prediction System</title>" in result.text
    assert "Prediction System</em>" in result.text


def test_internal_errors_do_not_leak_details(client, sample_customer):
    class BrokenModel:
        classes_ = [0, 1]
        def predict_proba(self, _):
            raise RuntimeError("PRIVATE_INTERNAL_SECRET")
    client.app.state.pipeline = BrokenModel()
    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    assert "PRIVATE_INTERNAL_SECRET" not in response.text
