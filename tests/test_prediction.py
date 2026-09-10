from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


def test_health_and_coverage() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["disease_count"] >= 25
    assert response.json()["validation_accuracy"] >= 0.80


def test_prediction_and_explanation() -> None:
    response = client.post("/predict", json={"symptoms": ["fever", "chills", "body_ache", "fatigue", "cough"]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["disease"]
    assert payload["shap_explanation"]
    assert payload["disclaimer"]
