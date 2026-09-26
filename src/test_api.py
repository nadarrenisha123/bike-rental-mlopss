"""
Basic smoke tests for the FastAPI service.
"""

import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_response():
    payload = {
        "season": 1,
        "yr": 1,
        "mnth": 7,
        "holiday": 0,
        "weekday": 2,
        "workingday": 1,
        "weathersit": 1,
        "temp": 20.5,
        "hum": 60.0,
        "windspeed": 10.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    body = response.json()

    assert "predicted_count" in body
    assert isinstance(body["predicted_count"], float)