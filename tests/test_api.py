import pytest
from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)

def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "version" in response.json()

def test_health():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_prediction():
    """Test prediction endpoint."""
    response = client.post(
        "/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]}
    )
    
    # If model not loaded, should return 503
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "processing_time" in data

def test_prediction_invalid_input():
    """Test prediction with invalid input."""
    response = client.post(
        "/predict",
        json={"features": [5.1, 3.5]}  # Too few features
    )
    assert response.status_code == 422  # Validation error

def test_batch_prediction():
    """Test batch prediction endpoint."""
    batch = {
        "requests": [
            {"features": [5.1, 3.5, 1.4, 0.2]},
            {"features": [6.2, 2.9, 4.3, 1.3]}
        ]
    }
    response = client.post("/predict/batch", json=batch)
    assert response.status_code in [200, 503]

def test_model_info():
    """Test model info endpoint."""
    response = client.get("/model/info")
    assert response.status_code in [200, 503]
