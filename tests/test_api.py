from fastapi.testclient import TestClient
import joblib
from pathlib import Path
from src.app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert "status" in j

def test_predict_response(monkeypatch):
    class Dummy:
        def predict(self, X):
            return [0 for _ in X]
    monkeypatch.setattr("src.app.main._model", Dummy())
    r = client.post("/predict", json={"inputs": [[1,2,3]]})
    assert r.status_code == 200
    assert r.json()["predictions"] == [0]
