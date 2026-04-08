#!/usr/bin/env bash
set -e
echo "Creating productionization files..."

# README.md
cat > README.md <<'EOF2'
# ML Model Deployment Pipeline (End-to-End)

## Overview
This repository demonstrates an end-to-end machine learning workflow: training, evaluation, serialization, and production-ready serving using Docker and CI.

## Architecture
- Training: Python scripts (scikit-learn / PyTorch)
- Model serialization: joblib (sklearn) or TorchScript / ONNX (CV)
- API Layer: FastAPI exposing `/predict` and `/health`
- Containerization: Docker
- CI/CD: GitHub Actions (lint, tests, build)
- Model versioning: simple filename conventions or MLflow

## Quickstart (local)
1. Create virtual env
   python -m venv .venv && source .venv/bin/activate
2. Install
   pip install -r requirements.txt
3. Train & save a model (example)
   python src/train.py --out models/model_v1.joblib
4. Run API
   uvicorn src.app.main:app --reload
5. Predict
   POST /predict with JSON {"inputs":[...]} or use the included example client.

## Endpoints
- GET /health — returns service health and model metadata
- POST /predict — returns model predictions for given inputs

## MLOps Perspective
This repo demonstrates:
- Reproducible pipelines (train -> save -> serve)
- Automated CI for tests & lint
- Containerized serving for portability
- Evaluation metrics output for model validation

## Next steps / improvements
- Add monitoring (Prometheus + Grafana)
- Add model registry (MLflow)
- Add CV-specific conversion and model optimization (TorchScript/ONNX)
EOF2

# Create directories
mkdir -p src/app tests .github/workflows models

# src/app/main.py
cat > src/app/main.py <<'EOF2'
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any, Dict
import joblib
import os
import logging

app = FastAPI(title="ML Model Serving")

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model_v1.joblib")

logger = logging.getLogger("uvicorn.error")

class PredictRequest(BaseModel):
    inputs: List[Any]

class PredictResponse(BaseModel):
    predictions: List[Any]
    metadata: Dict[str, Any] = {}

_model = None

def load_model(path: str):
    global _model
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    _model = joblib.load(path)
    return _model

@app.on_event("startup")
def startup_event():
    try:
        model = load_model(MODEL_PATH)
        logger.info(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        logger.exception("Failed to load model at startup")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "model_path": MODEL_PATH}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    preds = _model.predict(req.inputs).tolist()
    return PredictResponse(predictions=preds, metadata={"model_path": MODEL_PATH})
EOF2

# src/train.py
cat > src/train.py <<'EOF2'
import argparse
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score, classification_report
import json
from pathlib import Path

def main(out_path: str):
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_path)
    metrics = {"accuracy": acc, "report": report}
    with open(Path(out_path).with_suffix('.metrics.json'), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved model to {out_path} with accuracy {acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", dest="out_path", default="models/model_v1.joblib")
    args = parser.parse_args()
    main(args.out_path)
EOF2

# Dockerfile
cat > Dockerfile <<'EOF2'
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/model_v1.joblib

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8080
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
EOF2

# GitHub Actions workflow
cat > .github/workflows/ci.yml <<'EOF2'
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest
      - name: Run tests
        run: pytest -q
  docker-build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: |
          docker build -t ml-model-service:latest .
EOF2

# tests/test_api.py
cat > tests/test_api.py <<'EOF2'
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
EOF2

# requirements.txt
cat > requirements.txt <<'EOF2'
fastapi
uvicorn[standard]
scikit-learn
joblib
prometheus_client
pytest
EOF2

# pre-commit config
cat > .pre-commit-config.yaml <<'EOF2'
repos:
  - repo: https://github.com/psf/black
    rev: 24.1.0
    hooks:
      - id: black
  - repo: https://github.com/PyCQA/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://gitlab.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
EOF2

# .gitignore
cat > .gitignore <<'EOF2'
__pycache__/
.venv/
models/
.env
*.pyc
.pytest_cache/
EOF2

# package __init__ files
cat > src/__init__.py <<'EOF2'
# top-level package
EOF2

cat > src/app/__init__.py <<'EOF2'
# package init
EOF2

echo "All files created."

# Git operations
git add .
git commit -m "chore: productionize repo — add FastAPI service, training & eval, Dockerfile, CI, tests, and formatting config" || { echo "Nothing to commit."; exit 0; }
echo "About to push to origin/main. Make sure you want to modify main directly. Continue? (y/N)"
read -r confirm
if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
  git push origin main
  echo "Pushed to origin/main."
else
  echo "Aborted. To push later, run: git push origin main"
fi
