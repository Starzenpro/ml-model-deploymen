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
