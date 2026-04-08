ML Model Deployment Pipeline — Summary

What this project is
- End-to-end ML deployment example that trains a model, evaluates it, packages it, and serves it through a production-style REST API.
- Tech: Python, scikit-learn, FastAPI, Docker, GitHub Actions CI, MLflow (local model registry).

Key Features (what to talk about in interviews)
- Training & evaluation: `src/train.py` trains a model, computes accuracy and classification report, and saves metrics next to the model artifact.
- Reproducible model packaging: trained models are saved under `models/` and included as artifacts in CI runs.
- Production-ready API: `src/app/main.py` exposes `/health` and `/predict` endpoints via FastAPI (typed payloads and JSON responses).
- Containerization: `Dockerfile` produces a small non-root image to serve the API with uvicorn.
- CI/CD: `.github/workflows/ci.yml` runs tests, builds the image, and trains a model during the test job to ensure deterministic tests.
- Model registry (MLOps): MLflow logging is integrated in training — the repo stores mlruns locally; CI uploads the trained model artifact for traceability.

How to run locally (elevator pitch)
1. Create venv & install: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
2. Train model: `python src/train.py --out models/model_v1.joblib` (this also logs to MLflow locally)
3. Serve model: `uvicorn src.app.main:app --reload`
4. Predict: `POST /predict` with JSON `{"inputs": [[...]]}`

Talk track for interviews
- "I built an end-to-end pipeline where I train models reproducibly, log artifacts with MLflow, run tests and build images in CI, and serve predictions via a typed FastAPI service in Docker. The CI trains a model so tests are deterministic and artifacts are uploaded — this mirrors production practices where models are tracked and versioned."
