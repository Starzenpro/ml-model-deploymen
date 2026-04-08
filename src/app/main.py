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
