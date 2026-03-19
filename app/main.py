from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import os
import time
import logging

from app.schemas import PredictionRequest, PredictionResponse, HealthResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(
    title="ML Model API",
    description="Production-ready machine learning model deployment",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = os.getenv("MODEL_PATH", "app/models/model.pkl")
model = None

@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info(f"✅ Model loaded from {MODEL_PATH}")
    else:
        logger.warning("⚠️ No model found. Run 'make train' first")

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "ML Model API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_path=MODEL_PATH if model else None
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """Make a single prediction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Convert features to numpy array
        features = np.array(request.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get probability if available
        probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            probability = float(max(probabilities))
        
        processing_time = time.time() - start_time
        
        logger.info(f"Prediction made in {processing_time:.4f}s")
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=probability,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(requests: list[PredictionRequest]):
    """Make multiple predictions."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Prepare batch features
        features_list = [req.features for req in requests]
        features = np.array(features_list)
        
        # Make predictions
        predictions = model.predict(features).tolist()
        
        processing_time = time.time() - start_time
        
        return {
            "predictions": predictions,
            "count": len(predictions),
            "processing_time": processing_time
        }
    
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "n_features": model.n_features_in_,
        "n_classes": len(model.classes_),
        "classes": model.classes_.tolist(),
        "feature_names": getattr(model, "feature_names_in_", None),
        "is_fitted": True
    }
