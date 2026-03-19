from pydantic import BaseModel, Field
from typing import List, Optional

class PredictionRequest(BaseModel):
    """Prediction request schema."""
    features: List[float] = Field(..., description="Feature values for prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }

class PredictionResponse(BaseModel):
    """Prediction response schema."""
    prediction: int = Field(..., description="Predicted class")
    probability: Optional[float] = Field(None, description="Prediction probability")
    processing_time: float = Field(..., description="Processing time in seconds")

class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    model_loaded: bool
    model_path: Optional[str] = None
