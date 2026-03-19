#!/usr/bin/env python3
"""
Model training script for Iris dataset.
Saves model to app/models/model.pkl
"""

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import os
import json
from datetime import datetime

def train_model():
    """Train and save the model."""
    print("🌳 Training Iris Classifier")
    print("=" * 50)
    
    # Load data
    print("📊 Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]}")
    print(f"   Classes: {iris.target_names}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Train model
    print("\n🔄 Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"📊 Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Test set evaluation
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"✅ Test accuracy: {accuracy:.4f}")
    
    # Detailed metrics
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Feature importance
    print("\n🔥 Feature Importance:")
    for name, importance in zip(iris.feature_names, model.feature_importances_):
        print(f"   {name}: {importance:.4f}")
    
    # Save model
    os.makedirs("app/models", exist_ok=True)
    model_path = "app/models/model.pkl"
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved to {model_path}")
    
    # Save metadata
    metadata = {
        "model_type": type(model).__name__,
        "accuracy": accuracy,
        "features": iris.feature_names,
        "classes": iris.target_names.tolist(),
        "n_estimators": model.n_estimators,
        "trained_at": datetime.now().isoformat()
    }
    
    with open("app/models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"📝 Metadata saved to app/models/metadata.json")
    
    return model

if __name__ == "__main__":
    train_model()
