import argparse
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score, classification_report
import json
from pathlib import Path
import os

# Optional MLflow integration
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except Exception:
    _MLFLOW_AVAILABLE = False

def main(out_path: str):
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_path)
    metrics = {"accuracy": acc, "report": report}
    metrics_path = Path(out_path).with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved model to {out_path} with accuracy {acc}")

    # MLflow logging (optional). If MLFLOW_TRACKING_URI is set in env, MLflow will log there.
    if _MLFLOW_AVAILABLE:
        # Default to a local file store if no tracking URI provided
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(f"file://{str(Path('mlruns').absolute())}")
        with mlflow.start_run():
            mlflow.log_metric("accuracy", acc)
            # Log artifacts (model file and metrics)
            try:
                mlflow.log_artifact(out_path, artifact_path="model")
            except Exception:
                pass
            try:
                mlflow.log_artifact(str(metrics_path), artifact_path="model_metrics")
            except Exception:
                pass
            # Try to log sklearn model
            try:
                import mlflow.sklearn
                mlflow.sklearn.log_model(clf, artifact_path="sklearn_model")
            except Exception:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", dest="out_path", default="models/model_v1.joblib")
    args = parser.parse_args()
    main(args.out_path)
