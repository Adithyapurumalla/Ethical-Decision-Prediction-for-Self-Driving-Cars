"""
Self-Driving Car Ethics AI - FastAPI Backend Server & Static Host

Exposes REST API endpoints for real-time model inference, batch processing,
analytics data, and training report artifacts while serving the React.js frontend SPA.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Initialize FastAPI App
app = FastAPI(
    title="Ethical Decision Prediction for Self-Driving Cars",
    description="REST API & Web Platform for Autonomous Vehicle Ethical Decision Prediction",
    version="1.0.0"
)

# Enable CORS for React frontend (http://localhost:5173 or production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static outputs folder for serving EDA images & reports
outputs_path = Path("outputs")
if outputs_path.exists():
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Mount static assets from built React frontend
frontend_dist = Path("frontend/dist")
if (frontend_dist / "assets").exists():
    app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")

# ==============================================================================
# CACHED MODEL & ARTIFACT LOADERS
# ==============================================================================

MODEL_PATH = Path("models/best_model.pkl")
PREPROCESSOR_PATH = Path("models/preprocessor.pkl")
METRICS_PATH = Path("models/model_metrics.csv")
REPORT_PATH = Path("outputs/models/model_report.txt")

model = None
preprocessor = None

if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
        print(f"[SUCCESS] Loaded ML Model from '{MODEL_PATH}'")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")

if PREPROCESSOR_PATH.exists():
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print(f"[SUCCESS] Loaded Preprocessor from '{PREPROCESSOR_PATH}'")
    except Exception as e:
        print(f"[ERROR] Failed to load preprocessor: {e}")


# ==============================================================================
# DATA SCHEMAS
# ==============================================================================

class ScenarioInput(BaseModel):
    speed_mph: int = 45
    pedestrian_count: int = 3
    passenger_count: int = 1
    obstacle_type: str = "pedestrian_group"
    pedestrian_jaywalking: int = 0
    weather_condition: str = "clear"
    brake_status: str = "failed"
    road_type: str = "city_street"
    time_of_day: str = "day"
    ethical_score: float = 0.85


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def generate_explanation(inputs: Dict[str, Any], decision: str, confidence: float) -> str:
    """Generates natural language narrative explaining AI decision."""
    speed = inputs.get("speed_mph", 40)
    pedestrians = inputs.get("pedestrian_count", 1)
    passengers = inputs.get("passenger_count", 1)
    obstacle = inputs.get("obstacle_type", "pedestrian_group")
    brakes = inputs.get("brake_status", "functional")
    jaywalking = inputs.get("pedestrian_jaywalking", 0)

    reasons = []
    if brakes == "failed":
        reasons.append("the vehicle's mechanical brakes have failed")
    else:
        reasons.append("brakes are operational")

    if pedestrians > passengers:
        reasons.append(f"pedestrians ({pedestrians}) outnumber vehicle occupants ({passengers})")
    elif passengers > pedestrians:
        reasons.append(f"vehicle occupants ({passengers}) outnumber pedestrians ({pedestrians})")

    if obstacle == "pedestrian_group":
        reasons.append("a group of human pedestrians is directly in the path")
    elif obstacle == "school_zone":
        reasons.append("the scenario occurs within a designated school safety zone")

    if jaywalking == 1:
        reasons.append("the pedestrians crossed against the legal traffic signal")

    return (
        f"With {confidence:.1f}% confidence, the AI chose to '{decision.upper().replace('_', ' ')}' "
        f"because " + "; ".join(reasons) + ". "
        f"This decision prioritizes minimizing total casualties based on survey preferences."
    )


# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }


@app.post("/api/predict")
def predict_scenario(scenario: ScenarioInput):
    if model is None:
        raise HTTPException(status_code=500, detail="ML Model is not loaded. Please run train_model.py.")

    input_data = scenario.model_dump()
    input_df = pd.DataFrame([input_data])

    try:
        proba = model.predict_proba(input_df)[0]
        pred_idx = int(np.argmax(proba))

        classifier = model.named_steps["classifier"]
        classes = list(getattr(classifier, "classes_", [0, 1, 2, 3]))
        label_map = {0: "brake_hard", 1: "maintain_course", 2: "swerve_left", 3: "swerve_right"}

        if isinstance(classes[0], (int, np.integer)):
            class_labels = [label_map.get(c, str(c)) for c in classes]
        else:
            class_labels = [str(c) for c in classes]

        pred_decision = class_labels[pred_idx]
        confidence = float(proba[pred_idx] * 100)

        probabilities = [
            {"label": c.replace("_", " ").title(), "probability": round(float(p) * 100, 2)}
            for c, p in zip(class_labels, proba)
        ]
        probabilities.sort(key=lambda x: x["probability"], reverse=True)

        explanation = generate_explanation(input_data, pred_decision, confidence)

        return {
            "prediction": pred_decision,
            "confidence": round(confidence, 2),
            "probabilities": probabilities,
            "explanation": explanation,
            "input_scenario": input_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/api/batch-predict")
def batch_predict(scenarios: List[ScenarioInput]):
    if model is None:
        raise HTTPException(status_code=500, detail="ML Model is not loaded.")

    if not scenarios:
        return {"results": []}

    rows = [s.model_dump() for s in scenarios]
    df = pd.DataFrame(rows)

    try:
        preds = model.predict(df)
        probas = model.predict_proba(df)
        max_probas = np.max(probas, axis=1) * 100

        classifier = model.named_steps["classifier"]
        classes = list(getattr(classifier, "classes_", [0, 1, 2, 3]))
        label_map = {0: "brake_hard", 1: "maintain_course", 2: "swerve_left", 3: "swerve_right"}

        if isinstance(classes[0], (int, np.integer)):
            pred_labels = [label_map.get(p, str(p)) for p in preds]
        else:
            pred_labels = [str(p) for p in preds]

        results = []
        for i, row in enumerate(rows):
            results.append({
                **row,
                "Predicted_Decision": pred_labels[i],
                "Confidence_%": round(float(max_probas[i]), 2)
            })

        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/api/metrics")
def get_metrics():
    if not METRICS_PATH.exists():
        return {"metrics": []}
    try:
        df = pd.read_csv(METRICS_PATH)
        return {"metrics": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics error: {str(e)}")


@app.get("/api/report")
def get_report():
    if not REPORT_PATH.exists():
        return {"report": "No training report available."}
    return {"report": REPORT_PATH.read_text(encoding="utf-8")}


@app.get("/api/eda-images")
def get_eda_images():
    eda_dir = Path("outputs/eda")
    if not eda_dir.exists():
        return {"images": []}
    images = [f.name for f in eda_dir.glob("*.png")]
    return {"images": sorted(images)}


# ==============================================================================
# REACT FRONTEND SPA CATCH-ALL ROUTE
# ==============================================================================

@app.get("/{full_path:path}")
def serve_react_frontend(full_path: str):
    file_path = frontend_dist / full_path
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    index_path = frontend_dist / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "API active. React frontend build not found."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
