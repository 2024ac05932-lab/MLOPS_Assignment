"""
Heart Disease Prediction API.

Serves the trained pipeline (StandardScaler + RandomForestClassifier) behind
a FastAPI app. Combines:
  - Task 6: /predict endpoint returning prediction + confidence
  - Task 8: request logging + Prometheus /metrics endpoint

Run locally with:
    uvicorn app:app --host 0.0.0.0 --port 8000

Or via Docker (see Dockerfile).
"""
import logging

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CollectorRegistry, Counter, generate_latest
from pydantic import BaseModel

logging.basicConfig(
    filename="api_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    force=True,
)

# Private registry (not the global default) so re-importing this module in
# the same process can never raise "Duplicated timeseries in CollectorRegistry".
REGISTRY = CollectorRegistry()
REQUEST_COUNT = Counter(
    "api_requests_total", "Total prediction requests received", registry=REGISTRY
)

model = joblib.load("heart_disease_pipeline.joblib")

app = FastAPI(title="Heart Disease Prediction API")


class HeartData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


@app.get("/")
def home():
    logging.info("Home endpoint accessed")
    return {"message": "Heart Disease Prediction API"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(REGISTRY), media_type="text/plain")


@app.post("/predict")
def predict(data: HeartData):
    REQUEST_COUNT.inc()
    logging.info(f"Prediction request received: {data}")

    features = np.array([[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal,
    ]])

    prediction = int(model.predict(features)[0])
    confidence = float(max(model.predict_proba(features)[0]))

    logging.info(f"Prediction result: {prediction}, confidence: {confidence:.4f}")

    return {"prediction": prediction, "confidence": round(confidence, 4)}
