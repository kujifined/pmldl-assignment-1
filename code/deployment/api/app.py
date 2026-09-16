"""FastAPI inference service for the packaged penguin classifier."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

DEFAULT_MODEL_PATH = (Path(__file__).resolve().parent / "../../../models/model.joblib").resolve()


class PenguinFeatures(BaseModel):
    island: str
    sex: str
    bill_length_mm: float = Field(gt=0, le=100)
    bill_depth_mm: float = Field(gt=0, le=50)
    flipper_length_mm: float = Field(gt=0, le=400)
    body_mass_g: float = Field(gt=0, le=15_000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    if not model_path.exists():
        raise RuntimeError(f"Model artifact not found: {model_path}. Run Stage 2 first.")
    app.state.model = joblib.load(model_path)
    app.state.model_path = str(model_path)
    yield


app = FastAPI(title="Palmer Penguins API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    return {"status": "healthy", "model": request.app.state.model_path}


@app.post("/predict")
def predict(features: PenguinFeatures, request: Request) -> dict:
    frame = pd.DataFrame([features.model_dump()])
    try:
        prediction = str(request.app.state.model.predict(frame)[0])
        probabilities = request.app.state.model.predict_proba(frame)[0]
        classes = request.app.state.model.classes_
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc
    return {
        "prediction": prediction,
        "probabilities": {str(label): round(float(probability), 6) for label, probability in zip(classes, probabilities)},
    }
