"""FastAPI service wrapping the tabular classifier."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import CLASS_NAMES
from src.model_loader import load_active_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_active_model()
    yield


app = FastAPI(
    title="Project D — Tabular ML API",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    sepal_length: float = Field(..., ge=0, le=20)
    sepal_width: float = Field(..., ge=0, le=20)
    petal_length: float = Field(..., ge=0, le=20)
    petal_width: float = Field(..., ge=0, le=20)


class PredictResponse(BaseModel):
    label_index: int
    label_name: str
    probabilities: dict[str, float]
    model_version: str | None
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/model")
def model_info():
    _, registry = load_active_model()
    return {
        "active_version": registry.get("active_version"),
        "active_model_path": registry.get("active_model_path"),
    }


@app.post("/v1/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    t0 = time.perf_counter()
    model, registry = load_active_model()
    X = [
        [
            body.sepal_length,
            body.sepal_width,
            body.petal_length,
            body.petal_width,
        ]
    ]
    try:
        idx = int(model.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    proba = model.predict_proba(X)[0]
    names = list(CLASS_NAMES)
    if len(proba) != len(names):
        raise HTTPException(status_code=500, detail="Class count mismatch.")
    probs = {names[i]: float(proba[i]) for i in range(len(names))}
    ms = (time.perf_counter() - t0) * 1000
    return PredictResponse(
        label_index=idx,
        label_name=names[idx],
        probabilities=probs,
        model_version=registry.get("active_version"),
        latency_ms=round(ms, 3),
    )


def main():
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("src.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
