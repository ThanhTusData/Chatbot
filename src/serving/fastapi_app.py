# src/serving/fastapi_app.py
import os
import time
from typing import List

from fastapi import FastAPI, Depends, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import uvicorn
import logging

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.serving.schemas import PredictRequest, PredictResponse, PredictionItem, RetrieveRequest, RetrieveResponse
from src.serving.auth import get_current_user
from src.models.intent_model import IntentModel
from src.retrieval.serve_helpers import init as init_retrieval, retrieve_for_api
from src.serving.logging_config import get_logger

LOG = get_logger("fastapi_app")

app = FastAPI(title="Intent + Retrieval Service")

MODEL_DIR = os.getenv("INTENT_MODEL_DIR", "models/intent/latest")
INDEX_DIR = os.getenv("INDEX_DIR", "indexes/kb")

# Prometheus metrics
REQUEST_COUNT = Counter("app_requests_total", "Total HTTP requests", ["method", "endpoint", "http_status"])
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Request latency (seconds)", ["endpoint"])
INFERENCE_TIME = Histogram("inference_time_seconds", "Model inference time", ["model"])

# lazy load holders
_model: IntentModel = None
_retrieval_initialized = False


@app.on_event("startup")
def startup():
    global _model, _retrieval_initialized
    try:
        _model = IntentModel.load(MODEL_DIR)
        LOG.info(f"Loaded intent model from {MODEL_DIR} metadata={getattr(_model, 'metadata', None)}")
    except Exception as e:
        LOG.warning(f"Could not load intent model from {MODEL_DIR}: {e}")
        _model = None

    try:
        init_retrieval(INDEX_DIR)
        _retrieval_initialized = True
        LOG.info(f"Initialized retrieval from {INDEX_DIR}")
    except Exception as e:
        LOG.warning(f"Retrieval init failed for {INDEX_DIR}: {e}")
        _retrieval_initialized = False


# simple middleware-like decorator to record metrics
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.time()
    path = request.url.path
    method = request.method
    try:
        response: Response = await call_next(request)
        status = response.status_code
    except Exception as e:
        status = 500
        LOG.exception("Unhandled exception handling request")
        raise
    finally:
        elapsed = time.time() - start
        REQUEST_LATENCY.labels(endpoint=path).observe(elapsed)
        REQUEST_COUNT.labels(method=method, endpoint=path, http_status=str(status)).inc()
    return response


@app.get("/metrics")
def metrics():
    # Expose prometheus metrics
    data = generate_latest()
    return PlainTextResponse(data, media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "retrieval": _retrieval_initialized}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, user=Depends(get_current_user)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    texts = req.texts
    t0 = time.time()
    preds = _model.predict(texts)
    t_infer = time.time() - t0
    INFERENCE_TIME.labels(model=getattr(_model, "name", "intent_model")).observe(t_infer)

    probs = _model.predict_proba(texts)
    out = []
    for t, p, pr in zip(texts, preds, probs):
        # take max prob as confidence
        conf = float(max(pr)) if isinstance(pr, (list, tuple)) else float(pr)
        # redact or avoid logging raw user text in production logs
        LOG.info("predicted intent", extra={"intent": str(p), "confidence": conf})
        out.append(PredictionItem(text=t, intent=str(p), confidence=conf))
    return PredictResponse(predictions=out)


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest, user=Depends(get_current_user)):
    if not _retrieval_initialized:
        raise HTTPException(status_code=503, detail="Retrieval not initialized")
    results = retrieve_for_api(req.query, top_k=req.top_k)
    return RetrieveResponse(query=req.query, results=results)


if __name__ == "__main__":
    uvicorn.run("src.serving.fastapi_app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
