# tests/test_e2e_chat.py
import os
import json
import pytest

# Prefer TestClient if FastAPI app present; otherwise fall back to network requests.
try:
    from fastapi.testclient import TestClient  # type: ignore
    from src.serving.fastapi_app import app as fastapi_app  # type: ignore
    USE_TESTCLIENT = True
    client = TestClient(fastapi_app)
except Exception:
    USE_TESTCLIENT = False
    import requests  # type: ignore

SERVICE_URL = os.getenv("SERVICE_URL", "http://localhost:8000")
AUTH_HEADER = os.getenv("CANARY_AUTH_HEADER")  # optional, e.g. "Bearer dev-token"

def _post(path: str, json_body: dict):
    if USE_TESTCLIENT:
        headers = {}
        if AUTH_HEADER:
            # If using TestClient, supply Authorization header
            headers["Authorization"] = AUTH_HEADER
        resp = client.post(path, json=json_body, headers=headers)
        try:
            return resp.status_code, resp.json()
        except Exception:
            return resp.status_code, resp.text
    else:
        url = SERVICE_URL.rstrip("/") + path
        headers = {"Content-Type": "application/json"}
        if AUTH_HEADER:
            headers["Authorization"] = AUTH_HEADER
        resp = requests.post(url, json=json_body, headers=headers, timeout=10)
        try:
            return resp.status_code, resp.json()
        except Exception:
            return resp.status_code, resp.text

def _get(path: str):
    if USE_TESTCLIENT:
        headers = {}
        if AUTH_HEADER:
            headers["Authorization"] = AUTH_HEADER
        resp = client.get(path, headers=headers)
        try:
            return resp.status_code, resp.json()
        except Exception:
            return resp.status_code, resp.text
    else:
        url = SERVICE_URL.rstrip("/") + path
        headers = {}
        if AUTH_HEADER:
            headers["Authorization"] = AUTH_HEADER
        resp = requests.get(url, headers=headers, timeout=10)
        try:
            return resp.status_code, resp.json()
        except Exception:
            return resp.status_code, resp.text

@pytest.mark.integration
def test_health_endpoint():
    """Health should return status and keys model_loaded/retrieval. Skip if unreachable."""
    code, body = _get("/health")
    if code != 200:
        pytest.skip(f"Health endpoint not reachable (status {code}). SERVICE_URL={SERVICE_URL}")
    assert isinstance(body, dict), "Health did not return JSON"
    assert "status" in body
    assert body.get("status") == "ok"
    # model_loaded and retrieval may be True/False depending on environment; just ensure keys exist
    assert "model_loaded" in body
    assert "retrieval" in body

@pytest.mark.integration
def test_predict_basic():
    """Call /predict with a simple text. If model not loaded, skip test (canary script should enforce model_loaded)."""
    req = {"texts": ["canary smoke test"]}
    code, body = _post("/predict", req)
    if code == 503:
        pytest.skip("Model not loaded (503).")
    assert code == 200, f"/predict returned {code} - {body}"
    assert isinstance(body, dict)
    preds = body.get("predictions")
    assert isinstance(preds, list) and len(preds) > 0
    # each prediction should have text, intent, confidence
    p = preds[0]
    assert "intent" in p
    assert "confidence" in p
    assert isinstance(p["confidence"], (float, int))

@pytest.mark.integration
def test_retrieve_basic():
    """Call /retrieve with a simple query."""
    req = {"query": "reset password", "top_k": 3}
    code, body = _post("/retrieve", req)
    if code == 503:
        pytest.skip("Retrieval not initialized (503).")
    assert code == 200, f"/retrieve returned {code} - {body}"
    assert isinstance(body, dict)
    results = body.get("results")
    assert isinstance(results, list)
