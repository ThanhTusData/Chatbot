# tests/test_api.py
from fastapi.testclient import TestClient
from src.serving.fastapi_app import app
import os
import json

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert "status" in body

def test_predict_no_model(monkeypatch):
    # force model None to simulate missing
    import src.serving.fastapi_app as fa
    fa._model = None
    r = client.post("/predict", json={"texts": ["hello"]})
    assert r.status_code in (503, 200)

def test_predict_with_mock(monkeypatch):
    class MockModel:
        def predict(self, texts): return ["greeting" for _ in texts]
        def predict_proba(self, texts): return [[0.9, 0.1] for _ in texts]
    import src.serving.fastapi_app as fa
    fa._model = MockModel()
    r = client.post("/predict", json={"texts": ["hello", "hi"]})
    assert r.status_code == 200
    body = r.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 2

def test_retrieve_route(monkeypatch):
    # mock retrieve_for_api to not require index
    from src.serving import fastapi_app as fa
    def fake_retrieve(q, top_k=5):
        return [{"id": "d1", "score": 0.9, "text": "hello", "answer": ""}]
    monkeypatch.setattr("src.retrieval.serve_helpers.retrieve_for_api", fake_retrieve)
    fa._retrieval_initialized = True
    r = client.post("/retrieve", json={"query": "hello", "top_k": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "hello"
    assert len(body["results"]) == 1
