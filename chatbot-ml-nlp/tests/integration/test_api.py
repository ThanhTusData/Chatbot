import pytest
from fastapi.testclient import TestClient
from src.serving.fastapi_app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_predict_endpoint():
    response = client.post(
        "/predict",
        json={"text": "Hello", "top_k": 3}
    )
    
    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert isinstance(data["predictions"], list)

@pytest.mark.asyncio
async def test_chat_endpoint():
    response = client.post(
        "/chat",
        json={"message": "Hello"}
    )
    
    if response.status_code == 200:
        data = response.json()
        assert "response" in data
        assert "intent" in data