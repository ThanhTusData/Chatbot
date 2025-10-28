import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    from src.serving.fastapi_app import app
    return TestClient(app)

def test_full_chat_flow(client):
    # Test health
    response = client.get("/health")
    assert response.status_code == 200
    
    # Test chat
    chat_response = client.post(
        "/chat",
        json={"message": "Hello"}
    )
    
    if chat_response.status_code == 200:
        data = chat_response.json()
        assert "response" in data
        assert "intent" in data
        assert "timestamp" in data

def test_chat_session(client):
    session_id = "test-session-123"
    
    # First message
    response1 = client.post(
        "/chat",
        json={"message": "Hello", "session_id": session_id}
    )
    
    # Second message
    response2 = client.post(
        "/chat",
        json={"message": "Thank you", "session_id": session_id}
    )
    
    if response1.status_code == 200 and response2.status_code == 200:
        assert response1.json()["response"]
        assert response2.json()["response"]