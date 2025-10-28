import pytest
from src.serving.schemas import ChatRequest, ChatResponse, IntentResult

def test_chat_request_validation():
    request = ChatRequest(message="Hello")
    assert request.message == "Hello"
    
    with pytest.raises(ValueError):
        ChatRequest(message="")

def test_intent_result():
    result = IntentResult(intent="greeting", confidence=0.95)
    assert result.intent == "greeting"
    assert 0 <= result.confidence <= 1

def test_chat_response():
    response = ChatResponse(
        response="Hello!",
        intent=IntentResult(intent="greeting", confidence=0.95),
        timestamp="2025-01-01T00:00:00"
    )
    assert response.response == "Hello!"
    assert response.intent.confidence == 0.95