from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message", min_length=1)
    session_id: Optional[str] = Field(None, description="Session ID for context")
    user_id: Optional[str] = Field(None, description="User ID")

class IntentResult(BaseModel):
    intent: str
    confidence: float

class RetrievalResult(BaseModel):
    content: str
    score: float
    metadata: Optional[Dict] = None

class ChatResponse(BaseModel):
    response: str
    intent: IntentResult
    retrieved_docs: List[RetrievalResult] = []
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=10)

class PredictResponse(BaseModel):
    predictions: List[IntentResult]
    processing_time: float

class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    threshold: float = Field(0.7, ge=0.0, le=1.0)

class RetrieveResponse(BaseModel):
    results: List[RetrievalResult]
    query: str