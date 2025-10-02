# src/serving/schemas.py
from typing import List, Optional
from pydantic import BaseModel

class PredictRequest(BaseModel):
    texts: List[str]

class PredictionItem(BaseModel):
    text: str
    intent: str
    confidence: float

class PredictResponse(BaseModel):
    predictions: List[PredictionItem]

class RetrieveRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class RetrieveItem(BaseModel):
    id: str
    score: float
    text: Optional[str] = None
    answer: Optional[str] = None

class RetrieveResponse(BaseModel):
    query: str
    results: List[RetrieveItem]
