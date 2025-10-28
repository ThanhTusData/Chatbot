from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
import time
from typing import Dict
import logging

from serving.schemas import (
    ChatRequest, ChatResponse, HealthResponse,
    PredictRequest, PredictResponse,
    RetrieveRequest, RetrieveResponse,
    IntentResult, RetrievalResult
)
from classification.predictor import IntentPredictor
from retrieval.search import SemanticSearch
from response.response_generator import ResponseGenerator
from config.config import config

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chatbot ML/NLP API",
    description="Advanced chatbot with intent classification and semantic retrieval",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
predictor: IntentPredictor = None
searcher: SemanticSearch = None
response_gen: ResponseGenerator = None

@app.on_event("startup")
async def startup_event():
    global predictor, searcher, response_gen
    
    logger.info("Loading models...")
    try:
        predictor = IntentPredictor(
            model_path=config.INTENT_MODEL_PATH,
            tokenizer_path=config.TOKENIZER_PATH
        )
        
        searcher = SemanticSearch(
            vectorstore_path=config.KB_INDEX_PATH,
            embedding_model=config.EMBEDDING_MODEL
        )
        
        response_gen = ResponseGenerator()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="running",
        version="1.0.0",
        models_loaded=predictor is not None and searcher is not None
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if predictor and searcher else "unhealthy",
        version="1.0.0",
        models_loaded=predictor is not None and searcher is not None
    )

@app.post("/predict", response_model=PredictResponse)
async def predict_intent(request: PredictRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        predictions = predictor.predict(request.text, top_k=request.top_k)
        processing_time = time.time() - start_time
        
        return PredictResponse(
            predictions=[IntentResult(**pred) for pred in predictions],
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_documents(request: RetrieveRequest):
    if searcher is None:
        raise HTTPException(status_code=503, detail="Search engine not loaded")
    
    try:
        results = searcher.search(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold
        )
        
        return RetrieveResponse(
            results=[
                RetrievalResult(
                    content=doc.get('content', ''),
                    score=doc['score'],
                    metadata=doc.get('metadata')
                ) for doc in results
            ],
            query=request.query
        )
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not (predictor and searcher and response_gen):
        raise HTTPException(status_code=503, detail="Services not loaded")
    
    try:
        intent_results = predictor.predict(request.message, top_k=1)
        top_intent = intent_results[0]
        
        retrieved_docs = searcher.search(request.message, top_k=5)
        
        response_text = response_gen.generate_with_retrieval(
            intent=top_intent['intent'],
            confidence=top_intent['confidence'],
            retrieved_docs=retrieved_docs
        )
        
        return ChatResponse(
            response=response_text,
            intent=IntentResult(**top_intent),
            retrieved_docs=[
                RetrievalResult(
                    content=doc.get('content', ''),
                    score=doc['score'],
                    metadata=doc.get('metadata')
                ) for doc in retrieved_docs
            ],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    uvicorn.run(
        "serving.fastapi_app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.DEBUG
    )

if __name__ == "__main__":
    main()