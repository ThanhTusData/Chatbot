from fastapi import APIRouter, HTTPException
from serving.schemas import PredictRequest, PredictResponse, IntentResult
import time

router = APIRouter(prefix="/predict", tags=["prediction"])

@router.post("", response_model=PredictResponse)
async def predict_intent(request: PredictRequest):
    from serving.fastapi_app import predictor
    
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
        raise HTTPException(status_code=500, detail=str(e))