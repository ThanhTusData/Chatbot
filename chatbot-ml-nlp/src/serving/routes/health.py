from fastapi import APIRouter
from serving.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])

@router.get("", response_model=HealthResponse)
async def health_check():
    from serving.fastapi_app import predictor, searcher
    
    return HealthResponse(
        status="healthy" if predictor and searcher else "unhealthy",
        version="1.0.0",
        models_loaded=predictor is not None and searcher is not None
    )