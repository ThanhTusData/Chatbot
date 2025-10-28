from fastapi import APIRouter, HTTPException
from serving.schemas import RetrieveRequest, RetrieveResponse, RetrievalResult

router = APIRouter(prefix="/retrieve", tags=["retrieval"])

@router.post("", response_model=RetrieveResponse)
async def retrieve_documents(request: RetrieveRequest):
    from serving.fastapi_app import searcher
    
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
        raise HTTPException(status_code=500, detail=str(e))