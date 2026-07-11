from fastapi import APIRouter, Depends
from app.api.schemas.duplicate import DuplicateCheckRequest, DuplicateCheckResponse
from app.dependencies import verify_api_key

router = APIRouter(prefix="/api/duplicate", tags=["Duplicate Detection"])

@router.post("-check", response_model=DuplicateCheckResponse)
async def check_duplicate(request: DuplicateCheckRequest, source_service: str = Depends(verify_api_key)):
    request.source_service = source_service
    
    # TODO: Generate embedding and query pgvector
    
    return DuplicateCheckResponse(
        is_duplicate=False,
        matches=[]
    )
