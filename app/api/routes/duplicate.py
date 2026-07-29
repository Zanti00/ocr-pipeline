from fastapi import APIRouter, Depends, HTTPException, status
from app.api.schemas.duplicate import DuplicateCheckRequest, DuplicateCheckResponse
from app.dependencies import verify_api_key
from app.db.postgres import AsyncSessionLocal
from app.embeddings.generator import EmbeddingGenerator
from app.embeddings.similarity import find_similar_receipts

router = APIRouter(prefix="/api/duplicate", tags=["Duplicate Detection"])

@router.post("-check", response_model=DuplicateCheckResponse)
async def check_duplicate(request: DuplicateCheckRequest, source_service: str = Depends(verify_api_key)):
    request.source_service = source_service

    if not request.receipt_text or not request.receipt_text.strip():
        return DuplicateCheckResponse(is_duplicate=False, matches=[])

    try:
        embedding = EmbeddingGenerator.generate(request.receipt_text)
        async with AsyncSessionLocal() as session:
            matches = await find_similar_receipts(
                session=session,
                embedding=embedding,
                source_service=request.source_service,
                threshold=request.threshold,
                days_window=request.days_window,
            )
        return DuplicateCheckResponse(
            is_duplicate=len(matches) > 0,
            matches=matches
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Duplicate check failed: {str(exc)}"
        )
