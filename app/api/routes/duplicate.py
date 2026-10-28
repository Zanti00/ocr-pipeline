from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import delete

from app.api.schemas.duplicate import DuplicateCheckRequest, DuplicateCheckResponse
from app.db.models import ReceiptEmbedding
from app.db.postgres import AsyncSessionLocal
from app.dependencies import verify_api_key
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
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Duplicate check failed: {exc!s}"
        )


@router.delete("/embeddings/{receipt_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_receipt_embedding(
    receipt_id: int,
    source_service: str = Depends(verify_api_key),
):
    """Purge a receipt's embedding from pgvector upon cancellation or deletion."""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(
                delete(ReceiptEmbedding).where(
                    ReceiptEmbedding.receipt_id == receipt_id,
                    ReceiptEmbedding.source_service == source_service,
                )
            )
            await session.commit()
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete embedding: {exc!s}"
        )
