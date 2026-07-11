from fastapi import APIRouter, Depends
from app.dependencies import verify_api_key

router = APIRouter(prefix="/api/metrics", tags=["Metrics"])

@router.get("")
async def get_metrics(source_service: str = Depends(verify_api_key)):
    # TODO: Aggregate from MongoDB
    return {
        "total_jobs_processed": 0,
        "jobs_last_24h": 0,
        "average_processing_time_ms": 0,
        "success_rate": 0.0,
        "average_confidence_score": 0.0,
        "queue_depth": 0
    }
