from fastapi import APIRouter, Depends, HTTPException
from app.api.schemas.jobs import JobStatusResponse
from app.dependencies import verify_api_key

router = APIRouter(prefix="/api/jobs", tags=["Jobs"])

@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str, source_service: str = Depends(verify_api_key)):
    # TODO: Read from MongoDB
    
    raise HTTPException(status_code=404, detail="Job not found")
