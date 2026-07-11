from fastapi import APIRouter, Depends
from app.api.schemas.ocr import OcrProcessRequest, OcrProcessResponse
from app.dependencies import verify_api_key
from app.db.mongodb import MongoDBClient
from app.tasks.process_receipt import process_receipt_task
import uuid

router = APIRouter(prefix="/api/ocr", tags=["OCR"])

@router.post("/process", response_model=OcrProcessResponse, status_code=202)
async def process_receipt(request: OcrProcessRequest, source_service: str = Depends(verify_api_key)):
    request.source_service = source_service

    job_id = str(uuid.uuid4())
    
    job_data = {
        "job_id": job_id,
        "receipt_id": request.receipt_id,
        "file_url": request.file_url,
        "callback_url": request.callback_url,
        "source_service": request.source_service,
        "status": "queued"
    }
    await MongoDBClient.create_job(job_data)
    
    process_receipt_task.delay(
        job_id=job_id,
        receipt_id=request.receipt_id,
        file_url=request.file_url,
        callback_url=request.callback_url,
        source_service=request.source_service
    )
    
    return OcrProcessResponse(
        job_id=job_id,
        status="queued",
        message="Receipt queued for OCR processing."
    )
