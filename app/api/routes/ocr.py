import uuid
from io import BytesIO
import logging
import httpx
from PIL import Image
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from app.api.schemas.ocr import (
    OcrProcessRequest, OcrProcessResponse, QualityRejectionResponse, build_callback_payload,
)
from app.core.callback import send_callback
from app.core.image_quality import check_image_quality
from app.db.mongodb import MongoDBClient
from app.dependencies import verify_api_key
from app.tasks.process_receipt import process_receipt_task

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ocr", tags=["OCR"])


async def _download_file(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        res = await client.get(url, timeout=30.0)
        res.raise_for_status()
        return res.content


@router.post("/process", response_model=OcrProcessResponse, status_code=202)
async def process_receipt(
    request: OcrProcessRequest,
    source_service: str = Depends(verify_api_key),
):
    request.source_service = source_service

    urls = request.file_urls if (request.file_urls and len(request.file_urls) > 0) else ([request.file_url] if request.file_url else [])
    if not urls:
        raise HTTPException(status_code=400, detail="Either file_url or file_urls must be provided.")

    # Synchronous pre-OCR quality check if force_process is False.
    # Rejection is immediate (HTTP 422) — we also dispatch a callback if callback_url is present
    # so async consumers receive the rejection event.
    if not request.force_process:
        try:
            for idx, url in enumerate(urls):
                file_bytes = await _download_file(url)
                # Load PIL image
                if url.lower().split("?")[0].endswith(".pdf"):
                    # Quality check on PDF is skipped or performed on first page
                    pass
                else:
                    pil_img = Image.open(BytesIO(file_bytes))
                    result = check_image_quality(pil_img)
                    if not result.passed:
                        segment_idx = idx if len(urls) > 1 else None
                        reason = result.rejection_reason or "Image quality check failed."
                        code = result.rejection_code or "quality_failed"
                        job_id = str(uuid.uuid4())
                        try:
                            await MongoDBClient.create_job({
                                "job_id": job_id,
                                "receipt_id": request.receipt_id,
                                "file_url": request.file_url or (
                                    request.file_urls[0] if request.file_urls else ""
                                ),
                                "file_urls": request.file_urls or [],
                                "callback_url": request.callback_url,
                                "source_service": request.source_service,
                                "force_process": request.force_process,
                                "country": request.country,
                                "currency": request.currency,
                                "location": request.location,
                                "status": "rejected",
                                "error": reason,
                                "rejection_code": code,
                                "quality_check": {
                                    "rejection_code": code,
                                    "rejection_reason": reason,
                                    "blur_score": result.blur_score,
                                    "brightness": result.brightness,
                                    "resolution": list(result.resolution),
                                    "segment_index": segment_idx,
                                },
                            })
                        except Exception as mongo_exc:
                            logger.error(
                                "Failed to persist quality rejection for receipt_id=%s: %s",
                                request.receipt_id, mongo_exc, exc_info=True,
                            )

                        if request.callback_url:
                            cb_payload = build_callback_payload(
                                receipt_id=request.receipt_id,
                                status="rejected",
                                error=reason,
                                rejection_code=code,
                                rejection_reason=reason,
                            )
                            try:
                                await send_callback(request.callback_url, cb_payload.model_dump())
                            except Exception as cb_exc:
                                logger.error(
                                    "Failed to send quality rejection callback for receipt_id=%s: %s",
                                    request.receipt_id, cb_exc, exc_info=True,
                                )

                        response_payload = QualityRejectionResponse(
                            status="rejected",
                            message=reason,
                            rejection_code=code,
                            rejection_reason=reason,
                            blur_score=result.blur_score,
                            brightness=result.brightness,
                            resolution=result.resolution,
                            segment_index=segment_idx,
                            job_id=job_id,
                            receipt_id=request.receipt_id,
                        )
                        logger.warning(
                            "Receipt quality check rejected receipt_id=%s (code=%s, segment=%s, job=%s)",
                            request.receipt_id, code, segment_idx, job_id,
                        )
                        return JSONResponse(
                            status_code=422,
                            content=response_payload.model_dump(),
                        )
        except Exception as exc:
            logger.error("Pre-OCR quality check failed with exception: %s", exc, exc_info=True)
            # If image download fails during quality check, let Celery task handle download error or fail fast
            pass

    job_id = str(uuid.uuid4())

    job_data = {
        "job_id": job_id,
        "receipt_id": request.receipt_id,
        "file_url": request.file_url or (request.file_urls[0] if request.file_urls else ""),
        "file_urls": request.file_urls or [],
        "callback_url": request.callback_url,
        "source_service": request.source_service,
        "force_process": request.force_process,
        "country": request.country,
        "currency": request.currency,
        "location": request.location,
        "status": "queued",
    }
    await MongoDBClient.create_job(job_data)

    process_receipt_task.delay(
        job_id=job_id,
        receipt_id=request.receipt_id,
        file_url=request.file_url or "",
        file_urls=request.file_urls or [],
        callback_url=request.callback_url,
        source_service=request.source_service,
        force_process=request.force_process,
        country=request.country,
        currency=request.currency,
        location=request.location,
    )

    return OcrProcessResponse(
        job_id=job_id,
        status="queued",
        message="Receipt queued for OCR processing.",
    )
