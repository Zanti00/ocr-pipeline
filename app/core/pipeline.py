import httpx
import logging
from PIL import Image
from io import BytesIO
import asyncio

from app.core.preprocessing import preprocess_image
from app.core.pdf_handler import pdf_to_images
from app.core.ocr_engine import extract_text
from app.core.bir_validator import validate_tin, classify_vat
from app.core.confidence import compute_composite_score
from app.core.callback import send_callback
from app.llm.factory import create_provider
from app.embeddings.generator import EmbeddingGenerator
from app.db.mongodb import MongoDBClient
from app.db.postgres import AsyncSessionLocal
from app.db.models import ReceiptEmbedding

logger = logging.getLogger(__name__)

async def download_file(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=30.0)
        response.raise_for_status()
        return response.content

async def process_receipt(job_id: str, receipt_id: int, file_url: str, callback_url: str, source_service: str):
    logger.info(f"Starting OCR pipeline for job {job_id}")
    
    try:
        await MongoDBClient.update_job_status(job_id, "processing")
        
        # 1. Download File
        file_bytes = await download_file(file_url)
        
        # 2. Convert/Load Image
        images = []
        if file_url.lower().endswith(".pdf"):
            images = pdf_to_images(file_bytes)
        else:
            images = [Image.open(BytesIO(file_bytes))]
            
        if not images:
            raise ValueError("No images extracted from file")
            
        img = images[0]
        processed_img = preprocess_image(img)
        raw_text, tesseract_conf = extract_text(processed_img, lang="eng")
        
        # 3. LLM Extraction
        llm_provider = create_provider()
        extracted_data = await llm_provider.extract_receipt_fields(raw_text)
        
        # 4. BIR Validation
        tin = extracted_data.get("tin", "")
        tin_valid = validate_tin(tin)
        vat_class = classify_vat(raw_text, tin)
        extracted_data["vat_classification"] = vat_class
        
        # 5. Confidence Score
        composite_conf = compute_composite_score(tesseract_conf, extracted_data, tin_valid)
        
        # 6. Generate Embedding and Store in Postgres
        embedding = EmbeddingGenerator.generate(raw_text)
        async with AsyncSessionLocal() as session:
            db_embedding = ReceiptEmbedding(
                receipt_id=receipt_id,
                source_service=source_service,
                embedding=embedding,
                receipt_text=raw_text
            )
            session.add(db_embedding)
            await session.commit()
            
        # 7. Update MongoDB
        job_update = {
            "raw_ocr_text": raw_text,
            "tesseract_confidence": tesseract_conf,
            "composite_confidence_score": composite_conf,
            "extracted_data": extracted_data,
            "bir_validation": {
                "tin_valid": tin_valid,
                "vat_classification": vat_class
            }
        }
        await MongoDBClient.update_job_status(job_id, "completed", job_update)
        
        # 8. Send Callback
        payload = {
            "receipt_id": receipt_id,
            "ocr_confidence_score": composite_conf,
            "status": "completed",
            **extracted_data
        }
        
        callback_success = await send_callback(callback_url, payload)
        if not callback_success:
            raise RuntimeError(f"Callback failed to {callback_url}")
            
    except Exception as e:
        logger.error(f"Pipeline failed for job {job_id}: {str(e)}")
        await MongoDBClient.update_job_status(job_id, "failed", {"error": str(e)})
        
        # Send failure callback
        payload = {
            "receipt_id": receipt_id,
            "status": "failed",
            "error": str(e)
        }
        await send_callback(callback_url, payload)
        raise e
