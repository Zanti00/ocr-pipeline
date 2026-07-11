from app.tasks.celery_app import celery_app
from app.core.pipeline import process_receipt
import asyncio

@celery_app.task(bind=True, max_retries=3)
def process_receipt_task(self, job_id: str, receipt_id: int, file_url: str, callback_url: str, source_service: str):
    try:
        asyncio.run(process_receipt(job_id, receipt_id, file_url, callback_url, source_service))
    except Exception as exc:
        self.retry(exc=exc, countdown=10 * (self.request.retries + 1))
