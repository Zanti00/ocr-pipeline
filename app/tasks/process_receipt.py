import asyncio
from app.core.pipeline import process_receipt
from app.tasks.celery_app import celery_app

_loop = None


def get_loop():
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
    return _loop


@celery_app.task(bind=True, max_retries=3)
def process_receipt_task(
    self,
    job_id: str,
    receipt_id: int,
    file_url: str,
    callback_url: str,
    source_service: str,
    force_process: bool = False,
    file_urls: list[str] | None = None,
    country: str | None = None,
    currency: str | None = None,
    location: str | None = None,
):
    loop = get_loop()
    try:
        loop.run_until_complete(
            process_receipt(
                job_id,
                receipt_id,
                file_url,
                callback_url,
                source_service,
                force_process=force_process,
                file_urls=file_urls,
                country=country,
                currency=currency,
                location=location,
            )
        )
    except Exception as exc:
        self.retry(exc=exc, countdown=10 * (self.request.retries + 1))
