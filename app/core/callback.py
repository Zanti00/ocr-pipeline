import httpx
import logging
from app.config import settings

logger = logging.getLogger(__name__)

async def send_callback(callback_url: str, payload: dict) -> bool:
    headers = {
        "Authorization": f"Bearer {settings.callback_api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(callback_url, json=payload, headers=headers, timeout=30.0)
            response.raise_for_status()
            return True
        except httpx.HTTPStatusError as e:
            logger.error(f"Callback failed with status {e.response.status_code} for {callback_url}: {e.response.text}")
            return False
        except httpx.RequestError as e:
            logger.error(f"Callback request error for {callback_url}: {str(e)}")
            return False
