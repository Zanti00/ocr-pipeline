import httpx
import json
from app.llm.base import LLMProvider
from app.llm.prompts import RECEIPT_EXTRACTION_PROMPT
from app.config import settings

class OllamaProvider(LLMProvider):
    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model

    async def extract_receipt_fields(self, ocr_text: str) -> dict:
        prompt = RECEIPT_EXTRACTION_PROMPT.format(ocr_text=ocr_text)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0
            }
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(f"{self.base_url}/api/generate", json=payload, timeout=120.0)
                response.raise_for_status()
                data = response.json()
                result_json = data.get("response", "{}")
                return json.loads(result_json)
            except Exception as e:
                raise RuntimeError(f"Ollama extraction failed: {str(e)}")
