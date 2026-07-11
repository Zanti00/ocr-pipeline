from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    async def extract_receipt_fields(self, ocr_text: str) -> dict:
        """
        Extracts structured receipt fields from raw OCR text.
        Returns a dictionary matching the expected schema.
        """
        pass
