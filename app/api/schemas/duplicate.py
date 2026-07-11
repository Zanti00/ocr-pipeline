from pydantic import BaseModel
from typing import List

class DuplicateCheckRequest(BaseModel):
    receipt_text: str
    source_service: str
    threshold: float = 0.85
    days_window: int = 90

class DuplicateMatch(BaseModel):
    receipt_id: int
    source_service: str
    similarity_score: float
    processed_at: str

class DuplicateCheckResponse(BaseModel):
    is_duplicate: bool
    matches: List[DuplicateMatch]
