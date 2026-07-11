from pydantic import BaseModel
from typing import Optional

class JobStatusResponse(BaseModel):
    job_id: str
    receipt_id: int
    source_service: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    duration_ms: Optional[int] = None
