from pydantic import BaseModel
from typing import Dict

class HealthResponse(BaseModel):
    status: str
    dependencies: Dict[str, str]
    version: str
