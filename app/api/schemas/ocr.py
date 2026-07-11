from pydantic import BaseModel
from typing import Optional, List

class OcrProcessRequest(BaseModel):
    receipt_id: int
    file_url: str
    callback_url: str
    source_service: str

class OcrProcessResponse(BaseModel):
    job_id: str
    status: str
    message: str

class ReceiptItem(BaseModel):
    name: str
    quantity: int
    price: float

class OcrCallbackPayload(BaseModel):
    receipt_id: int
    vendor_name: Optional[str] = None
    transaction_date: Optional[str] = None
    total_amount: Optional[float] = None
    vat_amount: Optional[float] = None
    tin: Optional[str] = None
    invoice_number: Optional[str] = None
    vat_classification: Optional[str] = None
    expense_category: Optional[str] = None
    ocr_confidence_score: float = 0.0
    items: List[ReceiptItem] = []
    status: Optional[str] = None
    error: Optional[str] = None
