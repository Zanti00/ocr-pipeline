from typing import Any, Optional, List

from pydantic import BaseModel, Field, field_validator

from app.core.schema import EXPENSE_CATEGORIES
from app.core.verification import coerce_category

# ISO 4217 codes the pipeline and SERMS both understand.
# Mirrors ``COUNTRY_CURRENCY`` in ``app.core.locale`` so both sides stay in sync.
# Any currency outside this set is coerced to None rather than 422-ing the callback.
SUPPORTED_CURRENCIES: frozenset[str] = frozenset({
    "PHP", "USD", "BND", "MYR", "SGD", "JPY", "HKD", "THB", "AUD", "GBP", "EUR",
})


class OcrProcessRequest(BaseModel):
    receipt_id: int
    file_url: str
    callback_url: str
    source_service: Optional[str] = None


class OcrProcessResponse(BaseModel):
    job_id: str
    status: str
    message: str


class ReceiptItem(BaseModel):
    """A line item, shaped to survive SERMS' validation.

    SERMS requires ``items.*.quantity`` to be an integer of at least 1 and rejects
    the ENTIRE callback otherwise. A weighed item (0.5 kg) or an OCR'd zero would
    therefore lose a correctly extracted receipt over one line, so quantity is
    clamped here instead.
    """

    name: str = Field(max_length=255)
    quantity: int = Field(ge=1)
    price: float = Field(ge=0)

    @field_validator("quantity", mode="before")
    @classmethod
    def clamp_quantity(cls, value: Any) -> int:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 1
        return max(1, round(number))

    @field_validator("name", mode="before")
    @classmethod
    def truncate_name(cls, value: Any) -> str:
        return str(value or "item")[:255]


class OcrCallbackPayload(BaseModel):
    """The outbound contract, mirroring SERMS' ``OcrCallbackRequest`` rules.

    Enforced before sending. Previously the pipeline posted a raw dict straight
    from the language model, so there was no type boundary at all between a 1.5B
    model and the financial system.
    """

    receipt_id: int
    vendor_name: Optional[str] = Field(default=None, max_length=255)
    transaction_date: Optional[str] = None
    total_amount: Optional[float] = Field(default=None, ge=0)
    vat_amount: Optional[float] = Field(default=None, ge=0)
    tin: Optional[str] = Field(default=None, max_length=255)
    invoice_number: Optional[str] = Field(default=None, max_length=255)
    vat_classification: Optional[str] = None
    currency: Optional[str] = Field(default=None, max_length=3)
    expense_category: Optional[str] = None
    ocr_confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    items: List[ReceiptItem] = []
    status: Optional[str] = None
    error: Optional[str] = None

    @field_validator("vat_classification")
    @classmethod
    def restrict_vat_classification(cls, value: Optional[str]) -> Optional[str]:
        """SERMS validates this with ``in:vat,non-vat``.

        Any third value - including a semantically correct 'not_applicable' for a
        foreign receipt - would 422 the whole callback, so anything else becomes
        null. Non-PH tax semantics are carried internally by ``tax_type`` instead.
        """
        return value if value in ("vat", "non-vat") else None

    @field_validator("currency")
    @classmethod
    def restrict_currency(cls, value: Optional[str]) -> Optional[str]:
        """Coerce unrecognised currency codes to None.

        A code outside SUPPORTED_CURRENCIES would be stored as an uncontrolled
        string in SERMS and silently ignored rather than 422-ing the whole callback.
        OCR detection already limits output to the known set, so this is a safety
        net for edge cases rather than an expected path.
        """
        if value is None:
            return None
        upper = value.strip().upper()
        return upper if upper in SUPPORTED_CURRENCIES else None

    @field_validator("expense_category")
    @classmethod
    def restrict_category(cls, value: Optional[str]) -> Optional[str]:
        return coerce_category(value) if value else None

    @field_validator("vendor_name", "tin", "invoice_number", mode="before")
    @classmethod
    def blank_to_none(cls, value: Any) -> Any:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


def build_callback_payload(
    receipt_id: int,
    fields: dict[str, Any],
    confidence: float,
    status: str = "completed",
    items: list[dict[str, Any]] | None = None,
) -> OcrCallbackPayload:
    """Map the internal field set onto the SERMS contract.

    Only the keys SERMS declares are sent. Laravel's ``validated()`` silently drops
    undeclared keys, so the wider internal model (net_sales, service_charge,
    country, currency, review reasons) is persisted on our side rather than
    transmitted and discarded.
    """
    return OcrCallbackPayload(
        receipt_id=receipt_id,
        vendor_name=fields.get("vendor_name"),
        transaction_date=fields.get("transaction_date"),
        total_amount=fields.get("total_amount"),
        vat_amount=fields.get("tax_amount"),
        tin=fields.get("vendor_tax_id"),
        invoice_number=fields.get("invoice_number"),
        vat_classification=fields.get("vat_classification"),
        currency=fields.get("currency"),
        expense_category=fields.get("expense_category"),
        ocr_confidence_score=max(0.0, min(float(confidence), 1.0)),
        items=_valid_items(items or []),
        status=status,
    )


def _valid_items(raw_items: list[dict[str, Any]]) -> list[ReceiptItem]:
    """Keep the line items that validate; drop the ones that do not.

    Losing an unparseable line beats losing the receipt to a 422.
    """
    valid: list[ReceiptItem] = []
    for entry in raw_items:
        try:
            valid.append(ReceiptItem(**entry))
        except Exception:
            continue
    return valid


__all__ = [
    "OcrProcessRequest", "OcrProcessResponse", "ReceiptItem",
    "OcrCallbackPayload", "build_callback_payload", "EXPENSE_CATEGORIES",
    "SUPPORTED_CURRENCIES",
]
