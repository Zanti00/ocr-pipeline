"""Outbound callback contract.

Mirrors SERMS' OcrCallbackRequest rules. Each case here is a 422 that would
otherwise reject an entire correctly-extracted receipt.
"""

import pytest
from pydantic import ValidationError

from app.api.schemas.ocr import (
    OcrCallbackPayload, ReceiptItem, build_callback_payload,
)

FIELDS = {
    "vendor_name": "Yamachan Japanese Restaurant",
    "transaction_date": "2026-03-07",
    "total_amount": 1422.61,
    "tax_amount": 139.93,
    "net_sales": 1166.07,
    "service_charge": 116.61,
    "vendor_tax_id": "303-124-202-00000",
    "vat_classification": "vat",
    "expense_category": "Meals",
    "country": "PH",
    "currency": "PHP",
}


class TestFieldMapping:
    def test_internal_names_map_to_contract_names(self):
        payload = build_callback_payload(42, FIELDS, confidence=0.91)
        assert payload.tin == "303-124-202-00000"      # from vendor_tax_id
        assert payload.vat_amount == 139.93            # from tax_amount
        assert payload.total_amount == 1422.61

    def test_internal_only_fields_are_not_transmitted(self):
        # SERMS drops undeclared keys silently, so they are persisted our side.
        payload = build_callback_payload(42, FIELDS, confidence=0.91)
        transmitted = payload.model_dump()
        # net_sales and service_charge are internal breakdown fields, not callback fields.
        for internal in ("net_sales", "service_charge", "country"):
            assert internal not in transmitted

    def test_currency_is_now_transmitted(self):
        # currency is in CALLBACK_FIELDS since the locale-propagation change.
        payload = build_callback_payload(42, FIELDS, confidence=0.91)
        assert payload.currency == "PHP"


class TestVatClassification:
    def test_accepted_values_pass(self):
        assert OcrCallbackPayload(receipt_id=1, vat_classification="vat"
                                  ).vat_classification == "vat"
        assert OcrCallbackPayload(receipt_id=1, vat_classification="non-vat"
                                  ).vat_classification == "non-vat"

    def test_third_values_become_null_rather_than_422(self):
        # 'not_applicable' is semantically right for a US receipt but SERMS
        # validates in:vat,non-vat and would reject the whole callback.
        payload = OcrCallbackPayload(receipt_id=1, vat_classification="not_applicable")
        assert payload.vat_classification is None


class TestCurrency:
    def test_known_codes_pass_through(self):
        for code in ("PHP", "USD", "JPY", "EUR", "GBP", "SGD"):
            payload = OcrCallbackPayload(receipt_id=1, currency=code)
            assert payload.currency == code

    def test_unknown_code_becomes_none_rather_than_422(self):
        # An unexpected string must not 422 the whole callback.
        payload = OcrCallbackPayload(receipt_id=1, currency="XYZ")
        assert payload.currency is None

    def test_none_passthrough(self):
        # Locale detection can legitimately return None (unresolved locale).
        payload = OcrCallbackPayload(receipt_id=1, currency=None)
        assert payload.currency is None

    def test_lowercase_code_is_normalised(self):
        # Defensive: normalise lowercase input even though OCR always emits uppercase.
        payload = OcrCallbackPayload(receipt_id=1, currency="usd")
        assert payload.currency == "USD"


class TestItemQuantityClamping:
    def test_fractional_quantity_is_rounded_up_to_one(self):
        assert ReceiptItem(name="Rice 0.5kg", quantity=0.5, price=25.0).quantity == 1

    def test_zero_quantity_is_clamped(self):
        assert ReceiptItem(name="x", quantity=0, price=1.0).quantity == 1

    def test_unparseable_items_are_dropped_not_fatal(self):
        payload = build_callback_payload(
            7, FIELDS, confidence=0.9,
            items=[
                {"name": "Chickenjoy", "quantity": 1, "price": 29.99},
                {"name": "broken", "price": "not a number"},
            ],
        )
        assert len(payload.items) == 1
        assert payload.items[0].name == "Chickenjoy"


class TestConfidenceBounds:
    def test_score_is_clamped_into_range(self):
        # SERMS requires 0..1; its own ExpenseService compares against 80 on the
        # same column, so sending a 0-100 value here would be doubly wrong.
        assert build_callback_payload(1, FIELDS, confidence=1.4).ocr_confidence_score == 1.0
        assert build_callback_payload(1, FIELDS, confidence=-2).ocr_confidence_score == 0.0

    def test_out_of_range_is_rejected_when_constructed_directly(self):
        with pytest.raises(ValidationError):
            OcrCallbackPayload(receipt_id=1, ocr_confidence_score=85.0)


class TestAbstention:
    def test_null_total_is_contract_valid(self):
        # The abstention design depends on this: total_amount is nullable in
        # OcrCallbackRequest, so abstaining needs no contract change.
        payload = build_callback_payload(
            9, {"total_amount": None, "vendor_name": None}, confidence=0.45
        )
        assert payload.total_amount is None
        assert payload.ocr_confidence_score == 0.45
