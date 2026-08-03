"""Fast-path sufficiency: the single-pass OCR gate.

A fast-path result ships only when the money reconciles AND a total is present.
Anything less must escalate to the full OCR pool - that is what keeps accuracy
on hard receipts identical to the pre-fast-path pipeline.
"""

from app.core.extraction import Extraction, fast_path_sufficient
from app.core.reconcile import MoneyResult


def _extraction_with(money: MoneyResult | None, total: float | None) -> Extraction:
    extraction = Extraction()
    extraction.money = money
    if total is not None:
        extraction.fields["total_amount"] = total
    return extraction


def test_reconciled_with_total_ships():
    money = MoneyResult(reconciled=True)
    assert fast_path_sufficient(_extraction_with(money, 32.50)) is True


def test_reconciled_without_total_escalates():
    money = MoneyResult(reconciled=True)
    assert fast_path_sufficient(_extraction_with(money, None)) is False


def test_unreconciled_escalates_even_with_total():
    money = MoneyResult(reconciled=False)
    assert fast_path_sufficient(_extraction_with(money, 32.50)) is False


def test_no_money_escalates():
    assert fast_path_sufficient(_extraction_with(None, 32.50)) is False


def test_empty_extraction_escalates():
    assert fast_path_sufficient(_extraction_with(None, None)) is False
