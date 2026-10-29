"""Unit tests for line-item extraction on long receipts, split-line grocery formatting, and near-reconciliation."""
from app.core.items import ExtractedItem, parse_items, reconcile_items
from app.core.ocr_engine import Word
from app.core.verification import (
    REASON_NEAR_RECONCILED,
    REASON_SEGMENT_GAP,
    verify,
)


def _make_line(words_text: list[str], top: int = 10, line_idx: int = 1) -> list[Word]:
    line: list[Word] = []
    current_x = 10
    for w in words_text:
        word = Word(
            text=w,
            confidence=0.95,
            left=current_x,
            top=top,
            width=len(w) * 8,
            height=15,
            block=1,
            par=1,
            line=line_idx,
        )
        line.append(word)
        current_x += word.width + 5
    return line


def test_parse_items_handles_split_line_quantity_before_price():
    # Format: Line 1: '2 x 14.75'
    #         Line 2: '29.50 V'
    #         Line 3: '4800361415347 NSCFORGTP52'
    lines = [
        _make_line(["2", "x", "14.75"], top=10, line_idx=1),
        _make_line(["29.50", "V"], top=30, line_idx=2),
        _make_line(["4800361415347", "NSCFORGTP52"], top=50, line_idx=3),
    ]

    items = parse_items(lines)
    assert len(items) == 1
    assert items[0].quantity == 2
    assert items[0].price == 29.50
    assert "NSCFORGTP52" in items[0].full_name


def test_parse_items_handles_split_line_quantity_after_price():
    # Format: Line 1: '29.50 V'
    #         Line 2: '2 x 14.75'
    #         Line 3: '4800361415347 NSCFORGTP52'
    lines = [
        _make_line(["29.50", "V"], top=10, line_idx=1),
        _make_line(["2", "x", "14.75"], top=30, line_idx=2),
        _make_line(["4800361415347", "NSCFORGTP52"], top=50, line_idx=3),
    ]

    items = parse_items(lines)
    assert len(items) == 1
    assert items[0].quantity == 2
    assert items[0].price == 29.50
    assert "NSCFORGTP52" in items[0].full_name


def test_parse_items_exceeds_legacy_40_item_cap():
    # Generate 55 distinct grocery item lines
    lines: list[list[Word]] = []
    top = 10
    line_no = 1
    for i in range(55):
        lines.append(_make_line(["1", "x", f"{10.00 + i:.2f}", f"{10.00 + i:.2f}", "V"], top=top, line_idx=line_no))
        top += 20
        line_no += 1
        lines.append(_make_line([f"480000000{i:04d}", f"ITEM_NAME_ALPHA_{i}"], top=top, line_idx=line_no))
        top += 20
        line_no += 1

    items = parse_items(lines)
    # Must parse all 55 items, not cap at 40
    assert len(items) == 55


def test_reconcile_items_near_reconciliation_emits_items():
    # 25 items each 100.00 -> sum = 2500.00
    items = [
        ExtractedItem(name=f"Item {i}", quantity=1, price=100.0)
        for i in range(25)
    ]
    # Target printed subtotal is 2550.00 (discrepancy = 50.00 / 2550 = ~1.96% <= 3%)
    targets = {"subtotal": 2550.0}

    scan = reconcile_items(items, targets)
    # Strict reconciliation fails, but near-reconciliation passes
    assert scan.reconciled is False
    assert scan.near_reconciled is True
    # Items must NOT be withheld
    assert len(scan.payload()) == 25
    assert scan.total == 2500.0
    assert scan.target == 2550.0
    assert any("near-reconciled" in note for note in scan.notes)


def test_verification_caps_for_near_reconciled_and_segment_gap():
    fields = {"total_amount": 2550.0, "vendor_name": "Test Supermarket"}
    ocr_text = "Test Supermarket Total 2550.00"

    # Near-reconciled case: caps confidence at 0.79
    v_near = verify(
        fields=fields,
        ocr_text=ocr_text,
        reconciled=False,
        locale_resolved=True,
        near_reconciled=True,
    )
    assert REASON_NEAR_RECONCILED in v_near.reasons
    assert v_near.cap == 0.79
    assert v_near.needs_manual_review is True

    # Segment gap suspected: caps confidence at 0.75
    v_gap = verify(
        fields=fields,
        ocr_text=ocr_text,
        reconciled=True,
        locale_resolved=True,
        segment_gap=True,
    )
    assert REASON_SEGMENT_GAP in v_gap.reasons
    assert v_gap.cap == 0.75
    assert v_gap.needs_manual_review is True
