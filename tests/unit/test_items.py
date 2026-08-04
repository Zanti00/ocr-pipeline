"""Line-item extraction.

The receipt 1 case drives most of these: '1 Family bundle--10pcs $29.99' followed
by an unpriced 'Chickenjoy bucket + 3 large sides' is ONE purchase and must produce
ONE row, with quantity 1 rather than the 10 from '10pcs'.
"""

from __future__ import annotations

import pytest

from app.core.items import (
    UNREADABLE_NAME, ExtractedItem, looks_like_store_code, parse_items,
    reconcile_items,
)
from app.core.ocr_engine import Word


def _line(text: str, top: int, left: int = 0) -> list[Word]:
    """Build a synthetic OCR line, one Word per whitespace token."""
    words: list[Word] = []
    x = left
    for index, token in enumerate(text.split()):
        words.append(Word(text=token, confidence=0.9, left=x, top=top,
                          width=max(8 * len(token), 8), height=12,
                          block=0, par=0, line=index + top))
        x += 8 * len(token) + 8
    return words


def _lines(*rows: str) -> list[list[Word]]:
    return [_line(row, top=index * 20) for index, row in enumerate(rows)]


class TestReceiptOneCase:
    def test_description_line_folds_into_one_row(self):
        items = parse_items(_lines(
            "Details",
            "1 Family bundle--10pcs $29 ,99",
            "Chickenjoy bucket + 3 large sides",
            "Subtotal $29 ,99",
        ))
        assert len(items) == 1
        assert items[0].quantity == 1
        assert items[0].price == 29.99
        assert "Family bundle--10pcs" in items[0].full_name
        assert "Chickenjoy bucket" in items[0].full_name

    def test_pieces_in_the_name_are_not_a_quantity(self):
        # 10 x 29.99 would overstate the line tenfold.
        items = parse_items(_lines("Details", "1 Family bundle--10pcs $29 ,99"))
        assert items[0].quantity == 1

    def test_price_is_not_left_inside_the_name(self):
        # The reassembled token '$29,99' never matches the raw text '$29 ,99', so
        # the name must be rebuilt from the remaining words.
        items = parse_items(_lines("Details", "1 Family bundle--10pcs $29 ,99"))
        assert "29" not in items[0].full_name


class TestRegionBoundaries:
    def test_summary_lines_are_not_items(self):
        items = parse_items(_lines(
            "Details",
            "2 Iced Tea 150.00",
            "Subtotal 150.00",
            "Tax 18.00",
            "TOTAL AMOUNT DUE 168.00",
        ))
        assert len(items) == 1
        assert items[0].name.strip() == "Iced Tea"

    def test_payment_footer_is_ignored(self):
        items = parse_items(_lines(
            "Details", "1 Bottled Water 35.00", "Subtotal 35.00",
            "Payment 100.00", "Change Due 65.00",
        ))
        assert len(items) == 1

    def test_header_amounts_are_not_items(self):
        items = parse_items(_lines(
            "YAMACHAN JAPANESE RESTAURANT",
            "VAT Reg. TIN 303-124-202-00000",
            "Details",
            "1 Beef Gyudon 265.00",
            "Total Sales 265.00",
        ))
        assert len(items) == 1

    def test_region_without_a_marker_still_parses(self):
        # Thermal slips separate the region with asterisks that OCR turns to noise.
        items = parse_items(_lines(
            "Jollibee Yayasan Complex",
            "AERIS IER EOE Ea Ooo",
            "1 Chicken Teriyaki Set 285.00",
            "Subtotal 285.00",
        ))
        assert len(items) == 1


class TestQuantity:
    def test_leading_quantity(self):
        items = parse_items(_lines("Details", "3 California Maki 540.00"))
        assert items[0].quantity == 3

    def test_quantity_column_before_the_amount(self):
        # BIR form layout: 'name    qty    amount'.
        items = parse_items(_lines("Description Amount", "Bond Paper A4 Ream 2 490.00"))
        assert items[0].quantity == 2
        assert items[0].price == 490.00

    def test_missing_quantity_defaults_to_one(self):
        items = parse_items(_lines("Details", "Salmon Sashimi 420.00"))
        assert items[0].quantity == 1


class TestUnreadableNames:
    def test_noise_name_becomes_a_placeholder(self):
        # Receipt 2: every variant reads the price correctly and the name as noise.
        items = parse_items(_lines("Details", "1 ci cr } UM SDR 3.70"))
        assert len(items) == 1
        assert items[0].price == 3.70
        assert items[0].full_name == UNREADABLE_NAME
        assert items[0].name_is_readable is False

    def test_a_bare_amount_is_not_an_item(self):
        assert parse_items(_lines("Details", "29.99")) == []


class TestReconciliation:
    def _items(self, *pairs: tuple[int, float]) -> list[ExtractedItem]:
        return [
            ExtractedItem(name=f"item {index}", quantity=qty, price=price)
            for index, (qty, price) in enumerate(pairs)
        ]

    def test_line_total_basis(self):
        scan = reconcile_items(self._items((1, 29.99)), {"net_sales": 29.99})
        assert scan.reconciled
        assert scan.price_basis == "line_total"
        assert scan.payload()[0]["price"] == 29.99

    def test_unit_price_basis_is_detected(self):
        # Printed figure is the unit price: 2 x 75.00 = 150.00.
        scan = reconcile_items(self._items((2, 75.00)), {"net_sales": 150.00})
        assert scan.reconciled
        assert scan.price_basis == "unit_price"
        # The transmitted price is always the line total, whichever convention won.
        assert scan.payload()[0]["price"] == 150.00

    def test_unreconciled_items_are_still_transmitted_in_payload(self):
        # Items are transmitted so user input fields are populated, while reconciled remains False.
        scan = reconcile_items(self._items((1, 10.00), (1, 20.00)), {"net_sales": 99.00})
        assert scan.reconciled is False
        assert len(scan.payload()) == 2
        assert scan.payload()[0]["price"] == 10.00

    def test_vat_inclusive_total_is_an_acceptable_target(self):
        scan = reconcile_items(
            self._items((1, 285.00), (1, 180.00)),
            {"net_sales": 415.18, "total_sales": 465.00},
        )
        assert scan.reconciled
        assert scan.target == 465.00

    def test_no_subtotal_means_no_reconciliation(self):
        scan = reconcile_items(self._items((1, 29.99)), {"net_sales": None})
        assert scan.reconciled is False
        assert "no subtotal" in " ".join(scan.notes)

    def test_tolerance_scales_with_row_count(self):
        items = self._items(*[(1, 10.00)] * 8)
        scan = reconcile_items(items, {"net_sales": 80.05})
        assert scan.reconciled

    @pytest.mark.parametrize("price", [0.0, -5.0])
    def test_non_positive_prices_never_ship(self, price):
        # SERMS' PRS path validates item price as gt:0.
        scan = reconcile_items(self._items((1, 12.00)), {"net_sales": 12.00})
        scan.items.append(ExtractedItem(name="free", quantity=1, price=price))
        assert all(row["price"] > 0 for row in scan.payload())


class TestGroceryPriceFirstLayout:
    """Grocery-POS receipts print price first, then qty x price, then the
    barcode + store-code name line ('64.00 V' / '1 x 64.00' / '4800016522533 DRD PEV-CUT EP025G').
    Each physical item must produce ONE row named from the barcode line.
    """

    def test_price_first_three_line_item(self):
        items = parse_items(_lines(
            "64.00   V",
            "1  ×  64.00",
            "4800016522533   DRD   PEV-CUT   EP025G",
            "Subtotal 64.00",
        ))
        assert len(items) == 1
        assert items[0].price == 64.00
        assert items[0].quantity == 1
        assert "DRD PEV-CUT EP025G" in items[0].full_name
        assert items[0].name != "V"

    def test_quantity_line_merges_into_pending_item(self):
        items = parse_items(_lines(
            "44.90   V",
            "3   x   44.00",
            "0998066630917   WING5   SFF2XW5H100G",
            "Subtotal 134.70",
        ))
        assert len(items) == 1
        assert items[0].quantity == 3

    def test_multiple_price_first_items(self):
        items = parse_items(_lines(
            "17.15   V",
            "1  ×  17.15",
            "4000092552373   CA05SINI   BVANCA4JG",
            "122.00   V",
            "1   x   122.00",
            "4800016652035   JNJ   MRCHPSN / CHS24G",
            "Subtotal 139.15",
        ))
        assert len(items) == 2
        assert "CA05SINI BVANCA4JG" in items[0].full_name
        assert "JNJ MRCHPSN / CHS24G" in items[1].full_name

    def test_plain_money_line_without_name_stays_pending_until_summary(self):
        # No name line follows: the row is still emitted rather than dropped.
        items = parse_items(_lines(
            "64.00   V",
            "1  ×  64.00",
            "Subtotal 64.00",
        ))
        assert len(items) == 1
        assert items[0].price == 64.00


class TestLooksLikeStoreCode:
    def test_code_style_names_are_detected(self):
        assert looks_like_store_code("DRD PEV-CUT EP025G")
        assert looks_like_store_code("CA05SINI BVANCA4JG")
        assert looks_like_store_code("Z0NA04 CA15 500ML")

    def test_product_names_are_not_codes(self):
        assert not looks_like_store_code("SALAD")
        assert not looks_like_store_code("Chickenjoy bucket")
        assert not looks_like_store_code("SMART DW P-LEMON")
