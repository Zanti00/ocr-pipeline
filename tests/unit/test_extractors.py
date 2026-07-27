"""Deterministic extraction rules.

These cases are drawn from real failures observed in the sample corpus rather
than invented, so a regression here means a receipt that used to work has broken.
"""

from app.core.extractors import (
    find_dates, find_invoice_number, find_tax_ids, format_tax_id,
    select_vendor_tax_id,
)


class TestTaxIdFormatting:
    def test_five_digit_branch_code_is_preserved(self):
        # Receipt 3's vendor TIN, which the original 3-3-3-3 regex rejected.
        assert format_tax_id("30312420200000") == "303-124-202-00000"

    def test_three_digit_branch_code(self):
        assert format_tax_id("000774471025") == "000-774-471-025"


class TestTaxIdSelection:
    def test_vendor_wins_over_customer(self):
        # Receipt 3 carries both; picking the customer's is a compliance failure.
        lines = [
            "YAMACHAN JAPANESE RESTAURANT",
            "VAT Reg. TIN 303-124-202-00000 * Mark Anthony R. Alvaro - Prop.",
            "Registered Name : Scientific Biotech Specialties, Inc",
            "TIN : 201-841-917-000",
        ]
        selected = select_vendor_tax_id(find_tax_ids(lines))
        assert selected is not None
        assert selected.formatted == "303-124-202-00000"
        assert selected.role == "vendor"

    def test_ocr_split_separator_is_recovered(self):
        # Receipt 6: Tesseract splits '471' as '47 1'.
        lines = ["VAT REG. TIN 000-774-47 1-025"]
        selected = select_vendor_tax_id(find_tax_ids(lines))
        assert selected is not None
        assert selected.formatted == "000-774-471-025"

    def test_card_reference_is_not_a_tax_id(self):
        # Receipt 1's payment terminal string contains a 12-digit window.
        lines = ["Reference # 62845289260246240685c", "Card number XXXX 4922"]
        assert select_vendor_tax_id(find_tax_ids(lines)) is None

    def test_phone_number_block_is_not_a_tax_id(self):
        # Receipt 2: '223 2890 223 2891' is fourteen digits with single spaces.
        lines = ["Jollibee Yayasan Complex BO6", "223 2890 223 2891"]
        assert select_vendor_tax_id(find_tax_ids(lines)) is None


class TestDateSelection:
    def test_transaction_date_beats_card_terminal_date(self):
        # Receipt 1: the card block sits next to the word 'Date', so positive
        # label matching alone picks the wrong one.
        lines = [
            "5/24/2025, 11:47:32 AM",
            "Invoice number: 340",
            "Card type Debit",
            "Date/t ime 11/20/2019 11:09 AM",
        ]
        best = find_dates(lines)[0]
        assert best.value.isoformat() == "2025-05-24"

    def test_atp_print_date_is_rejected(self):
        lines = ["0500 Date lssued 03/02/20", "Valid until 03/02/25"]
        candidates = [c for c in find_dates(lines) if c.score > 0]
        assert candidates == []

    def test_month_name_format(self):
        best = find_dates(["Date__ Aug. 9, 2012"])[0]
        assert best.value.isoformat() == "2012-08-09"

    def test_future_dates_are_rejected(self):
        assert find_dates(["Date: 01/01/2099"]) == []


class TestInvoiceNumber:
    def test_labelled_invoice_number(self):
        number, _ = find_invoice_number(["Invoice number: 340"])
        assert number == "340"

    def test_check_number_on_pos_receipt(self):
        number, _ = find_invoice_number(["Chk No:20052"])
        assert number == "20052"

    def test_printer_accreditation_number_is_not_a_receipt_number(self):
        # Receipt 8: 'No. 038MP20140000000045' would otherwise yield '038'.
        number, _ = find_invoice_number(
            ["DAU PRINTERS Printers Accreditation No. 038MP20140000000045"]
        )
        assert number is None
