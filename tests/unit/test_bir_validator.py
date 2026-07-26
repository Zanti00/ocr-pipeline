from app.core.bir_validator import classify_vat, validate_tin


class TestValidateTin:
    def test_three_digit_branch_code(self):
        assert validate_tin("123-456-789-000") is True
        assert validate_tin("123-456-789-000V") is True

    def test_five_digit_branch_code_is_valid(self):
        # Receipt 3's real vendor TIN. The original regex required exactly three
        # digits in the last group and rejected this outright.
        assert validate_tin("303-124-202-00000") is True

    def test_ocr_mangled_separators_still_validate(self):
        # Receipt 6: Tesseract splits '471' into '47 1'.
        assert validate_tin("000-774-47 1-025") is True

    def test_wrong_digit_counts_are_invalid(self):
        assert validate_tin("123456789") is False
        assert validate_tin("123-456-789") is False
        assert validate_tin("1234-5678-9012-3456-7890") is False

    def test_empty_and_non_numeric_are_invalid(self):
        assert validate_tin(None) is False
        assert validate_tin("") is False
        assert validate_tin("not-a-tin-at-all") is False


class TestClassifyVat:
    def test_bare_keywords(self):
        assert classify_vat("This is a VAT receipt") == "vat"
        assert classify_vat("This is a NON-VAT receipt") == "non-vat"
        assert classify_vat("This is VAT EXEMPT") == "non-vat"
        assert classify_vat("No tax info here") == "non-vat"

    def test_registration_marker_wins_over_column_headings(self):
        # A non-VAT sales invoice still prints 'VATable Sales' as a column heading,
        # which the previous keyword-only check read as VAT-registered.
        text = "Non VAT Reg. TIN: 403-205-152-000\nVATable Sales\nZero-Rated Sales"
        assert classify_vat(text) == "non-vat"

    def test_vat_registered_header(self):
        assert classify_vat("VAT Reg. TIN 303-124-202-00000") == "vat"

    def test_input_tax_disclaimer_implies_non_vat(self):
        text = "SALES INVOICE\nTHIS DOCUMENT IS NOT VALID FOR CLAIMING INPUT TAXES"
        assert classify_vat(text) == "non-vat"
