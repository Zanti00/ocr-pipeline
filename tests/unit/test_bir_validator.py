import pytest
from app.core.bir_validator import validate_tin, classify_vat

def test_validate_tin():
    assert validate_tin("123-456-789-000") is True
    assert validate_tin("123-456-789-000V") is True
    assert validate_tin("123456789") is False
    assert validate_tin(None) is False
    assert validate_tin("") is False

def test_classify_vat():
    assert classify_vat("This is a VAT receipt") == "vat"
    assert classify_vat("This is a NON-VAT receipt") == "non-vat"
    assert classify_vat("This is VAT EXEMPT") == "non-vat"
    assert classify_vat("No tax info here") == "non-vat"
