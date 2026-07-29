"""Unit tests for country and currency detection."""

from app.core.locale import detect_locale


def test_us_receipt_with_city_state_and_dollar_symbols():
    text = (
        "THE FRESH FARMER\n"
        "1468 Tanglewood Road\n"
        "Memphis, MS\n"
        "5-29-2018\n"
        "Eggs/Chz $10.98\n"
        "Coffee $3.45\n"
        "AMT $18.92\n"
        "SUB-TOTAL $18.92\n"
        "TAX $1.23\n"
        "BALANCE $20.15"
    )
    guess = detect_locale(text)
    assert guess.country == "US"
    assert guess.currency == "USD"
    assert guess.resolved is True
    assert guess.score >= 3.0


def test_ph_receipt_with_bir_marking_and_peso_symbol():
    text = (
        "JOLLIBEE MAKATI\n"
        "VAT REG TIN: 000-123-456-000\n"
        "OFFICIAL RECEIPT\n"
        "TOTAL SALES ₱150.00"
    )
    guess = detect_locale(text)
    assert guess.country == "PH"
    assert guess.currency == "PHP"
    assert guess.resolved is True


def test_ph_receipt_with_bir_marking_where_ocr_mangled_peso_to_dollar():
    text = (
        "JOLLIBEE MAKATI\n"
        "VAT REG TIN: 000-123-456-000\n"
        "OFFICIAL RECEIPT\n"
        "TOTAL SALES $150.00\n"
        "CASH $200.00"
    )
    guess = detect_locale(text)
    assert guess.country == "PH"
    assert guess.currency == "PHP"
    assert guess.resolved is True


def test_singapore_receipt_with_locality_and_dollar_symbol():
    text = (
        "SINGAPORE TRADING PTE LTD\n"
        "ORCHARD ROAD, SINGAPORE\n"
        "TOTAL $45.80"
    )
    guess = detect_locale(text)
    assert guess.country == "SG"
    assert guess.currency == "SGD"
    assert guess.resolved is True


def test_multiple_dollar_symbols_defaults_to_usd_prior():
    text = (
        "COFFEE SHOP\n"
        "ITEM 1 $5.00\n"
        "ITEM 2 $3.50\n"
        "TOTAL $8.50"
    )
    guess = detect_locale(text)
    assert guess.country == "US"
    assert guess.currency == "USD"
    assert guess.resolved is True


def test_single_stray_dollar_symbol_returns_unresolved():
    text = "RANDOM TEXT $5"
    guess = detect_locale(text)
    assert guess.country is None
    assert guess.currency is None
    assert guess.resolved is False


def test_unambiguous_currency_symbols():
    assert detect_locale("TOTAL ₱100.00").currency == "PHP"
    assert detect_locale("TOTAL RM50.00").currency == "MYR"
    assert detect_locale("TOTAL £15.50").currency == "GBP"
    assert detect_locale("TOTAL €20.00").currency == "EUR"
