"""Locale-aware money parsing.

The '32,50' case is the one that matters: Tesseract corrupts receipt 1's decimal
point into a comma, and reading it as 3250 instead of 32.50 would inflate a claim
one hundred fold. The same characters are a legitimate decimal comma in a
comma-decimal locale, which is why parsing depends on detected country.
"""

from app.core.numbers import normalize_money


def test_ocr_comma_is_a_decimal_point_in_dot_decimal_locales():
    assert normalize_money("$32,50", "US") == 32.50
    assert normalize_money("29 ,99", "US") == 29.99
    assert normalize_money("P1,306.00", "PH") == 1306.00


def test_comma_is_respected_as_decimal_in_comma_decimal_locales():
    assert normalize_money("32,50", "DE") == 32.50
    assert normalize_money("1.306,07", "DE") == 1306.07


def test_thousands_separators_are_stripped():
    assert normalize_money("1,422.61", "PH") == 1422.61
    assert normalize_money("12,345.67", "US") == 12345.67


def test_unparseable_tokens_return_none():
    assert normalize_money(None) is None
    assert normalize_money("") is None
    assert normalize_money("abc") is None
    assert normalize_money("---") is None


def test_numeric_inputs_pass_through_rounded():
    assert normalize_money(1306) == 1306.00
    assert normalize_money(139.933) == 139.93


def test_trailing_minus_sign_parsed_as_negative_float():
    assert normalize_money("2.98-", "US") == -2.98
    assert normalize_money("0.50-", "US") == -0.50
    assert normalize_money("$1.71-", "US") == -1.71
