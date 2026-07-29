"""Locale-aware numeric parsing.

Shared by the extraction pipeline and the evaluation harness so both interpret a
token identically - if they disagreed, the measured accuracy would be measuring
the harness rather than the pipeline.
"""

from __future__ import annotations

import re

# Locales using '.' as the decimal separator. Outside these, a trailing ',dd' is
# a genuine decimal rather than an OCR corruption of '.dd'.
DOT_DECIMAL_COUNTRIES = {"PH", "US", "BN", "MY", "SG", "GB", "AU", "JP", "HK", "TH"}


def digits_only(value: object) -> str:
    return re.sub(r"\D", "", str(value or ""))


def normalize_money(raw: object, country: str | None = None) -> float | None:
    """Parse an OCR'd money token into a float.

    Handles the corruption seen on receipt 1, where Tesseract renders ``$32.50``
    as ``$32,50``. In a dot-decimal locale a trailing ``,dd`` is treated as a
    decimal point; elsewhere it is respected as a genuine decimal comma.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return round(float(raw), 2)

    text = re.sub(r"[^\d.,\-]", "", str(raw)).strip()
    if not text or not re.search(r"\d", text):
        return None

    # Handle trailing minus signs (e.g. '2.98-' or '0.50-')
    if text.endswith("-") and not text.startswith("-"):
        text = "-" + text[:-1]

    dot_decimal = country is None or country.upper() in DOT_DECIMAL_COUNTRIES

    if re.fullmatch(r"-?\d{1,3}(\.\d{3})*,\d{1,2}", text) and not dot_decimal:
        text = text.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"-?\d+,\d{2}", text) and dot_decimal:
        # OCR corruption of a decimal point in a dot-decimal locale.
        text = text.replace(",", ".")
    else:
        text = text.replace(",", "")

    try:
        return round(float(text), 2)
    except ValueError:
        return None
