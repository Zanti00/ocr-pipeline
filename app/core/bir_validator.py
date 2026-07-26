"""BIR validation helpers.

Both functions here were defective and are now thin wrappers over the shared
extraction logic:

* ``validate_tin`` matched ``^\\d{3}-\\d{3}-\\d{3}-\\d{3}V?$`` and therefore
  REJECTED five-digit branch codes. Real TINs use both forms, so a valid vendor
  such as ``303-124-202-00000`` failed validation and lost 10% of its confidence
  score. Comparison is now on digits, which also survives OCR mangling the
  separators (``000-774-47 1-025``).
* ``classify_vat`` returned "vat" whenever the word appeared anywhere, so any
  receipt printing a 'VATable Sales' column read as VAT-registered. Registration
  markers are now checked before bare keywords.
"""

from __future__ import annotations

import re

from app.core.numbers import digits_only

VALID_TIN_DIGIT_LENGTHS = (12, 14)


def validate_tin(tin: str | None) -> bool:
    """Is this a structurally valid PH TIN?

    Accepts 9-3 and 9-5 forms (three digit groups plus a three- or five-digit
    branch code), with any separator style and an optional trailing 'V'.
    """
    if not tin:
        return False
    text = str(tin).strip().rstrip("Vv")
    if not re.fullmatch(r"[\d\s\-]+", text):
        return False
    return len(digits_only(text)) in VALID_TIN_DIGIT_LENGTHS


def classify_vat(ocr_text: str, tin: str | None = None) -> str:
    """Classify a Philippine receipt as ``vat`` or ``non-vat``.

    Registration markers take precedence over loose keyword presence, because
    'VATable Sales' and 'VAT-Exempt Sales' are printed column headings on the
    standard BIR form and appear on receipts of both kinds.
    """
    text = (ocr_text or "").upper()

    if "NON-VAT REG" in text or "NON VAT REG" in text:
        return "non-vat"
    if "NOT VALID FOR CLAIMING INPUT TAX" in text:
        return "non-vat"
    if "VAT REG" in text:
        return "vat"

    if "NON-VAT" in text or "NON VAT" in text:
        return "non-vat"
    if "VAT-EXEMPT" in text or "VAT EXEMPT" in text:
        return "non-vat"
    if "VAT" in text:
        return "vat"
    return "non-vat"
