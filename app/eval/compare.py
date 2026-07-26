"""Field comparators, outcome classification, and the grounding check.

Two independent questions are answered here, and keeping them separate is the
core of the anti-hallucination measurement:

1. ACCURACY   - does the extracted value match ground truth?
2. GROUNDING  - does the extracted value actually appear in the OCR text?

A value can be wrong but grounded (a misread) or wrong and ungrounded (a
fabrication). Only the second one is a hallucination, and only the second one
is unfixable by better OCR.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from rapidfuzz import fuzz

from app.core.grounding import is_grounded as core_is_grounded
from app.core.grounding import normalize_text, parse_date_value as parse_date
from app.core.numbers import DOT_DECIMAL_COUNTRIES, digits_only, normalize_money
from app.eval.groundtruth import FieldKind, FieldSpec

__all__ = [
    "Outcome", "classify", "values_match", "is_grounded", "normalize_text",
    "normalize_money", "digits_only", "parse_date", "DOT_DECIMAL_COUNTRIES",
]

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

FUZZY_ACCURACY_THRESHOLD = 90.0   # vendor_name counts as correct at/above this
FUZZY_GROUNDING_THRESHOLD = 75.0  # vendor_name counts as grounded at/above this
MONEY_TOLERANCE = 0.01
RATE_TOLERANCE = 0.005


class Outcome(str, Enum):
    CORRECT = "correct"
    CORRECT_NULL = "correct_null"        # correctly reported nothing
    MISSED = "missed"                    # returned null, truth had a value
    WRONG = "wrong"                      # returned a value, truth differs
    FALSE_POSITIVE = "false_positive"    # returned a value, truth is null
    CONFUSED = "confused"                # matched a known trap value
    SKIPPED = "skipped"                  # ground truth unverifiable

    @property
    def is_pass(self) -> bool:
        return self in (Outcome.CORRECT, Outcome.CORRECT_NULL)

    @property
    def is_populated(self) -> bool:
        """Did the system emit a non-null value for this field?"""
        return self in (
            Outcome.CORRECT, Outcome.WRONG, Outcome.FALSE_POSITIVE, Outcome.CONFUSED
        )


# --------------------------------------------------------------------------
# normalisation helpers
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------

def values_match(spec: FieldSpec, expected: Any, actual: Any, country: str | None) -> bool:
    if spec.kind is FieldKind.MONEY:
        a, b = normalize_money(expected, country), normalize_money(actual, country)
        return a is not None and b is not None and abs(a - b) <= MONEY_TOLERANCE

    if spec.kind is FieldKind.RATE:
        try:
            return abs(float(expected) - float(actual)) <= RATE_TOLERANCE
        except (TypeError, ValueError):
            return False

    if spec.kind is FieldKind.TAX_ID:
        # Digit comparison, so '000-774-47 1-025' still matches '000-774-471-025'.
        return bool(digits_only(expected)) and digits_only(expected) == digits_only(actual)

    if spec.kind is FieldKind.DATE:
        a, b = parse_date(expected), parse_date(actual)
        return a is not None and a == b

    if spec.kind is FieldKind.FUZZY_TEXT:
        a, b = normalize_text(expected), normalize_text(actual)
        if not a or not b:
            return False
        return fuzz.token_set_ratio(a, b) >= FUZZY_ACCURACY_THRESHOLD

    return normalize_text(expected) == normalize_text(actual)


def classify(
    spec: FieldSpec,
    expected: Any,
    actual: Any,
    country: str | None = None,
    trap_values: list[str] | None = None,
    accepted_alternatives: list[str] | None = None,
) -> Outcome:
    """Classify one extracted field against ground truth."""
    expected_missing = expected is None
    actual_missing = actual is None or (isinstance(actual, str) and not actual.strip())

    if expected_missing and actual_missing:
        return Outcome.CORRECT_NULL
    if expected_missing and not actual_missing:
        for trap in trap_values or []:
            if values_match(spec, trap, actual, country):
                return Outcome.CONFUSED
        return Outcome.FALSE_POSITIVE
    if actual_missing:
        return Outcome.MISSED
    if values_match(spec, expected, actual, country):
        return Outcome.CORRECT
    for alternative in accepted_alternatives or []:
        if values_match(spec, alternative, actual, country):
            return Outcome.CORRECT

    for trap in trap_values or []:
        if values_match(spec, trap, actual, country):
            return Outcome.CONFUSED
    return Outcome.WRONG


# --------------------------------------------------------------------------
# grounding
# --------------------------------------------------------------------------

def is_grounded(spec: FieldSpec, value: Any, ocr_text: str) -> bool:
    """Is there textual evidence in the OCR output for this value?

    Delegates to the production implementation. It previously had its own copy,
    and the two drifted: the harness treated a computed ``tax_rate`` as an
    ungrounded fabrication while the pipeline correctly exempted it, inflating the
    measured fabrication rate. Measuring the pipeline requires using the
    pipeline's own rules.
    """
    return core_is_grounded(spec.name, value, ocr_text)
