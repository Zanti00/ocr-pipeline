"""Country and currency detection.

This runs BEFORE any amount is parsed, because number formatting is
locale-dependent. Tesseract renders receipt 1's ``$32.50`` as ``$32,50``; in a
dot-decimal locale that is an OCR error for 32.50, while in a comma-decimal
locale it is genuinely thirty-two fifty. Same characters, answers 100x apart.

Detection is evidence-based and refuses to default. An unresolved locale returns
``None`` and drives a manual-review flag, because silently assuming PHP on a
foreign receipt is the same class of error as inventing a total.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

COUNTRY_CURRENCY: dict[str, str] = {
    "PH": "PHP", "US": "USD", "BN": "BND", "MY": "MYR", "SG": "SGD",
    "JP": "JPY", "HK": "HKD", "TH": "THB", "AU": "AUD", "GB": "GBP",
}

# Unambiguous currency symbols only. '$' is deliberately excluded - it is shared
# by USD, SGD, HKD, AUD and BND, so it cannot identify a currency on its own.
#
# Each pattern requires the symbol to sit next to a number. Bare substring
# matching produced nonsense on noisy OCR: 'rm' fires inside 'Printers', and a
# stray '\u20ac' artefact in '/\u20acashier' was enough to label a Philippine
# receipt as euro-denominated.
UNAMBIGUOUS_SYMBOLS: tuple[tuple[str, str], ...] = (
    (r"\u20b1\s*\d", "PHP"),
    (r"\bphp\s*\d", "PHP"),
    (r"\brm\s*\d", "MYR"),
    (r"\u20ac\s*\d{1,3}[.,]\d{2}", "EUR"),
    (r"\u00a3\s*\d{1,3}[.,]\d{2}", "GBP"),
    (r"\u0e3f\s*\d", "THB"),
)

_US_STATES = (
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO "
    "MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC"
).split()

# (regex, country, weight, label)
EVIDENCE: tuple[tuple[str, str, float, str], ...] = (
    (r"vat\s*reg\.?\s*tin", "PH", 4.0, "PH VAT registration line"),
    (r"non\s*-?\s*vat\s*reg", "PH", 4.0, "PH non-VAT registration line"),
    (r"\bbir\b|authority\s+to\s+print|\batp\b", "PH", 4.0, "BIR / ATP marking"),
    (r"philippines|makati|quezon\s+city|manila|pasig|cebu|davao|binondo|rizal",
     "PH", 2.5, "PH locality"),
    (r"sc\s*/\s*pwd|solo\s+parent|senior\s+citizen", "PH", 3.0, "PH statutory discount"),
    (r"vatable\s+sales|zero\s*-?\s*rated\s+sales|vat\s*-?\s*exempt\s+sales",
     "PH", 3.0, "PH VAT breakdown labels"),
    (r"official\s+receipt|sales\s+invoice", "PH", 1.0, "PH document type"),
    (r"\btin\b\s*[:#]?\s*\d{3}", "PH", 2.0, "TIN field"),

    (r"sdn\.?\s*bhd", "MY", 2.0, "Sdn Bhd entity"),
    (r"\bmalaysia\b|kuala\s+lumpur|\bgst\b|\bsst\b", "MY", 3.0, "MY marker"),
    (r"\brm\s*\d", "MY", 3.0, "RM currency"),

    (r"brunei|bandar\s+seri|yayasan", "BN", 3.5, "BN locality"),
    (r"sdn\.?\s*bhd.{0,12}\(\s*b\s*\)|ent\s*rek\s*\(\s*b\s*\)", "BN", 2.5,
     "(B) Brunei entity suffix"),

    (r"\bsales\s+tax\b", "US", 3.0, "US sales tax"),
    (r"\bsingapore\b", "SG", 3.5, "SG locality"),
    (r"\bjapan\b|\u5186", "JP", 3.0, "JP marker"),
    (r"\bhong\s*kong\b", "HK", 3.0, "HK locality"),
    (r"\bthailand\b|bangkok", "TH", 3.0, "TH locality"),
)

_US_ZIP_STATE = re.compile(
    r"\b(" + "|".join(_US_STATES) + r")\s+\d{5}(?:-\d{4})?\b"
)
_US_PHONE = re.compile(r"\(\d{3}\)\s*\d{3}\s*-\s*\d{4}")

MIN_CONFIDENT_SCORE = 3.0


@dataclass
class LocaleGuess:
    country: str | None
    currency: str | None
    score: float
    evidence: list[str] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.country is not None and self.score >= MIN_CONFIDENT_SCORE

    @property
    def certainty(self) -> float:
        """0.0-1.0, for the confidence model."""
        if self.country is None:
            return 0.0
        return min(1.0, self.score / 8.0)


def detect_locale(text: str) -> LocaleGuess:
    lowered = text.casefold()
    scores: dict[str, float] = {}
    evidence: dict[str, list[str]] = {}

    for pattern, country, weight, label in EVIDENCE:
        if re.search(pattern, lowered):
            scores[country] = scores.get(country, 0.0) + weight
            evidence.setdefault(country, []).append(label)

    if _US_ZIP_STATE.search(text):
        scores["US"] = scores.get("US", 0.0) + 4.0
        evidence.setdefault("US", []).append("US state + ZIP")
    if _US_PHONE.search(text):
        scores["US"] = scores.get("US", 0.0) + 2.0
        evidence.setdefault("US", []).append("US phone format")

    currency_symbol = _currency_from_symbol(lowered)

    if not scores:
        # No country evidence. A currency symbol alone can still pin the currency.
        return LocaleGuess(country=None, currency=currency_symbol, score=0.0,
                           evidence=["no country evidence"])

    country = max(scores, key=lambda key: scores[key])
    score = scores[country]

    # Brunei and Malaysia share 'Sdn Bhd'; a BN locality reference outranks it.
    if country == "MY" and scores.get("BN", 0.0) >= 2.5:
        country, score = "BN", scores["BN"]

    confident = score >= MIN_CONFIDENT_SCORE
    # Country evidence outranks a symbol match. Country is established by several
    # corroborating signals; a symbol is one character that OCR may have invented.
    if confident:
        currency = COUNTRY_CURRENCY.get(country) or currency_symbol
    else:
        currency = currency_symbol

    return LocaleGuess(
        country=country if confident else None,
        currency=currency,
        score=score,
        evidence=evidence.get(country, []),
    )


def _currency_from_symbol(lowered: str) -> str | None:
    for pattern, currency in UNAMBIGUOUS_SYMBOLS:
        if re.search(pattern, lowered):
            return currency
    return None
