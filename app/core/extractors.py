"""Deterministic field extraction from OCR text and geometry.

Everything a rule can decide is decided here. The language model is left with
only the two jobs a 1.5B model is actually reliable at - picking a vendor name
from candidate header lines, and classifying an expense category - and even those
are constrained to a closed candidate list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime

from app.core.ocr_engine import Word
from app.core.textquality import looks_like_words

# ---------------------------------------------------------------------------
# tax identifiers
# ---------------------------------------------------------------------------

# A 12- or 14-digit run whose digits may be separated by single hyphens or
# spaces. Tolerating an interior space is essential: receipt 6 comes back as
# '000-774-47 1-025', with the split falling inside the third group.
#
# The lookarounds prevent matching a window inside a longer digit run, which is
# how receipt 1's 20-digit card reference '62845289260246240685C' became a
# vendor TIN.
_DIGIT_RUN_RE = re.compile(r"(?<![\d-])\d(?:[\s-]?\d){11,13}(?![\d-])")

# A bare digit run is only accepted when the line names it as a tax id.
# Otherwise a phone block like '223 2890 223 2891' - fourteen digits separated by
# single spaces - would be read as a tax number.
_TAX_ID_LINE_MARKER = re.compile(r"\btin\b|\bt\.?i\.?n\.?\b|tax\s*(id|no)", re.IGNORECASE)
_MIN_SEPARATORS = 2

VENDOR_TAX_MARKERS = (
    "vat reg", "vat registered", "non vat reg", "non-vat reg", "vat reg. tin",
    "prop.", "proprietor",
)
CUSTOMER_TAX_MARKERS = (
    "registered name", "received from", "sold to", "with tin", "address/tin",
    "customer", "bill to", "business address", "business style",
)


@dataclass
class TaxIdCandidate:
    value: str
    normalized: str
    line_index: int
    line_text: str
    role: str = "unknown"   # vendor | customer | unknown
    score: float = 0.0

    @property
    def formatted(self) -> str:
        return format_tax_id(self.normalized)


def format_tax_id(digits: str) -> str:
    """Regroup a digit run into canonical PH TIN form.

    12 digits -> 3-3-3-3, 14 digits -> 3-3-3-5 (five-digit branch code, as on
    receipt 3's ``303-124-202-00000``, which the previous regex rejected outright).
    """
    if len(digits) == 12:
        return f"{digits[0:3]}-{digits[3:6]}-{digits[6:9]}-{digits[9:12]}"
    if len(digits) == 14:
        return f"{digits[0:3]}-{digits[3:6]}-{digits[6:9]}-{digits[9:14]}"
    return digits


def find_tax_ids(lines: list[str]) -> list[TaxIdCandidate]:
    """Locate tax id candidates and decide which belongs to the vendor.

    Receipt 3 carries two: the vendor's in the printed header and the customer's
    beside 'Registered Name'. Picking the wrong one is a compliance failure, not a
    typo, so role assignment is explicit rather than left to salience.
    """
    candidates: list[TaxIdCandidate] = []
    for index, line_text in enumerate(lines):
        lowered = line_text.casefold()
        labelled_line = bool(_TAX_ID_LINE_MARKER.search(line_text))
        for match in _DIGIT_RUN_RE.finditer(line_text):
            raw = match.group(0)
            digits = re.sub(r"\D", "", raw)
            if len(digits) not in (12, 14):
                continue
            if raw.count("-") < _MIN_SEPARATORS and not labelled_line:
                continue
            candidate = TaxIdCandidate(
                value=match.group(0).strip(),
                normalized=digits,
                line_index=index,
                line_text=line_text,
            )
            _assign_role(candidate, lowered, lines, index)
            candidates.append(candidate)

    # Header position is a strong prior: vendor details are printed above the
    # customer block on every PH form in the corpus.
    total = max(len(lines), 1)
    for candidate in candidates:
        position_bonus = 1.0 - (candidate.line_index / total)
        candidate.score += position_bonus
        if candidate.role == "vendor":
            candidate.score += 3.0
        elif candidate.role == "customer":
            candidate.score -= 3.0

    return sorted(candidates, key=lambda c: -c.score)


def _assign_role(
    candidate: TaxIdCandidate, lowered: str, lines: list[str], index: int
) -> None:
    if any(marker in lowered for marker in VENDOR_TAX_MARKERS):
        candidate.role = "vendor"
        return
    if any(marker in lowered for marker in CUSTOMER_TAX_MARKERS):
        candidate.role = "customer"
        return
    # Look back a couple of lines: 'Registered Name' often sits above the TIN.
    for offset in (1, 2):
        if index - offset < 0:
            break
        previous = lines[index - offset].casefold()
        if any(marker in previous for marker in CUSTOMER_TAX_MARKERS):
            candidate.role = "customer"
            return
        if any(marker in previous for marker in VENDOR_TAX_MARKERS):
            candidate.role = "vendor"
            return


def select_vendor_tax_id(candidates: list[TaxIdCandidate]) -> TaxIdCandidate | None:
    vendor = [c for c in candidates if c.role == "vendor"]
    if vendor:
        return vendor[0]
    unknown = [c for c in candidates if c.role != "customer"]
    return unknown[0] if unknown else None


# ---------------------------------------------------------------------------
# dates
# ---------------------------------------------------------------------------

DATE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b", ("%Y", "%m", "%d")),
    (r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\b", ("%m", "%d", "%Y")),
    (r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2})\b", ("%m", "%d", "%y")),
)

MONTH_NAME_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})\s*,?\s*(\d{4})\b",
    re.IGNORECASE,
)
DAY_MONTH_NAME_RE = re.compile(
    r"\b(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{2,4})\b",
    re.IGNORECASE,
)

_MONTH_NUM = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

DATE_LABELS = ("date", "issued on", "transaction", "invoice date", "or date")

# Dates in these contexts belong to the payment terminal or the print shop, not
# the transaction. Receipt 1 carries a card-terminal '11/20/2019' that sits right
# next to the word 'Date', so positive label matching alone picks the wrong one.
DATE_NEGATIVE_CONTEXT = (
    "card", "reference", "approved", "valid until", "date of atp", "atp",
    "accreditation", "printer", "permit", "ocn", "expiry", "expires",
    # 'ssued' rather than 'issued': Tesseract routinely reads the leading 'I' of
    # 'Issued' as a lowercase 'l', and the ATP print date must never be selected
    # as the transaction date.
    "ssued", "bklts", "booklets",
    # 'Date/time' (OCR'd 'Date/t ime') labels a payment-terminal timestamp. A
    # transaction date is labelled 'Date' or 'Date:', never 'Date/', so the slash
    # is a clean discriminator - and it is what separates receipt 1's real date
    # of 5/24/2025 from the card block's 11/20/2019.
    "date/",
)


@dataclass
class DateCandidate:
    value: date
    raw: str
    line_index: int
    line_text: str
    score: float = 0.0


def find_dates(lines: list[str], today: date | None = None) -> list[DateCandidate]:
    today = today or date.today()
    candidates: list[DateCandidate] = []

    for index, line_text in enumerate(lines):
        for parsed, raw in _parse_line_dates(line_text):
            if not (date(1990, 1, 1) <= parsed <= today):
                continue  # reject impossible / future transaction dates
            candidates.append(
                DateCandidate(value=parsed, raw=raw, line_index=index,
                              line_text=line_text)
            )

    total = max(len(lines), 1)
    for candidate in candidates:
        lowered = candidate.line_text.casefold()
        if any(label in lowered for label in DATE_LABELS):
            candidate.score += 2.0
        if any(bad in lowered for bad in DATE_NEGATIVE_CONTEXT):
            candidate.score -= 5.0
        # Transaction dates appear near the top of a receipt.
        candidate.score += 1.5 * (1.0 - candidate.line_index / total)

    return sorted(candidates, key=lambda c: -c.score)


def _parse_line_dates(line_text: str) -> list[tuple[date, str]]:
    found: list[tuple[date, str]] = []

    for match in MONTH_NAME_RE.finditer(line_text):
        month = _MONTH_NUM[match.group(1)[:3].lower()]
        found.append(
            (_safe_date(int(match.group(3)), month, int(match.group(2))), match.group(0))
        )
    for match in DAY_MONTH_NAME_RE.finditer(line_text):
        month = _MONTH_NUM[match.group(2)[:3].lower()]
        year = int(match.group(3))
        year += 2000 if year < 100 else 0
        found.append((_safe_date(year, month, int(match.group(1))), match.group(0)))

    for pattern, order in DATE_PATTERNS:
        for match in re.finditer(pattern, line_text):
            parts = dict(zip(order, match.groups()))
            year = int(parts.get("%Y") or parts.get("%y") or 0)
            if "%y" in parts and year < 100:
                year += 2000
            month, day = int(parts["%m"]), int(parts["%d"])
            if month > 12 and day <= 12:
                month, day = day, month  # day-first rendering
            found.append((_safe_date(year, month, day), match.group(0)))

    return [(value, raw) for value, raw in found if value is not None]


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return datetime(year, month, day).date()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# invoice / receipt number
# ---------------------------------------------------------------------------

# The trailing lookahead rejects digits that run into letters. A printer's
# accreditation number '038MP20140000000045' sits next to 'No.' and would
# otherwise be captured as '038' - a receipt number that never existed.
INVOICE_LABEL_RE = re.compile(
    r"(?:invoice\s*(?:number|no|#)|receipt\s*(?:no|#)|"
    r"\bor\s*no|\bchk\s*no|\bcheck\s*no|\bno[.:\u00b0\u00ba]|\bn[\u00b0\u00ba\u00a9\u00a2])"
    r"\s*[:.#]?\s*([0-9]{3,8})(?![0-9A-Za-z])",
    re.IGNORECASE,
)

INVOICE_NOISE = (
    "authority to print", "ocn", "permit", "accreditation", "booklets", "bklts",
    "valid until", "atp", "tin", "tel", "phone", "zip",
)


def find_invoice_number(lines: list[str]) -> tuple[str | None, str | None]:
    """Return ``(number, evidence_line)``.

    Skips print-shop identifiers - ATP numbers, OCNs and permit numbers all sit
    next to the word 'No.' and look exactly like receipt numbers.
    """
    for line_text in lines:
        lowered = line_text.casefold()
        if any(noise in lowered for noise in INVOICE_NOISE):
            continue
        match = INVOICE_LABEL_RE.search(line_text)
        if match:
            return match.group(1), line_text
    return None, None


# ---------------------------------------------------------------------------
# vendor name candidates (handed to the LLM as a closed list)
# ---------------------------------------------------------------------------

VENDOR_NAME_EXCLUSIONS = (
    "received from", "registered name", "sold to", "vat reg", "telefax",
    "official receipt", "sales invoice", "authority to print", "booklets",
    "cashier", "business style", "nature of service", "particulars",
    # Form furniture and money labels. A printed label is not a business name,
    # and 'Total Sales (VAT Inclusive)' was being proposed as a vendor.
    "total sales", "vatable sales", "less:", "add:", "amount due", "total amount",
    "zero-rated", "vat-exempt", "vat exempt", "withholding", "in settlement",
    "settlement of the following", "signature", "subscription",
    "representative", "received the", "solo parent", "sc/pwd", "unit cost",
)

# Short exclusions need word boundaries. As plain substrings, 'tel' matches inside
# 'Telecommunications' - which silently discarded receipt 6's only real candidate,
# 'Bayan Telecommunications, Inc.', leaving the customer's name as the best guess.
VENDOR_NAME_EXCLUSION_WORDS = re.compile(
    r"\b(tel|tin|date|qty|atp|ocn|permit|address|invoice|discount)\b", re.IGNORECASE
)

# Document titles, matched loosely because OCR mangles them. 'OFFICIAL RECEIPT'
# came back as 'OFFICIAL RECEIpr' and slipped past an exact-substring exclusion.
DOCUMENT_TITLE_RE = re.compile(
    r"offi\w*\s*rece\w*|sale\w*\s*invo\w*|provisional\s*rece\w*|"
    r"servi\w*\s*invo\w*|billing\s*state\w*",
    re.IGNORECASE,
)

# Words that make up table column headings. A line composed only of these is
# furniture, not a business name - 'Description Amount' was being reported as the
# vendor with full confidence, which is worse than reporting nothing.
HEADING_VOCABULARY = frozenset({
    "description", "amount", "qty", "quantity", "unit", "units", "price", "cost",
    "particulars", "nature", "service", "item", "items", "total", "no", "rate",
    "vat", "sales", "net", "gross", "less", "add", "code", "ref",
})

CUSTOMER_MARKERS = ("received from", "registered name", "sold to", "bill to")


@dataclass
class CandidateMeta:
    """Signals used to rank a header line.

    ``confidence`` is Tesseract's mean per-word confidence for the line, and it is
    the discriminator that separates a correct reading from a corrupted one when
    both look structurally similar - 'UNITED DAILY PRESS INC' against
    'URITVED DAILY PRESS INGa'. ``occurrences`` counts how many independent
    readings produced the line; agreement across variants is corroboration.
    """

    index: int
    confidence: float
    occurrences: int = 1


@dataclass
class VendorCandidates:
    lines: list[str] = field(default_factory=list)
    customer_names: list[str] = field(default_factory=list)
    meta: dict[str, CandidateMeta] = field(default_factory=dict)


def find_vendor_candidates(
    line_sets: list[list[tuple[str, float]]], header_lines: int = 12
) -> VendorCandidates:
    """Collect plausible vendor-name lines from the printed headers of all readings.

    Drawing from every pooled reading matters: the primary reading is chosen for
    money-anchor coverage, so its header is often the worst of the set. Receipt 6's
    'Bayan Telecommunications, Inc.' is legible in one variant and absent from
    another, and considering only the primary picked the customer instead.

    Names following a customer marker are collected separately so the selection
    step can be told explicitly what NOT to choose.
    """
    result = VendorCandidates()

    for lines in line_sets:
        for index, (line_text, confidence) in enumerate(lines[:header_lines]):
            stripped = line_text.strip()
            lowered = stripped.casefold()
            if any(bad in lowered for bad in VENDOR_NAME_EXCLUSIONS):
                continue
            if VENDOR_NAME_EXCLUSION_WORDS.search(stripped):
                continue
            if DOCUMENT_TITLE_RE.search(stripped):
                continue
            if _is_column_heading(stripped):
                continue
            if not looks_like_words(stripped):
                continue

            existing = result.meta.get(stripped)
            if existing is None:
                result.meta[stripped] = CandidateMeta(index=index, confidence=confidence)
            else:
                existing.index = min(existing.index, index)
                existing.confidence = max(existing.confidence, confidence)
                existing.occurrences += 1

        for line_text, _ in lines:
            lowered = line_text.casefold()
            for marker in CUSTOMER_MARKERS:
                position = lowered.find(marker)
                if position == -1:
                    continue
                tail = line_text[position + len(marker):]
                tail = re.split(
                    r"\s{2,}|with tin|and address|engaged in",
                    tail, flags=re.IGNORECASE,
                )[0]
                tail = tail.strip(" :_-\u2014\u2013.,")
                if len(tail) >= 3:
                    result.customer_names.append(tail)

    result.lines = [
        name for name, _ in sorted(result.meta.items(), key=lambda kv: kv[1].index)
    ]
    return result


def _is_column_heading(line: str) -> bool:
    """Is every word on this line part of a table heading?

    Requires all tokens to be heading vocabulary, so a real name containing one
    such word ('Metro Hardware and Construction Supply') is unaffected.
    """
    tokens = re.findall(r"[A-Za-z]{2,}", line.casefold())
    if not tokens:
        return False
    return all(token in HEADING_VOCABULARY for token in tokens)


def text_lines(lines: list[list[Word]]) -> list[str]:
    return [" ".join(word.text for word in line) for line in lines]
