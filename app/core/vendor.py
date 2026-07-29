"""Vendor name selection.

The one field where judgement is genuinely required: a receipt header mixes the
business name with addresses, phone numbers, tax ids and a proprietor's name, and
which line is "the vendor" is not decidable by pattern alone.

Even so the language model does not *generate* the name - it picks from a ranked
shortlist this module produces. A choice outside the shortlist is rejected. That
makes fabricating a vendor structurally impossible rather than merely discouraged,
and it means a model failure degrades to the deterministic pick instead of to
nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from app.core.textquality import looks_like_words

# Legal and trade suffixes are the strongest signal that a line names a business.
ENTITY_MARKERS = (
    "inc", "inc.", "incorporated", "corp", "corp.", "corporation", "co.",
    "company", "ltd", "ltd.", "limited", "sdn bhd", "sdn.bhd", "sdn. bhd",
    "enterprises", "enterprise", "trading", "restaurant", "cafe", "coffee",
    "store", "mart", "market", "supermarket", "pharmacy", "laboratory", "labs",
    "press", "printing", "services", "solutions", "hotel", "inn", "resort",
    "wholesaling", "retail", "foods", "bakeshop", "bake shop", "bakery", "grill", "diner",
    "transport", "logistics", "construction", "supply", "supplies", "industries",
    "holdings", "group", "ventures", "marketing", "distributor", "clinic",
    "diagnostics", "telecom", "communications", "philippines", "residences",
    "farmer", "tree", "target", "safeway", "dollar tree",
)

# Address, contact and form-furniture lines. These sit in the header and look like
# candidates, but naming one as the vendor is a misread.
ADDRESS_MARKERS = (
    " st.", " st ", " street ", " ave ", " avenue ", " road ", " rd ", "brgy", "barangay",
    " city ", " unit ", " blk ", " block ", " suite ", " floor ", " flr ", " bldg ", " building ",
    "district", "province", "philippines", " zip ", "p.o. box",
    "telefax", "fax", "phone", "mobile", "email", "@", "www",
)

PROPRIETOR_MARKERS = ("prop.", "proprietor", "- prop", "owner")

# Structural address patterns the substring list cannot express. A floor
# designation like '2/F Ever Gotesco Cc' carries no address *word*, so it scored as
# a plausible business name and outranked the real header.
ADDRESS_PATTERNS = re.compile(
    r"\b\d+\s*/\s*[a-z]\b"          # 2/F, 3/f
    r"|\b[a-z]?\d+\s*(?:st|nd|rd|th)\s+(?:floor|flr)\b"
    r"|\bcor\.?\s"                   # 'cor.' as in 'X cor. Y'
    r"|\b(?:blvd|ave|st|rd|hwy|lot|blk|brgy|bldg|unit|suite|rm)\b\.?"
    r"|\b\d{4,5}\b.*\b(?:city|manila|philippines)\b",
    re.IGNORECASE,
)

MIN_CANDIDATE_SCORE = 1.0
SHORTLIST_SIZE = 5


@dataclass
class VendorChoice:
    name: str | None
    method: str
    shortlist: list[str]
    score: float = 0.0


def rank_candidates(
    candidate_lines: list[str],
    customer_names: list[str] | None = None,
    meta: dict[str, Any] | None = None,
) -> list[tuple[str, float]]:
    """Score header lines by how much they look like a business name."""
    customer_names = customer_names or []
    meta = meta or {}
    ranked: list[tuple[str, float]] = []

    for index, raw in enumerate(candidate_lines):
        line = _tidy(raw)
        if not _plausible(line):
            continue

        info = meta.get(raw)
        position = getattr(info, "index", index)
        score = 2.0 - (position * 0.35)  # the business name leads the header
        lowered = line.casefold()

        # OCR confidence separates a clean reading from a corrupted one when both
        # are structurally plausible, and cross-variant agreement corroborates it.
        if info is not None:
            score += 1.8 * max(0.0, min(getattr(info, "confidence", 0.0), 1.0))
            score += 0.5 * min(getattr(info, "occurrences", 1) - 1, 3)

        # Cumulative, not binary. 'UNITED DAILY PRESS INC' carries two markers
        # ('press', 'inc') while its corrupted twin 'URITVED DAILY PRESS INGa'
        # carries one, and that difference is what separates them.
        marker_hits = sum(1 for marker in ENTITY_MARKERS if marker in lowered)
        score += 1.2 * min(marker_hits, 2)

        if any(marker in lowered for marker in ADDRESS_MARKERS):
            score -= 2.5
        if ADDRESS_PATTERNS.search(line):
            score -= 2.5
        if any(marker in lowered for marker in PROPRIETOR_MARKERS):
            score -= 1.0

        digits = sum(1 for ch in line if ch.isdigit())
        if digits >= 4:
            score -= 1.5
        if len(line.split()) == 1:
            score -= 0.5
        if line.isupper() and len(line) > 8:
            score += 0.4  # printed headers are usually set in caps

        # Never propose the payer as the vendor.
        if any(fuzz.token_set_ratio(lowered, name.casefold()) >= 85
               for name in customer_names):
            score -= 4.0

        ranked.append((line, round(score, 3)))

    ranked.sort(key=lambda pair: -pair[1])
    return ranked


def select_vendor_name(
    candidate_lines: list[str],
    customer_names: list[str] | None = None,
    llm_choice: str | None = None,
    meta: dict[str, Any] | None = None,
) -> VendorChoice:
    """Pick a vendor name, optionally honouring a model's selection.

    ``llm_choice`` is accepted only if it matches a shortlisted candidate. This is
    the constrained-selection rule: the model narrows a decision, it never invents
    an answer.
    """
    ranked = rank_candidates(candidate_lines, customer_names, meta)
    shortlist = [name for name, score in ranked[:SHORTLIST_SIZE]
                 if score >= MIN_CANDIDATE_SCORE]

    if not shortlist:
        return VendorChoice(name=None, method="no_candidate", shortlist=[])

    best, best_score = ranked[0][0], ranked[0][1]

    if llm_choice:
        matched = _closest_candidate(llm_choice, shortlist)
        if matched is not None:
            return VendorChoice(name=matched, method="llm_selection",
                                shortlist=shortlist, score=best_score)
        return VendorChoice(name=best, method="llm_rejected_off_shortlist",
                            shortlist=shortlist, score=best_score)

    return VendorChoice(name=best, method="deterministic", shortlist=shortlist,
                        score=best_score)


def _closest_candidate(choice: str, shortlist: list[str]) -> str | None:
    """Map a model's answer onto the shortlist entry it most closely matches.

    Ranked by exact-string similarity after a containment check, because
    ``token_set_ratio`` alone scores 100 for any subset: a model answering
    'Jollibee' would otherwise be resolved to 'Jollibee Las Vegas'.
    """
    needle = choice.casefold().strip()
    if not needle:
        return None

    scored: list[tuple[float, float, str]] = []
    for candidate in shortlist:
        target = candidate.casefold()
        containment = fuzz.token_set_ratio(needle, target)
        exactness = fuzz.ratio(needle, target)
        if containment >= 85:
            scored.append((containment, exactness, candidate))

    if not scored:
        return None
    scored.sort(key=lambda entry: (-entry[1], -entry[0]))
    return scored[0][2]


def _tidy(line: str) -> str:
    line = re.sub(r"\s+", " ", line).strip()
    # A pipe usually marks a column break where OCR merged unrelated regions, as in
    # '<aauanvor we rouowne| DETOXICARE MOLECULAR DIAGNOSTICS LABORATORY, INC'.
    # Keeping the longest segment discards the debris and preserves the name.
    if "|" in line:
        line = max(line.split("|"), key=lambda part: _alpha_count(part)).strip()
    line = line.strip("|_-*:;,.<>~@\\/{}()[]# ")
    # Drop single-char prefix noise like 'G DOLLAR TREE' -> 'DOLLAR TREE'
    line = re.sub(r"^[A-Za-z0-9@\\/{}\(\)\[\]#]\s+(?=[A-Za-z]{2,})", "", line).strip()
    # Drop leading lowercase debris
    line = re.sub(r"^(?:[a-z0-9@\\/{}\(\)\[\]#]{1,3}\s+)+(?=[A-Z]{2,})", "", line).strip()
    cleaned = line.strip("|_-*:;,.<>~@\\/{}()[]# ")

    # Brand OCR aliases for common retail chains
    lowered = cleaned.casefold()
    if "larget" in lowered or "target" in lowered or "iar more" in lowered or "pay get" in lowered:
        return "Target"
    if "dollar tree" in lowered or "dollartree" in lowered:
        return "Dollar Tree"
    if "safeway" in lowered or "sfwy" in lowered or "safew" in lowered:
        return "Safeway"

    return cleaned


def _alpha_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def _plausible(line: str) -> bool:
    # Same word-shape test used during candidate collection, so ranking is safe to
    # call on an unfiltered list.
    return looks_like_words(line)
