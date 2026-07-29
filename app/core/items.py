"""Line-item extraction.

Deterministic throughout: an item row is a line inside the item region carrying a
money token, and its quantity is the leading integer. No language model is
involved, because an item name is a transcription rather than a judgement - a model
asked to tidy 'ChickenJjoy' will eventually produce product names that were never
printed, and unlike a vendor name there is no candidate shortlist to constrain it
against.

Two rules carry most of the weight:

* A line with no money token is a DESCRIPTION of the item above it, not a new
  item. 'Family bundle--10pcs' followed by 'Chickenjoy bucket + 3 large sides' is
  one physical purchase and must produce one row.
* Items are emitted only when they add up. Receipts disagree about whether the
  printed figure is the line total or the unit price, so both readings are summed
  and whichever reconciles against the subtotal wins. If neither does, the items
  are withheld entirely - a partial list looks authoritative while being wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.layout import _best_label, _collect_money_tokens
from app.core.numbers import normalize_money
from app.core.ocr_engine import Word
from app.core.textquality import looks_like_words

# Lines that introduce the item region. The region may also begin at a printed
# rule, so a separator counts too.
REGION_START_PATTERNS = (
    r"^details$", r"nature\s+of\s+service", r"particulars", r"description",
    r"\bqty\b", r"unit\s+cost", r"unit\s+price", r"^item", r"^\*{4,}$", r"^-{4,}$",
)
_REGION_START_RE = re.compile("|".join(REGION_START_PATTERNS), re.IGNORECASE)

# The item region ends at the first summary line. These are the money labels the
# layout scanner already recognises.
SUMMARY_FIELDS = frozenset({
    "total_amount", "total_sales", "net_sales", "tax_amount",
    "service_charge", "discount_amount", "withholding_tax",
})

# Footer and payment noise that can carry money tokens.
ITEM_NOISE_PATTERNS = (
    r"change\s+due", r"payment", r"\bcash\b", r"\bcard\b", r"tender",
    r"authority\s+to\s+print", r"\bocn\b", r"booklets", r"valid\s+until",
    r"\batp\b", r"vat\s*reg", r"\btin\b", r"balance", r"amount\s+in\s+words",
    r"total\s+item", r"item\s+sold", r"regular\s+price", r"card\s+savings?",
    r"card\s+saver", r"mfrcpn", r"cartwheel",
)
_ITEM_NOISE_RE = re.compile("|".join(ITEM_NOISE_PATTERNS), re.IGNORECASE)

_LEADING_QUANTITY_RE = re.compile(r"^\s*(\d{1,3})\s*(?:x|pc|pcs|pieces)?\s+(?=\S)")
_UPC_PREFIX_RE = re.compile(r"^\s*(?:\d{8,14}|[A-Za-z0-9]{2,6}\d{7,14})\s+")
# 'name ... x2' and, more importantly, the BIR form layout which prints the
# quantity in its own column immediately left of the amount: 'Bond Paper 2 490.00'.
# Without this the quantity silently defaults to 1 on every form receipt.
_TRAILING_QUANTITY_RE = re.compile(
    r"(?:\s+x\s*(\d{1,3})|\s+(\d{1,3}))\s*$", re.IGNORECASE
)

MAX_NAME_LENGTH = 255
MAX_ITEMS = 40

# Stand-in for a name OCR could not read. Faded thermal print yields rows whose
# price is perfectly legible and whose product name is noise ('ci cr } UM SDR \').
#
# Dropping such a row would be worse: on a three-item receipt one unreadable name
# would leave two rows summing to less than the subtotal, the sum check would fail,
# and ALL items would be withheld - including the two read correctly. Keeping the
# row preserves the arithmetic, and a placeholder states honestly that the name is
# unknown instead of presenting noise as a product.
UNREADABLE_NAME = "Unreadable item"


@dataclass
class ExtractedItem:
    name: str
    quantity: int
    price: float
    line_text: str = ""
    descriptions: list[str] = field(default_factory=list)

    def payload(self, price: float | None = None) -> dict[str, object]:
        return {
            "name": self.full_name,
            "quantity": self.quantity,
            "price": round(price if price is not None else self.price, 2),
        }

    @property
    def full_name(self) -> str:
        parts = [self.name, *self.descriptions]
        combined = " ".join(part for part in parts if part).strip()
        if not looks_like_item_name(combined):
            return UNREADABLE_NAME
        return combined[:MAX_NAME_LENGTH]

    @property
    def name_is_readable(self) -> bool:
        return self.full_name != UNREADABLE_NAME


@dataclass
class ItemScan:
    items: list[ExtractedItem] = field(default_factory=list)
    reconciled: bool = False
    price_basis: str = "none"
    """``line_total`` or ``unit_price`` - which reading of the printed figure
    reconciled. The emitted price is always the line total regardless, because
    SERMS' receipt_items table has a single price column."""
    total: float | None = None
    target: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.items)

    def payload(self) -> list[dict[str, object]]:
        """Rows to transmit."""
        rows: list[dict[str, object]] = []
        for item in self.items:
            price = (
                item.price if self.price_basis in ("line_total", "none")
                else round(item.price * item.quantity, 2)
            )
            if price <= 0:
                # SERMS' PRS path validates item price as gt:0, so a non-positive
                # line is withheld from payload to avoid API rejection.
                continue
            rows.append(item.payload(price))
        return rows


def parse_items(lines: list[list[Word]]) -> list[ExtractedItem]:
    """Extract item rows from one OCR reading.

    Expects a reading that preserves visual rows. Sparse-text segmentation emits
    the amount column separately from the names - and in reverse order - so a
    caller must supply a block-layout reading for this to mean anything.

    Tried twice. The first pass requires a region marker ('Details', a column
    heading, a printed rule); the second drops that requirement, because thermal
    slips separate the region with a row of asterisks that OCR turns into noise
    ('AERIS IER EOE Ea Ooo'). The noise filters and the summary-label terminator
    still bound the region in the second pass.
    """
    items = _parse_region(lines, require_marker=True)
    if not items:
        items = _parse_region(lines, require_marker=False)
    return _strip_running_subtotal(items)


def _strip_running_subtotal(items: list[ExtractedItem]) -> list[ExtractedItem]:
    """Drop a row whose amount equals the sum of the rows above it.

    Such a row is a subtotal, not a product. Receipt 11 prints its department
    subtotal as 'FOOD 64.90', which no label vocabulary would catch - but the
    arithmetic gives it away, because 64.90 is exactly the eight lines above it.

    Solving this by arithmetic rather than by adding more label spellings means it
    works for 'BEVERAGE', 'DEPT 1', 'SUB-TOTAL' and anything else a POS prints,
    including wordings OCR has mangled beyond recognition.
    """
    if len(items) < 3:
        return items  # too few rows for a sum to be meaningful

    running = 0.0
    for index, item in enumerate(items):
        if index >= 2 and abs(item.price - running) <= 0.02:
            return items[:index]
        running = round(running + item.price, 2)
    return items


def _parse_region(lines: list[list[Word]], require_marker: bool) -> list[ExtractedItem]:
    tokens = _collect_money_tokens(lines)
    money_by_word = {id(token.word): token.text for token in tokens}
    # Every word consumed by a money token, so the item name can be rebuilt from
    # the remaining words. A text replace cannot do this: the reassembled token
    # '$29,99' does not appear in the raw line '1 Family bundle--10pcs $29 ,99'.
    token_word_ids = {id(word) for token in tokens for word in token.words}

    items: list[ExtractedItem] = []
    started = not require_marker
    seen_item = False

    for line in lines:
        text = " ".join(word.text for word in line).strip()
        if not text:
            continue

        label = _best_label(text)
        if label and label[0] in SUMMARY_FIELDS:
            if seen_item:
                break  # the first summary line after the items ends the region
            # A summary label before any item does NOT open the region. Treating it
            # as an opener made a footer 'Discount' line start a phantom region, so
            # 'AnnivDiscount $8.2' and 'Amount $16.0' were returned as the items and
            # the real rows above were never reached.
            continue

        if not started and _REGION_START_RE.search(text):
            started = True
            continue
        if not started:
            continue
        if _ITEM_NOISE_RE.search(text):
            continue

        money = [money_by_word[id(word)] for word in line if id(word) in money_by_word]

        if money:
            remainder = " ".join(
                word.text for word in line if id(word) not in token_word_ids
            )
            item = _build_item(remainder, money[-1], line_text=text)
            if item is not None:
                items.append(item)
                seen_item = True
            continue

        # No money on this line: a description of the item above it.
        if items and _plausible_description(text):
            items[-1].descriptions.append(text)

        if len(items) >= MAX_ITEMS:
            break

    return items


def _build_item(
    remainder: str, money_token: str, line_text: str = ""
) -> ExtractedItem | None:
    """Build an item from a line with its money words already removed."""
    price = normalize_money(money_token)
    if price is None or price == 0:
        return None

    quantity = 1

    match = _LEADING_QUANTITY_RE.match(remainder)
    if match:
        quantity = max(1, int(match.group(1)))
        remainder = remainder[match.end():]
    else:
        trailing = _TRAILING_QUANTITY_RE.search(remainder)
        if trailing:
            digits = trailing.group(1) or trailing.group(2)
            quantity = max(1, int(digits))
            remainder = remainder[: trailing.start()]

    name = _clean_name(remainder)
    if not name:
        return None
    return ExtractedItem(
        name=name, quantity=quantity, price=price, line_text=line_text or remainder
    )


def _clean_name(text: str) -> str:
    """Tidy the remaining words of an item line into a name.

    Deliberately keeps unreadable text rather than discarding the row: quality is
    judged later by ``full_name``, which substitutes a placeholder. Dropping the
    row here would break the item sum and cost the whole list.

    A remainder with no letters at all is rejected, because a bare amount with no
    accompanying text is a summary figure rather than a product line.
    """
    cleaned = re.sub(r"[|_]+", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;-*\\/")
    cleaned = _UPC_PREFIX_RE.sub("", cleaned).strip(" .,:;-*\\/")
    if not any(ch.isalpha() for ch in cleaned):
        return ""
    return cleaned[:MAX_NAME_LENGTH]


def looks_like_item_name(text: str) -> bool:
    """Is this a plausible product name?

    Deliberately laxer than the vendor-name test. That one requires two
    word-shaped tokens because receipt headers are surrounded by OCR debris, but
    real menu lines are frequently a single short word - 'SALAD', 'SODA', 'FRIES'.
    Reusing the stricter rule replaced perfectly good names with a placeholder.
    """
    stripped = text.strip()
    letters = sum(1 for ch in stripped if ch.isalpha())
    if letters < 3:
        return False
    if letters < len(stripped) * 0.5:
        return False

    tokens = re.findall(r"[A-Za-z][A-Za-z.'&/-]*", stripped)
    solid = [token for token in tokens if len(token) >= 3]
    if not solid:
        return False
    # One word is enough if it is a real word length ('SALAD', 'SODA'), but a lone
    # three-letter fragment among noise is not - 'ci cr } UM SDR \' would otherwise
    # pass on the strength of 'SDR'.
    if len(solid) == 1 and len(solid[0]) < 4:
        return False
    # Noise alternates case mid-word; product names are caps or title case.
    odd = sum(1 for token in solid if re.search(r"[a-z][A-Z]", token))
    return odd <= len(solid) * 0.5


def _plausible_description(text: str) -> bool:
    letters = sum(1 for ch in text if ch.isalpha())
    return letters >= 3 and letters >= len(text) * 0.4


def reconcile_items(
    items: list[ExtractedItem],
    targets: dict[str, float | None],
    tolerance_base: float = 0.02,
) -> ItemScan:
    """Decide whether the item rows add up, and on which price basis.

    ``targets`` holds the candidate subtotals - ``net_sales`` on a tax-exclusive
    receipt, ``total_sales`` on a VAT-inclusive one. Either is a legitimate match
    depending on the document, so both are tried.
    """
    scan = ItemScan(items=items)
    if not items:
        scan.notes.append("no item rows found")
        return scan

    tolerance = max(tolerance_base, 0.01 * len(items))
    line_total = round(sum(item.price for item in items), 2)
    unit_total = round(sum(item.price * item.quantity for item in items), 2)

    candidates = {
        name: value for name, value in targets.items() if value is not None
    }
    if not candidates:
        scan.notes.append("no subtotal available to check items against")
        return scan

    # Prefer the line-total reading: it needs no assumption about the receipt's
    # convention, and for single-quantity rows the two are identical anyway.
    for basis, total in (("line_total", line_total), ("unit_price", unit_total)):
        for name, target in candidates.items():
            if abs(total - target) <= tolerance:
                scan.reconciled = True
                scan.price_basis = basis
                scan.total = total
                scan.target = target
                scan.notes.append(
                    f"items reconcile on {basis}: {total:.2f} vs {name}={target:.2f}"
                )
                return scan

    scan.total = line_total
    scan.notes.append(
        f"items do not reconcile: line_total={line_total:.2f}, "
        f"unit_total={unit_total:.2f}, targets="
        + ", ".join(f"{k}={v:.2f}" for k, v in candidates.items())
        + " -> withheld"
    )
    return scan
