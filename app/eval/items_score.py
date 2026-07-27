"""Scoring for line items.

Three metrics rather than one, because "the items are wrong" means three different
things and they have different consequences:

* ROW COUNT   - did we emit the right number of rows? This is what catches a
                continuation line being split into a phantom second item, which
                would otherwise score well on names and prices.
* FIELD MATCH - per-row quantity, price and name accuracy, matched greedily on
                name so row ordering does not matter.
* RECONCILED  - how often the emitted rows add up to the printed subtotal. This is
                the pipeline's own gate, so it measures how often items are
                suppressed rather than shipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz

from app.core.items import UNREADABLE_NAME
from app.core.numbers import normalize_money

NAME_MATCH_THRESHOLD = 85.0
PRICE_TOLERANCE = 0.01


@dataclass
class ItemTally:
    receipts_with_expected_items: int = 0
    row_count_exact: int = 0
    expected_rows: int = 0
    matched_rows: int = 0
    quantity_correct: int = 0
    price_correct: int = 0
    name_correct: int = 0
    name_unreadable: int = 0
    reconciled_receipts: int = 0
    suppressed_receipts: int = 0
    misses: list[str] = field(default_factory=list)
    quantity_misses: list[str] = field(default_factory=list)

    def _ratio(self, numerator: int, denominator: int) -> float | None:
        return (numerator / denominator) if denominator else None

    @property
    def row_count_accuracy(self) -> float | None:
        return self._ratio(self.row_count_exact, self.receipts_with_expected_items)

    @property
    def recall(self) -> float | None:
        return self._ratio(self.matched_rows, self.expected_rows)

    @property
    def quantity_accuracy(self) -> float | None:
        return self._ratio(self.quantity_correct, self.matched_rows)

    @property
    def price_accuracy(self) -> float | None:
        return self._ratio(self.price_correct, self.matched_rows)

    @property
    def name_accuracy(self) -> float | None:
        return self._ratio(self.name_correct, self.matched_rows)

    @property
    def reconcile_rate(self) -> float | None:
        return self._ratio(self.reconciled_receipts, self.receipts_with_expected_items)


def score_items(
    tally: ItemTally,
    image: str,
    expected: list[dict[str, Any]] | None,
    produced: list[dict[str, Any]],
    reconciled: bool,
) -> None:
    """Accumulate one receipt's item results into ``tally``."""
    if not expected:
        return

    tally.receipts_with_expected_items += 1
    tally.expected_rows += len(expected)
    if reconciled:
        tally.reconciled_receipts += 1
    if not produced:
        tally.suppressed_receipts += 1

    if len(produced) == len(expected):
        tally.row_count_exact += 1
    else:
        tally.misses.append(
            f"{image}: expected {len(expected)} row(s), got {len(produced)}"
        )

    remaining = list(produced)
    for want in expected:
        best_index, best_score = None, 0.0
        for index, got in enumerate(remaining):
            score = fuzz.token_set_ratio(
                str(want.get("name", "")).casefold(),
                str(got.get("name", "")).casefold(),
            )
            if score > best_score:
                best_index, best_score = index, score

        # Fall back to price matching when the name is unreadable: the row was
        # still found, and judging it purely on a placeholder name would report a
        # miss where extraction actually succeeded on the money.
        if best_index is None or best_score < NAME_MATCH_THRESHOLD:
            best_index = _match_on_price(remaining, want)
            if best_index is None:
                continue
            best_score = 0.0

        got = remaining.pop(best_index)
        tally.matched_rows += 1

        if int(got.get("quantity", 0)) == int(want.get("quantity", 0)):
            tally.quantity_correct += 1
        else:
            tally.quantity_misses.append(
                f"{image}: qty expected {want.get('quantity')}, got "
                f"{got.get('quantity')} for {str(got.get('name'))[:40]!r}"
            )
        want_price = normalize_money(want.get("price"))
        got_price = normalize_money(got.get("price"))
        if (want_price is not None and got_price is not None
                and abs(want_price - got_price) <= PRICE_TOLERANCE):
            tally.price_correct += 1
        if str(got.get("name")) == UNREADABLE_NAME:
            tally.name_unreadable += 1
        elif best_score >= NAME_MATCH_THRESHOLD:
            tally.name_correct += 1


def _match_on_price(rows: list[dict[str, Any]], want: dict[str, Any]) -> int | None:
    target = normalize_money(want.get("price"))
    if target is None:
        return None
    for index, row in enumerate(rows):
        price = normalize_money(row.get("price"))
        if price is not None and abs(price - target) <= PRICE_TOLERANCE:
            return index
    return None


def render_items(tally: ItemTally) -> str:
    if not tally.receipts_with_expected_items:
        return "\nLINE ITEMS  --  no ground truth available"

    def show(value: float | None) -> str:
        return "   n/a" if value is None else f"{value:6.1%}"

    lines = [
        "",
        f"LINE ITEMS  --  {tally.receipts_with_expected_items} receipts, "
        f"{tally.expected_rows} expected rows",
        "",
        f"  row count exact      {show(tally.row_count_accuracy)}   gate 90%"
        f"   ({tally.row_count_exact}/{tally.receipts_with_expected_items})",
        f"  rows matched         {show(tally.recall)}"
        f"          ({tally.matched_rows}/{tally.expected_rows})",
        f"  quantity correct     {show(tally.quantity_accuracy)}   gate 95%",
        f"  price correct        {show(tally.price_accuracy)}   gate 90%",
        f"  name correct         {show(tally.name_accuracy)}   not gated"
        f"   ({tally.name_unreadable} unreadable placeholder(s))",
        f"  items reconciled     {show(tally.reconcile_rate)}"
        f"          ({tally.suppressed_receipts} receipt(s) shipped no items)",
    ]
    if tally.misses:
        lines.append("\n  row-count mismatches (first 10):")
        lines.extend(f"    {miss}" for miss in tally.misses[:10])
    if tally.quantity_misses:
        lines.append("\n  quantity mismatches (first 10):")
        lines.extend(f"    {miss}" for miss in tally.quantity_misses[:10])
    return "\n".join(lines)
