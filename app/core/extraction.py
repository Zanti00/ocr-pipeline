"""Deterministic extraction orchestrator.

Produces a complete field set from OCR output alone, with no language model
involved. The LLM is layered on afterwards for vendor-name selection and expense
category only, so this stage is independently testable and independently
measurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.categorize import (
    CategoryChoice, Embedder, Tiebreaker, classify_category,
)
from app.core.extractors import (
    DateCandidate,
    TaxIdCandidate,
    VendorCandidates,
    find_dates,
    find_invoice_number,
    find_tax_ids,
    find_vendor_candidates,
    select_vendor_tax_id,
    text_lines,
)
from app.core.items import ItemScan, parse_items, reconcile_items
from app.core.layout import LayoutScan, scan_layout
from app.core.locale import LocaleGuess, detect_locale
from app.core.ocr_engine import OcrBundle
from app.core.reconcile import (
    MoneyResult, classify_tax, resolve_money, resolve_tax_type,
)
from app.core.vendor import VendorChoice, select_vendor_name


@dataclass
class Extraction:
    """Everything derivable without a language model, plus the evidence trail."""

    fields: dict[str, Any] = field(default_factory=dict)
    locale: LocaleGuess | None = None
    money: MoneyResult | None = None
    layout: LayoutScan | None = None
    tax_id_candidates: list[TaxIdCandidate] = field(default_factory=list)
    date_candidates: list[DateCandidate] = field(default_factory=list)
    vendor_candidates: VendorCandidates = field(default_factory=VendorCandidates)
    vendor_choice: VendorChoice | None = None
    category_choice: CategoryChoice | None = None
    item_scan: ItemScan | None = None
    evidence: dict[str, str] = field(default_factory=dict)
    lines: list[str] = field(default_factory=list)

    @property
    def reconciled(self) -> bool:
        return bool(self.money and self.money.reconciled)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.fields)


def extract(
    bundle: OcrBundle,
    *,
    llm_vendor_choice: str | None = None,
    embedder: Embedder | None = None,
    category_tiebreaker: Tiebreaker | None = None,
) -> Extraction:
    """Run extraction over a pooled OCR reading.

    All three optional arguments are upgrades, not requirements. With none of them
    supplied the pipeline is fully deterministic and offline, which is also how the
    evaluation harness runs it.
    """
    primary_lines = text_lines(bundle.primary.lines)
    pooled_lines = _pooled_lines(bundle)
    combined = bundle.combined_text

    locale = detect_locale(combined)

    # Money pairing is geometric, so the reading that best preserves row structure
    # wins - not necessarily the one with the best anchor score. Block-oriented
    # segmentation keeps a label and its figure on one row, while sparse-text mode
    # scatters them, so every pooled reading is scanned and the most productive
    # one is kept.
    # VAT registration is read from the printed header, before any money is
    # resolved, because it decides whether VAT may be derived at all.
    vat_registered = classify_tax(combined, locale, None) == "vat"
    layout, money = _best_layout(bundle, locale, vat_registered)

    tax_ids = find_tax_ids(pooled_lines)
    vendor_tax = select_vendor_tax_id(tax_ids)

    dates = find_dates(pooled_lines, country=locale.country)
    best_date = dates[0] if dates and dates[0].score > 0 else None

    invoice_number, invoice_line = find_invoice_number(pooled_lines)
    vendors = find_vendor_candidates(
        [_lines_with_confidence(reading)
         for reading in [bundle.primary, *bundle.supporting]]
    )

    # Deterministic ranking runs first and always. An LLM selection, when one is
    # supplied, may only reorder within this shortlist - never add to it.
    vendor_choice = select_vendor_name(
        vendors.lines, vendors.customer_names,
        llm_choice=llm_vendor_choice, meta=vendors.meta,
    )
    category = classify_category(
        vendor_choice.name,
        item_text=" ".join(pooled_lines[:25]),
        embedder=embedder,
        tiebreaker=category_tiebreaker,
    )

    extraction = Extraction(
        locale=locale,
        money=money,
        layout=layout,
        tax_id_candidates=tax_ids,
        date_candidates=dates,
        vendor_candidates=vendors,
        lines=pooled_lines,
    )

    item_scan = _best_items(bundle, money)
    vat_classification = classify_tax(combined, locale, money.get("tax_amount"))

    extraction.fields = {
        "country": locale.country,
        "currency": locale.currency,
        "vendor_tax_id": vendor_tax.formatted if vendor_tax else None,
        "vendor_tax_id_type": _tax_id_type(locale, vendor_tax),
        "transaction_date": best_date.value.isoformat() if best_date else None,
        "net_sales": money.get("net_sales"),
        "tax_amount": money.get("tax_amount"),
        "tax_type": resolve_tax_type(
            locale.country, vat_classification, money.get("tax_amount")
        ),
        "tax_rate": money.tax_rate,
        "total_sales": money.get("total_sales"),
        "service_charge": money.get("service_charge"),
        "total_amount": money.get("total_amount"),
        "vat_classification": vat_classification,
        "invoice_number": invoice_number,
        "vendor_name": vendor_choice.name,
        "expense_category": category.category,
    }
    extraction.vendor_choice = vendor_choice
    extraction.category_choice = category
    extraction.item_scan = item_scan

    extraction.evidence.update(money.evidence)
    if vendor_tax:
        extraction.evidence["vendor_tax_id"] = (
            f"role={vendor_tax.role} line={vendor_tax.line_index} "
            f"{vendor_tax.line_text!r}"
        )
    if best_date:
        extraction.evidence["transaction_date"] = (
            f"{best_date.raw!r} score={best_date.score:.2f} on {best_date.line_text!r}"
        )
    if invoice_line:
        extraction.evidence["invoice_number"] = invoice_line
    if locale.evidence:
        extraction.evidence["locale"] = ", ".join(locale.evidence)
    extraction.evidence["vendor_name"] = (
        f"{vendor_choice.method} from {vendor_choice.shortlist[:3]}"
    )
    if item_scan.notes:
        extraction.evidence["items"] = (
            f"{item_scan.count} row(s), basis={item_scan.price_basis}; "
            + "; ".join(item_scan.notes)
        )
    extraction.evidence["expense_category"] = (
        f"{category.method}"
        + (f" (runner-up {category.runner_up}, sim {category.similarity})"
           if category.runner_up else "")
    )

    return extraction


def _best_items(bundle: OcrBundle, money: MoneyResult) -> ItemScan:
    """Parse items from whichever reading produces a set that adds up.

    Item parsing needs a reading that preserves visual rows, and that is generally
    NOT the anchor-selected primary: sparse-text segmentation scores highest on
    anchors while emitting the amount column separately from the item names, in
    reverse order. Rather than guess which page-segmentation mode to trust, every
    pooled reading is parsed and the arithmetic decides - a reading whose items sum
    to the printed subtotal has almost certainly read the rows correctly.

    Uses readings already computed for the OCR stage, so this costs no extra
    Tesseract work.
    """
    targets = {
        "net_sales": money.get("net_sales"),
        "total_sales": money.get("total_sales"),
    }

    # Many POS slips print no subtotal at all - receipt 11 goes straight from the
    # item lines to TAX and TOTAL. Without a target the item sum cannot be checked
    # and every row is discarded, so the subtotal implied by the total is offered
    # as a fallback candidate.
    total = money.get("total_amount")
    if total is not None:
        implied = round(
            total - (money.get("tax_amount") or 0.0)
            - (money.get("service_charge") or 0.0), 2
        )
        targets["total_amount"] = total
        if implied > 0 and abs(implied - total) > 0.005:
            targets["total_less_tax_and_charges"] = implied

    best: ItemScan | None = None
    for reading in bundle.all_readings:
        scan = reconcile_items(parse_items(reading.lines), targets)
        rank = (1 if scan.reconciled else 0, scan.count)
        if best is None or rank > (1 if best.reconciled else 0, best.count):
            best = scan
        if scan.reconciled:
            break  # an arithmetically consistent reading is good enough

    return best or ItemScan()


def _best_layout(
    bundle: OcrBundle, locale: LocaleGuess, vat_registered: bool = True
) -> tuple[LayoutScan, MoneyResult]:
    """Scan every pooled reading and keep the most productive money extraction.

    Ranked by: arithmetic reconciliation first, then how many distinct labels were
    paired, then whether a total was found. Reconciliation leads because a set of
    figures whose identities close is far more likely to be read correctly than a
    larger set that contradicts itself.
    """
    best: tuple[tuple[int, int, int], LayoutScan, MoneyResult] | None = None

    for reading in [bundle.primary, *bundle.supporting]:
        scan = scan_layout(reading.lines)
        money = resolve_money(scan, locale, vat_registered)
        rank = (
            1 if money.reconciled else 0,
            scan.label_count,
            1 if money.get("total_amount") is not None else 0,
        )
        if best is None or rank > best[0]:
            best = (rank, scan, money)

    if best is None:  # pragma: no cover - bundle always has a primary
        empty = LayoutScan()
        return empty, resolve_money(empty, locale, vat_registered)
    return best[1], best[2]


def _lines_with_confidence(reading) -> list[tuple[str, float]]:
    """Pair each line's text with Tesseract's mean per-word confidence for it."""
    pairs: list[tuple[str, float]] = []
    for line in reading.lines:
        text = " ".join(word.text for word in line)
        confidence = sum(word.confidence for word in line) / len(line) if line else 0.0
        pairs.append((text, confidence))
    return pairs


def _pooled_lines(bundle: OcrBundle) -> list[str]:
    """Lines from the primary reading first, then anything new from supporters."""
    seen: set[str] = set()
    pooled: list[str] = []
    for reading in [bundle.primary, *bundle.supporting]:
        for line in text_lines(reading.lines):
            key = line.strip().casefold()
            if key and key not in seen:
                seen.add(key)
                pooled.append(line)
    return pooled


def _tax_id_type(locale: LocaleGuess, candidate: TaxIdCandidate | None) -> str | None:
    if candidate is None:
        return None
    return "PH_TIN" if locale.country == "PH" else "TAX_ID"
