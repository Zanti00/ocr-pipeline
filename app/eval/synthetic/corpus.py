"""Corpus assembly: render, degrade, and emit matching ground truth.

Ground-truth documents use the same format as the real-sample fixtures, so the
evaluation harness reads both corpora through one code path.

Generation is seeded end to end. A corpus is therefore reproducible, and an
accuracy figure can be re-derived later rather than taken on trust.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.schema import MONEY_FIELDS
from app.eval.synthetic.degrade import DEGRADATIONS, degrade
from app.eval.synthetic.render import render_receipt
from app.eval.synthetic.spec import ReceiptSpec, build_spec

# Weighted towards clean and moderate: most submissions are ordinary photographs,
# and a corpus dominated by severe cases would understate real-world accuracy.
TIER_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("clean", 0.40), ("moderate", 0.40), ("severe", 0.20),
)

JPEG_QUALITY = 95


@dataclass
class CorpusStats:
    written: int = 0
    by_template: dict[str, int] = field(default_factory=dict)
    by_tier: dict[str, int] = field(default_factory=dict)
    corrupted: int = 0

    def record(self, spec: ReceiptSpec, tier: str) -> None:
        self.written += 1
        self.by_template[spec.template] = self.by_template.get(spec.template, 0) + 1
        self.by_tier[tier] = self.by_tier.get(tier, 0) + 1
        if spec.corrupt_arithmetic:
            self.corrupted += 1


def generate_corpus(
    output_dir: Path,
    count: int,
    corrupted_count: int = 0,
    seed: int = 1,
) -> CorpusStats:
    """Write ``count`` sound receipts plus ``corrupted_count`` inconsistent ones."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    stats = CorpusStats()

    for index in range(count):
        spec = build_spec(index, rng)
        _emit(spec, _tier_for(index, count), index, output_dir, rng, stats)

    # Deliberately inconsistent receipts are always VAT forms: they are the only
    # template where an identity exists to violate.
    for offset in range(corrupted_count):
        index = count + offset
        spec = build_spec(index, rng, template="ph_vat_or")
        spec.corrupt_arithmetic = True
        _emit(spec, _tier_for(offset, max(corrupted_count, 1)), index,
              output_dir, rng, stats)

    return stats


def _emit(
    spec: ReceiptSpec,
    tier: str,
    index: int,
    output_dir: Path,
    rng: random.Random,
    stats: CorpusStats,
) -> None:
    image_name = f"syn-{index:04d}-{spec.template}-{tier}.jpg"
    image = degrade(render_receipt(spec), tier, rng)
    image.convert("L").save(output_dir / image_name, quality=JPEG_QUALITY)

    document = _ground_truth(spec, tier, image_name)
    (output_dir / f"{Path(image_name).stem}.json").write_text(
        json.dumps(document, indent=2, sort_keys=True), encoding="utf-8"
    )
    stats.record(spec, tier)


def _tier_for(index: int, count: int) -> str:
    """Deterministic stratification, so tier counts are exact rather than random."""
    position = index / max(count, 1)
    cumulative = 0.0
    for tier, weight in TIER_WEIGHTS:
        cumulative += weight
        if position < cumulative:
            return tier
    return DEGRADATIONS[-1]


def _ground_truth(spec: ReceiptSpec, tier: str, image_name: str) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "vendor_name": spec.vendor_name,
        "country": spec.country,
        "currency": spec.currency,
        "vendor_tax_id": spec.vendor_tax_id,
        "vendor_tax_id_type": spec.vendor_tax_id_type,
        "transaction_date": spec.transaction_date.isoformat(),
        "net_sales": spec.net_sales,
        "tax_amount": spec.tax_amount,
        "tax_type": spec.tax_type,
        "total_sales": spec.total_sales,
        "service_charge": spec.service_charge,
        "total_amount": spec.total_amount,
        "vat_classification": spec.vat_classification,
        "invoice_number": spec.invoice_number,
        "expense_category": spec.expense_category,
    }

    not_expected: dict[str, list[str]] = {}
    if spec.customer_tax_id:
        not_expected["vendor_tax_id"] = [spec.customer_tax_id]
    if spec.customer_name:
        not_expected["vendor_name"] = [spec.customer_name]

    # tax_rate is derived rather than printed. Items ARE scored now: the generator
    # knows every row exactly, so it is the only place item accuracy can be
    # measured at a useful sample size.
    unknown = ["tax_rate"]

    notes = (
        f"Generated {spec.template} receipt, degradation '{tier}', "
        f"font '{spec.font_name}', date rendered as '{spec.date_text}'."
    )

    if spec.corrupt_arithmetic:
        # The printed tax contradicts the sales figures on purpose. Scoring the
        # amounts here would reward reading a knowingly wrong number, so they are
        # excluded and the expectation becomes the gate's behaviour instead.
        unknown.extend(MONEY_FIELDS)
        notes += (
            " DELIBERATELY INCONSISTENT: the printed tax figure does not satisfy the"
            f" VAT identity (printed {spec.printed_tax_amount}, consistent value"
            f" {spec.tax_amount}). The reconciliation gate must fail and the receipt"
            " must be flagged for review rather than accepted."
        )

    return {
        "image": image_name,
        "source": "synthetic",
        "script": "latin",
        "print_type": "machine_printed",
        "truth_source": "generated",
        "degradation": tier,
        "template": spec.template,
        "expect_manual_review": spec.corrupt_arithmetic,
        "expect_reconciliation_failure": spec.corrupt_arithmetic,
        "notes": notes,
        "expected": expected,
        # expected_name folds any description line in, because a printed
        # continuation belongs to the item above it and must not become a row.
        "expected_items": [
            {"name": item.expected_name, "quantity": item.quantity,
             "price": item.amount}
            for item in spec.items
        ],
        "not_expected_values": not_expected,
        "unknown": sorted(set(unknown)),
    }
