"""Run the extraction pipeline over a local image and print what SERMS would get.

    python scripts/try_receipt.py "docs/receipts/receipt 1.jpg"
    python scripts/try_receipt.py path/to/photo.jpg --json
    python scripts/try_receipt.py path/to/scan.pdf --variants

Requires no running services. Uses the same OCR, extraction, verification and
confidence code as the Celery worker, so what is printed here is what the callback
would carry - only the HTTP delivery, Mongo write and embedding storage are absent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from app.api.schemas.ocr import build_callback_payload  # noqa: E402
from app.core.confidence import SERMS_REVIEW_THRESHOLD, compute_confidence  # noqa: E402
from app.core.extraction import extract, fast_path_sufficient  # noqa: E402
from app.core.ocr_engine import read_pooled  # noqa: E402
from app.core.verification import verify  # noqa: E402


def load_first_page(path: Path) -> Image.Image:
    if path.suffix.lower() == ".pdf":
        from app.core.pdf_handler import pdf_to_images

        pages = pdf_to_images(path.read_bytes())
        if not pages:
            raise SystemExit(f"No pages found in {path}")
        return pages[0]
    return Image.open(path)


def run(path: Path, as_json: bool, show_variants: bool, receipt_id: int) -> int:
    if not path.exists():
        raise SystemExit(f"No such file: {path}")

    started = time.perf_counter()
    image = load_first_page(path)
    bundle = asyncio.run(read_pooled(image, lang="eng"))
    result = extract(bundle)
    # Mirror the production early-exit: escalate to the full pool when the
    # single-pass result does not reconcile.
    if bundle.early_exit and not fast_path_sufficient(result):
        bundle = asyncio.run(read_pooled(image, lang="eng", fast_path=False))
        result = extract(bundle)

    verification = verify(
        result.as_dict(),
        bundle.combined_text,
        reconciled=result.reconciled,
        locale_resolved=bool(result.locale and result.locale.resolved),
        mean_word_confidence=bundle.confidence,
        derived_fields=set(result.money.derived) if result.money else set(),
    )
    breakdown = compute_confidence(
        verification,
        anchor_score=bundle.primary.score,
        mean_word_confidence=bundle.confidence,
        reconciled=result.reconciled,
        locale_certainty=result.locale.certainty if result.locale else 0.0,
    )
    payload = build_callback_payload(
        receipt_id=receipt_id, fields=verification.fields, confidence=breakdown.score,
        items=result.item_scan.payload() if result.item_scan else [],
    )
    elapsed = time.perf_counter() - started

    if as_json:
        print(json.dumps({
            "callback_payload": payload.model_dump(),
            "internal_fields": verification.fields,
            "confidence": breakdown.as_dict(),
            "ocr_selection": {
                "engine": bundle.primary.engine,
                "variant": bundle.primary.variant,
                "psm": bundle.primary.psm,
                "anchor_score": round(bundle.primary.score, 4),
                "confidence": round(bundle.primary.confidence, 4),
                "candidates_evaluated": len(bundle.all_readings),
            },
            "rejected": {k: str(v) for k, v in verification.rejected.items()},
            "evidence": result.evidence,
            "seconds": round(elapsed, 2),
        }, indent=2, default=str))
        return 0

    _report(path, bundle, result, verification, breakdown, payload, elapsed,
            show_variants)
    return 0


def _report(path, bundle, result, verification, breakdown, payload, elapsed,
            show_variants) -> None:
    print(f"\n{'=' * 74}")
    print(f"  {path.name}")
    print(f"{'=' * 74}")
    print(f"  OCR       {len(bundle.all_readings)} candidates, selected "
          f"{bundle.primary.engine}/{bundle.primary.variant}/psm{bundle.primary.psm} "
          f"(anchors {bundle.primary.score:.2f}, word conf {bundle.confidence:.2f})")
    print(f"  pooled    {', '.join(r.label for r in bundle.supporting) or 'none'}")
    print(f"  elapsed   {elapsed:.1f}s")

    if show_variants:
        print("\n  variant scores")
        for reading in sorted(bundle.all_readings, key=lambda r: -r.score)[:8]:
            print(f"    {reading.engine:<10} {reading.variant:<10} psm{reading.psm:<3} "
                  f"anchors {reading.score:5.2f}  conf {reading.confidence:.2f}")

    print("\n  CALLBACK TO SERMS")
    for key, value in payload.model_dump().items():
        if key == "items":
            print(f"    {key:<22}{len(value)} row(s)")
            for row in value:
                print(f"      qty {row['quantity']:<4}{row['price']:>10.2f}  "
                      f"{row['name'][:52]}")
        else:
            print(f"    {key:<22}{value}")

    if result.item_scan:
        print("\n  ITEMS")
        print(f"    {'rows parsed':<22}{result.item_scan.count}")
        print(f"    {'reconciled':<22}{result.item_scan.reconciled}")
        print(f"    {'price basis':<22}{result.item_scan.price_basis}")
        for note in result.item_scan.notes:
            print(f"    {note}")

    print("\n  INTERNAL ONLY (stored, not transmitted)")
    for key in ("country", "currency", "net_sales", "total_sales", "service_charge",
                "tax_type", "tax_rate", "vendor_tax_id_type"):
        print(f"    {key:<22}{verification.fields.get(key)}")

    print("\n  CONFIDENCE")
    for key, value in breakdown.as_dict().items():
        print(f"    {key:<22}{value}")

    routing = "flagged (manual review)" if breakdown.score < SERMS_REVIEW_THRESHOLD \
        else "pending (accepted)"
    print(f"    {'SERMS routing':<22}{routing}")

    if verification.rejected:
        print("\n  REJECTED (ungrounded or unreconciled, withheld from the callback)")
        for key, value in verification.rejected.items():
            print(f"    {key:<22}{value}")

    if result.money and result.money.reconciliation_notes:
        print("\n  RECONCILIATION")
        for note in result.money.reconciliation_notes:
            print(f"    {note}")

    print("\n  EVIDENCE")
    for key, note in sorted(result.evidence.items()):
        print(f"    {key:<22}{note}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the pipeline on one image")
    parser.add_argument("path", type=Path)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--variants", action="store_true",
                        help="show per-variant OCR scores")
    parser.add_argument("--receipt-id", type=int, default=1)
    args = parser.parse_args()
    return run(args.path, args.json, args.variants, args.receipt_id)


if __name__ == "__main__":
    raise SystemExit(main())
