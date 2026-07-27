"""Receipt processing pipeline.

Stage order is load-bearing:

    download -> page selection -> pooled OCR (variants x PSMs)
      -> locale detection            (must precede number parsing)
      -> deterministic extraction    (money by geometry, TIN roles, dates)
      -> optional model assist       (vendor selection, category tiebreak)
      -> verification gate           (null anything ungrounded)
      -> confidence with hard caps   (calibrated to the consumer's 0.80 threshold)
      -> validated callback

The language model no longer reads the receipt. It selects from closed candidate
lists, and every optional assist fails soft: a model timeout degrades the result
to the deterministic answer instead of failing the job.
"""

from __future__ import annotations

import logging
from io import BytesIO

import httpx
from PIL import Image

from app.core.callback import send_callback
from app.core.confidence import ConfidenceBreakdown, compute_confidence
from app.core.extraction import Extraction, extract
from app.core.ocr_engine import OcrBundle, read_pooled
from app.core.pdf_handler import pdf_to_images
from app.core.verification import Verification, verify
from app.api.schemas.ocr import build_callback_payload
from app.db.models import ReceiptEmbedding
from app.db.mongodb import MongoDBClient
from app.db.postgres import AsyncSessionLocal
from app.embeddings.generator import EmbeddingGenerator
from app.llm.factory import create_provider

logger = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT = 30.0
MAX_PDF_PAGES = 5


async def download_file(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()
        return response.content


def load_images(file_bytes: bytes, file_url: str) -> list[Image.Image]:
    if file_url.lower().split("?")[0].endswith(".pdf"):
        return pdf_to_images(file_bytes)[:MAX_PDF_PAGES]
    return [Image.open(BytesIO(file_bytes))]


async def process_receipt(
    job_id: str,
    receipt_id: int,
    file_url: str,
    callback_url: str,
    source_service: str,
) -> None:
    logger.info("Starting OCR pipeline for job %s", job_id)

    try:
        await MongoDBClient.update_job_status(job_id, "processing")

        file_bytes = await download_file(file_url)
        images = load_images(file_bytes, file_url)
        if not images:
            raise ValueError("No images extracted from file")

        bundle, extraction = await _read_and_extract(images)

        verification = verify(
            extraction.as_dict(),
            bundle.combined_text,
            reconciled=extraction.reconciled,
            locale_resolved=bool(extraction.locale and extraction.locale.resolved),
            mean_word_confidence=bundle.confidence,
            tax_id_ambiguous=_tax_id_ambiguous(extraction),
            derived_fields=set(extraction.money.derived) if extraction.money else set(),
        )
        breakdown = compute_confidence(
            verification,
            anchor_score=bundle.primary.score,
            mean_word_confidence=bundle.confidence,
            reconciled=extraction.reconciled,
            locale_certainty=extraction.locale.certainty if extraction.locale else 0.0,
        )

        await _store_embedding(receipt_id, source_service, bundle.combined_text)
        await MongoDBClient.update_job_status(
            job_id, "completed",
            _job_document(bundle, extraction, verification, breakdown),
        )

        payload = build_callback_payload(
            receipt_id=receipt_id,
            fields=verification.fields,
            confidence=breakdown.score,
            status="completed",
            items=extraction.item_scan.payload() if extraction.item_scan else [],
        )
        logger.info(
            "Job %s complete: variant=%s psm=%s score=%.3f reasons=%s",
            job_id, bundle.primary.variant, bundle.primary.psm, breakdown.score,
            verification.reasons or "none",
        )

        if not await send_callback(callback_url, payload.model_dump()):
            raise RuntimeError(f"Callback failed to {callback_url}")

    except Exception as exc:
        logger.error("Pipeline failed for job %s: %s", job_id, exc, exc_info=True)
        await MongoDBClient.update_job_status(job_id, "failed", {"error": str(exc)})
        await send_callback(
            callback_url,
            {"receipt_id": receipt_id, "status": "failed", "error": str(exc)},
        )
        raise


async def _read_and_extract(images: list[Image.Image]) -> tuple[OcrBundle, Extraction]:
    """Read the strongest page and extract from it.

    Multi-page PDFs are scored per page and the best-reading page wins. The
    previous implementation used ``images[0]`` and silently discarded the rest,
    which loses the receipt whenever page one is a cover sheet.
    """
    best: tuple[OcrBundle, Extraction] | None = None

    for index, image in enumerate(images):
        bundle = read_pooled(image, lang="eng")
        extraction = await _extract_with_assists(bundle)
        rank = (
            1 if extraction.reconciled else 0,
            1 if extraction.fields.get("total_amount") is not None else 0,
            bundle.primary.score,
        )
        if best is None or rank > _rank_of(best):
            best = (bundle, extraction)
        logger.debug("Page %s scored %s", index, rank)

    assert best is not None  # images is non-empty by the time we get here
    return best


def _rank_of(pair: tuple[OcrBundle, Extraction]) -> tuple[int, int, float]:
    bundle, extraction = pair
    return (
        1 if extraction.reconciled else 0,
        1 if extraction.fields.get("total_amount") is not None else 0,
        bundle.primary.score,
    )


async def _extract_with_assists(bundle: OcrBundle) -> Extraction:
    """Deterministic extraction, then optional model and embedding assists.

    Runs the deterministic pass first so a shortlist exists to constrain the model
    with. Both assists are best-effort: any failure leaves the deterministic
    result in place.
    """
    baseline = extract(bundle)

    vendor_choice = await _ask_model_for_vendor(baseline)
    embedder = _embedder()

    if vendor_choice is None and embedder is None:
        return baseline

    return extract(
        bundle,
        llm_vendor_choice=vendor_choice,
        embedder=embedder,
        category_tiebreaker=None,
    )


async def _ask_model_for_vendor(baseline: Extraction) -> str | None:
    shortlist = baseline.vendor_choice.shortlist if baseline.vendor_choice else []
    if len(shortlist) < 2:
        return None  # nothing to disambiguate
    try:
        provider = create_provider()
        return await provider.select_vendor_name(
            shortlist, baseline.vendor_candidates.customer_names
        )
    except Exception as exc:
        logger.warning("Vendor selection unavailable, keeping deterministic pick: %s", exc)
        return None


def _embedder():
    """Return an embedding callable, or None when the model is unavailable."""
    try:
        model = EmbeddingGenerator.get_model()
    except Exception as exc:
        logger.debug("Embeddings unavailable for categorisation: %s", exc)
        return None

    def encode(texts):
        return [vector.tolist() for vector in model.encode(list(texts))]

    return encode


def _tax_id_ambiguous(extraction: Extraction) -> bool:
    """Several tax ids and none carrying an explicit vendor marker."""
    candidates = extraction.tax_id_candidates
    if any(candidate.role == "vendor" for candidate in candidates):
        return False
    return len([c for c in candidates if c.role == "unknown"]) > 1


async def _store_embedding(receipt_id: int, source_service: str, text: str) -> None:
    """Persist a text embedding for duplicate detection.

    Deliberately non-fatal: duplicate detection is a separate concern, and losing
    an embedding should not cost a successfully extracted receipt.
    """
    if not text.strip():
        return
    try:
        embedding = EmbeddingGenerator.generate(text)
        async with AsyncSessionLocal() as session:
            session.add(
                ReceiptEmbedding(
                    receipt_id=receipt_id,
                    source_service=source_service,
                    embedding=embedding,
                    receipt_text=text,
                )
            )
            await session.commit()
    except Exception as exc:
        logger.warning("Embedding storage failed for receipt %s: %s", receipt_id, exc)


def _job_document(
    bundle: OcrBundle,
    extraction: Extraction,
    verification: Verification,
    breakdown: ConfidenceBreakdown,
) -> dict:
    """The audit record.

    Holds the full internal model, not just the subset SERMS accepts. Laravel's
    ``validated()`` discards undeclared keys, so net_sales, service_charge,
    country, currency, review reasons and rejected values are only useful if we
    keep them - and they are what makes a score explainable after the fact.
    """
    return {
        "raw_ocr_text": bundle.combined_text,
        "tesseract_confidence": round(bundle.confidence, 4),
        "composite_confidence_score": breakdown.score,
        "confidence_breakdown": breakdown.as_dict(),
        "extracted_data": verification.fields,
        "ocr_selection": {
            "variant": bundle.primary.variant,
            "psm": bundle.primary.psm,
            "anchor_score": round(bundle.primary.score, 4),
            "pooled_variants": [r.variant for r in bundle.supporting],
            "candidates_evaluated": len(bundle.all_readings),
        },
        "verification": {
            "needs_manual_review": verification.needs_manual_review,
            "review_reasons": verification.reasons,
            "rejected_fields": {k: str(v) for k, v in verification.rejected.items()},
            "grounding_pass_rate": round(verification.grounding_pass_rate, 4),
        },
        "evidence": extraction.evidence,
        "items": {
            "rows_found": extraction.item_scan.count if extraction.item_scan else 0,
            "reconciled": bool(extraction.item_scan and extraction.item_scan.reconciled),
            "price_basis": extraction.item_scan.price_basis if extraction.item_scan
            else "none",
            "notes": extraction.item_scan.notes if extraction.item_scan else [],
            # Rows found but withheld are kept here: useful for auditing why a
            # receipt shipped without items, and never transmitted.
            "candidates": [
                {"name": item.full_name, "quantity": item.quantity,
                 "price": item.price, "line": item.line_text}
                for item in (extraction.item_scan.items if extraction.item_scan else [])
            ],
        },
        "reconciliation": {
            "reconciled": extraction.reconciled,
            "notes": extraction.money.reconciliation_notes if extraction.money else [],
            "derived_fields": sorted(extraction.money.derived) if extraction.money else [],
        },
        "locale": {
            "country": extraction.fields.get("country"),
            "currency": extraction.fields.get("currency"),
            "evidence": extraction.locale.evidence if extraction.locale else [],
        },
    }
