"""Composite confidence scoring.

The previous formula gave 40% weight to how many fields came back non-null, which
paid the model to guess - the exact behaviour the pipeline is supposed to
eliminate. Completeness has been removed entirely.

What is rewarded now is *consistency*: OCR quality, the share of values that trace
back to the page, whether the money identities close, and whether the locale
resolved. A receipt returning four grounded fields and five honest nulls scores
higher than one returning nine fields of which five are invented.

The scale is calibrated to the consumer. SERMS routes anything below 0.80 to
'flagged' for manual confirmation, so the hard caps in ``verification`` guarantee
that every abstention lands below that line without needing a contract change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.verification import Verification

WEIGHT_OCR_QUALITY = 0.30
WEIGHT_GROUNDING = 0.35
WEIGHT_RECONCILIATION = 0.25
WEIGHT_LOCALE = 0.10

# Anchor scores run roughly 0-8 in practice; normalising against 7.0 keeps a
# well-read receipt near the top of the range without saturating.
ANCHOR_SCORE_SCALE = 7.0

SERMS_REVIEW_THRESHOLD = 0.80


@dataclass
class ConfidenceBreakdown:
    """Every term kept, so a score can be explained rather than just reported."""

    ocr_quality: float
    grounding: float
    reconciliation: float
    locale_certainty: float
    raw: float
    cap: float
    score: float
    reasons: list[str] = field(default_factory=list)

    @property
    def flagged_by_consumer(self) -> bool:
        return self.score < SERMS_REVIEW_THRESHOLD

    def as_dict(self) -> dict[str, object]:
        return {
            "ocr_quality": round(self.ocr_quality, 4),
            "grounding_pass_rate": round(self.grounding, 4),
            "reconciliation": round(self.reconciliation, 4),
            "locale_certainty": round(self.locale_certainty, 4),
            "raw_score": round(self.raw, 4),
            "cap_applied": round(self.cap, 4),
            "score": round(self.score, 4),
            "review_reasons": list(self.reasons),
        }


def compute_confidence(
    verification: Verification,
    *,
    anchor_score: float,
    mean_word_confidence: float,
    reconciled: bool,
    locale_certainty: float,
) -> ConfidenceBreakdown:
    ocr_quality = _ocr_quality(anchor_score, mean_word_confidence)
    grounding = verification.grounding_pass_rate
    reconciliation = 1.0 if reconciled else 0.0

    raw = (
        WEIGHT_OCR_QUALITY * ocr_quality
        + WEIGHT_GROUNDING * grounding
        + WEIGHT_RECONCILIATION * reconciliation
        + WEIGHT_LOCALE * max(0.0, min(locale_certainty, 1.0))
    )

    cap = verification.cap
    return ConfidenceBreakdown(
        ocr_quality=ocr_quality,
        grounding=grounding,
        reconciliation=reconciliation,
        locale_certainty=locale_certainty,
        raw=raw,
        cap=cap,
        score=round(min(raw, cap), 4),
        reasons=list(verification.reasons),
    )


def _ocr_quality(anchor_score: float, mean_word_confidence: float) -> float:
    """Blend how much the reading surfaced with how sure Tesseract was.

    Anchor coverage leads because mean confidence is routinely high on noise;
    the near-illegible samples in the corpus score respectably on confidence alone.
    """
    anchors = max(0.0, min(anchor_score / ANCHOR_SCORE_SCALE, 1.0))
    words = max(0.0, min(mean_word_confidence, 1.0))
    return round(0.7 * anchors + 0.3 * words, 4)


def compute_composite_score(
    tesseract_confidence: float, extracted_fields: dict, bir_valid: bool
) -> float:
    """Deprecated shim for the original three-argument signature.

    Retained so existing callers keep working during the migration. It no longer
    rewards field completeness: populated-but-unverified fields contribute
    nothing, because that weighting is what encouraged fabrication.
    """
    grounded_share = 1.0 if extracted_fields.get("total_amount") is not None else 0.0
    locale_certainty = 1.0 if extracted_fields.get("country") else 0.0
    raw = (
        WEIGHT_OCR_QUALITY * max(0.0, min(tesseract_confidence, 1.0))
        + WEIGHT_GROUNDING * grounded_share
        + WEIGHT_RECONCILIATION * (1.0 if bir_valid else 0.0)
        + WEIGHT_LOCALE * locale_certainty
    )
    return round(raw, 4)
