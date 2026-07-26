"""Confidence scoring.

The behaviour under test is deliberately the inverse of the original formula,
which gave 40% weight to fields being non-null and therefore paid the model to
guess. Honest abstention must now outscore confident invention.
"""

from app.core.confidence import SERMS_REVIEW_THRESHOLD, compute_confidence
from app.core.verification import (
    REASON_HANDWRITTEN, REASON_NO_TOTAL, Verification,
)


def _verification(populated: int, grounded: int, reasons: list[str]) -> Verification:
    result = Verification()
    result.populated_count = populated
    result.grounded_count = grounded
    result.reasons = reasons
    return result


class TestCalibration:
    def test_clean_reconciled_receipt_clears_the_consumer_threshold(self):
        breakdown = compute_confidence(
            _verification(populated=12, grounded=12, reasons=[]),
            anchor_score=7.0,
            mean_word_confidence=0.9,
            reconciled=True,
            locale_certainty=1.0,
        )
        assert breakdown.score >= SERMS_REVIEW_THRESHOLD
        assert not breakdown.flagged_by_consumer

    def test_abstention_is_always_flagged(self):
        # Handwritten receipt: few fields, all honestly grounded. A vacuous 1.0
        # grounding rate must not carry it past the review threshold.
        breakdown = compute_confidence(
            _verification(populated=2, grounded=2,
                          reasons=[REASON_HANDWRITTEN, REASON_NO_TOTAL]),
            anchor_score=6.0,
            mean_word_confidence=0.8,
            reconciled=False,
            locale_certainty=1.0,
        )
        assert breakdown.score < SERMS_REVIEW_THRESHOLD
        assert breakdown.flagged_by_consumer

    def test_score_never_exceeds_the_applicable_cap(self):
        breakdown = compute_confidence(
            _verification(populated=10, grounded=10, reasons=[REASON_NO_TOTAL]),
            anchor_score=8.0,
            mean_word_confidence=1.0,
            reconciled=True,
            locale_certainty=1.0,
        )
        assert breakdown.score <= 0.50


class TestConsistencyBeatsCompleteness:
    def test_honest_nulls_outscore_ungrounded_values(self):
        honest = compute_confidence(
            _verification(populated=5, grounded=5, reasons=[]),
            anchor_score=6.0, mean_word_confidence=0.8,
            reconciled=True, locale_certainty=1.0,
        )
        invented = compute_confidence(
            _verification(populated=10, grounded=5, reasons=[]),
            anchor_score=6.0, mean_word_confidence=0.8,
            reconciled=True, locale_certainty=1.0,
        )
        assert honest.score > invented.score

    def test_failed_reconciliation_costs_a_quarter_of_the_score(self):
        with_math = compute_confidence(
            _verification(populated=8, grounded=8, reasons=[]),
            anchor_score=7.0, mean_word_confidence=0.9,
            reconciled=True, locale_certainty=1.0,
        )
        without_math = compute_confidence(
            _verification(populated=8, grounded=8, reasons=[]),
            anchor_score=7.0, mean_word_confidence=0.9,
            reconciled=False, locale_certainty=1.0,
        )
        assert round(with_math.raw - without_math.raw, 4) == 0.25


class TestBreakdown:
    def test_breakdown_is_serialisable_for_audit(self):
        breakdown = compute_confidence(
            _verification(populated=4, grounded=4, reasons=[REASON_NO_TOTAL]),
            anchor_score=5.0, mean_word_confidence=0.7,
            reconciled=False, locale_certainty=0.8,
        )
        payload = breakdown.as_dict()
        assert payload["review_reasons"] == [REASON_NO_TOTAL]
        assert payload["score"] == breakdown.score
