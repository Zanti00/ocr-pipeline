"""Synthetic corpus generation.

The generator produces its own ground truth, so a bug here silently corrupts every
accuracy figure measured against it. These tests check the properties the corpus
is supposed to guarantee - above all that the arithmetic closes by construction,
since that is what makes the reconciliation gate testable.
"""

from __future__ import annotations

import json
import random

import pytest

from app.core.numbers import digits_only
from app.eval.synthetic.corpus import _ground_truth, generate_corpus
from app.eval.synthetic.degrade import DEGRADATIONS, degrade
from app.eval.synthetic.render import render_receipt
from app.eval.synthetic.spec import TEMPLATES, build_spec

TOLERANCE = 0.02


def _specs(count: int = 60, seed: int = 1):
    rng = random.Random(seed)
    return [build_spec(index, rng) for index in range(count)]


class TestArithmeticConsistency:
    @pytest.mark.parametrize("template", TEMPLATES)
    def test_identities_close_for_every_template(self, template):
        rng = random.Random(99)
        for index in range(12):
            spec = build_spec(index, rng, template=template)

            if spec.tax_amount is not None:
                assert abs(spec.net_sales + spec.tax_amount - spec.total_sales) <= TOLERANCE
            else:
                assert abs(spec.net_sales - spec.total_sales) <= TOLERANCE

            expected_total = spec.total_sales + (spec.service_charge or 0.0)
            assert abs(expected_total - spec.total_amount) <= TOLERANCE

    def test_ph_vat_is_twelve_percent_of_net(self):
        rng = random.Random(7)
        for index in range(12):
            spec = build_spec(index, rng, template="ph_vat_or")
            assert spec.tax_amount is not None
            assert abs(spec.net_sales * 0.12 - spec.tax_amount) <= 0.02

    def test_non_vat_invoices_carry_no_tax(self):
        rng = random.Random(5)
        spec = build_spec(0, rng, template="ph_nonvat_si")
        assert spec.tax_amount is None
        assert spec.vat_classification == "non-vat"


class TestCoverage:
    def test_both_branch_code_widths_are_generated(self):
        # The original validator rejected five-digit branch codes outright, so the
        # corpus must contain them or the fix goes untested.
        widths = {
            len(digits_only(spec.vendor_tax_id))
            for spec in _specs()
            if spec.vendor_tax_id
        }
        assert widths == {12, 14}

    def test_some_receipts_carry_a_customer_tax_id(self):
        # Vendor/customer disambiguation is the highest-consequence extraction
        # decision, so it must be exercised rather than assumed.
        with_customer = [s for s in _specs() if s.customer_tax_id]
        assert 5 <= len(with_customer) <= 55

    def test_multiple_locales_are_generated(self):
        assert {spec.country for spec in _specs()} >= {"PH", "US"}

    def test_date_formats_vary(self):
        assert len({spec.date_text for spec in _specs()}) > 10


class TestGroundTruthDocuments:
    def test_document_matches_the_spec(self):
        rng = random.Random(3)
        spec = build_spec(0, rng, template="ph_vat_or")
        document = _ground_truth(spec, "clean", "syn-0000.jpg")

        assert document["print_type"] == "machine_printed"
        assert document["expected"]["total_amount"] == spec.total_amount
        assert document["expected"]["vendor_tax_id"] == spec.vendor_tax_id
        assert document["expected"]["transaction_date"] == spec.transaction_date.isoformat()
        # Items ARE scored now: the generator knows every row exactly, so the
        # corpus is the only place item accuracy can be measured at a useful sample
        # size. tax_rate stays unknown because it is derived, not printed.
        assert "items" not in document["unknown"]
        assert "tax_rate" in document["unknown"]
        assert len(document["expected_items"]) == len(spec.items)

    def test_customer_tax_id_is_recorded_as_a_trap(self):
        rng = random.Random(2)
        spec = build_spec(0, rng, template="ph_vat_or")
        spec.customer_tax_id = "201-841-917-000"
        spec.customer_name = "Scientific Biotech Specialties, Inc."
        document = _ground_truth(spec, "clean", "syn-0000.jpg")
        assert document["not_expected_values"]["vendor_tax_id"] == ["201-841-917-000"]

    def test_corrupted_receipts_expect_review(self):
        rng = random.Random(4)
        spec = build_spec(0, rng, template="ph_vat_or")
        spec.corrupt_arithmetic = True
        document = _ground_truth(spec, "clean", "syn-0000.jpg")
        assert document["expect_manual_review"] is True
        assert document["expect_reconciliation_failure"] is True


class TestRenderingAndDegradation:
    @pytest.mark.parametrize("template", TEMPLATES)
    def test_every_template_renders(self, template):
        rng = random.Random(11)
        image = render_receipt(build_spec(0, rng, template=template))
        assert image.width > 200 and image.height > 200

    @pytest.mark.parametrize("tier", DEGRADATIONS)
    def test_every_degradation_tier_produces_an_image(self, tier):
        rng = random.Random(13)
        image = render_receipt(build_spec(0, rng, template="ph_vat_or"))
        result = degrade(image, tier, rng)
        assert result.width > 100 and result.height > 100

    def test_severe_degradation_actually_changes_the_image(self):
        rng = random.Random(17)
        original = render_receipt(build_spec(0, rng, template="ph_pos"))
        assert degrade(original, "severe", rng).size != original.size


class TestReproducibility:
    def test_same_seed_produces_identical_ground_truth(self, tmp_path):
        first = tmp_path / "a"
        second = tmp_path / "b"
        generate_corpus(first, count=4, corrupted_count=1, seed=42)
        generate_corpus(second, count=4, corrupted_count=1, seed=42)

        for path in sorted(first.glob("*.json")):
            mirror = second / path.name
            assert json.loads(path.read_text()) == json.loads(mirror.read_text())

    def test_generation_writes_an_image_per_document(self, tmp_path):
        stats = generate_corpus(tmp_path, count=5, corrupted_count=2, seed=8)
        assert stats.written == 7
        assert len(list(tmp_path.glob("*.jpg"))) == 7
        assert len(list(tmp_path.glob("*.json"))) == 7
