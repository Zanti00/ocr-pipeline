"""Vendor-name selection and expense categorisation.

Cases come from observed corpus failures. The constrained-selection rule is the
important one: a model may reorder the shortlist but never add to it.
"""

from app.core.categorize import classify_category
from app.core.extractors import CandidateMeta
from app.core.vendor import rank_candidates, select_vendor_name


def _meta(**entries: tuple[int, float, int]) -> dict[str, CandidateMeta]:
    return {
        name: CandidateMeta(index=index, confidence=confidence, occurrences=occurrences)
        for name, (index, confidence, occurrences) in entries.items()
    }


class TestVendorRanking:
    def test_business_name_beats_street_address(self):
        ranked = rank_candidates([
            "Bayan Telecommunications, Inc",
            "2/F Ever Gotesco Commonwealth, Commonwealth Ave., Quezon City",
        ])
        assert ranked[0][0] == "Bayan Telecommunications, Inc"

    def test_customer_name_is_never_proposed(self):
        # Receipt 6: 'Rey Nimfa' follows 'RECEIVED from' and is the payer.
        choice = select_vendor_name(
            ["Rey Nimfa", "Bayan Telecommunications, Inc"],
            customer_names=["Rey Nimfa"],
        )
        assert choice.name == "Bayan Telecommunications, Inc"

    def test_cumulative_entity_markers_separate_a_corrupted_twin(self):
        # Receipt 8: both lines contain 'press'; only the clean one adds 'inc'.
        ranked = rank_candidates(
            ["URITVED DAILY PRESS INGa=", "UNITED DAILY PRESS INC"],
            meta=_meta(**{
                "URITVED DAILY PRESS INGa=": (0, 0.45, 1),
                "UNITED DAILY PRESS INC": (1, 0.50, 1),
            }),
        )
        assert ranked[0][0] == "UNITED DAILY PRESS INC"

    def test_pipe_debris_is_stripped(self):
        # Receipt 7: OCR merges an unrelated column into the header line.
        ranked = rank_candidates(
            ["<aauanvor we rouowne| DETOXICARE MOLECULAR DIAGNOSTICS LABORATORY, INC"]
        )
        assert ranked[0][0] == "DETOXICARE MOLECULAR DIAGNOSTICS LABORATORY, INC"

    def test_no_plausible_candidate_returns_none(self):
        choice = select_vendor_name(["a b", "!!", "PS ee Z ie Qlva"])
        assert choice.name is None
        assert choice.method == "no_candidate"


class TestConstrainedSelection:
    def test_model_choice_is_honoured_when_shortlisted(self):
        choice = select_vendor_name(
            ["Jollibee Las Vegas", "Jollibee"], llm_choice="Jollibee"
        )
        assert choice.name == "Jollibee"
        assert choice.method == "llm_selection"

    def test_invented_name_is_rejected_and_falls_back(self):
        # A model returning something not on the page cannot be trusted with it.
        choice = select_vendor_name(
            ["Jollibee Las Vegas", "Jollibee"],
            llm_choice="McDonald's Corporation",
        )
        assert choice.name == "Jollibee Las Vegas"
        assert choice.method == "llm_rejected_off_shortlist"


class TestCategoryClassification:
    def test_known_food_vendor(self):
        result = classify_category("Jollibee Las Vegas")
        assert result.category == "Meals"
        assert result.method == "lexicon"

    def test_restaurant_keyword(self):
        assert classify_category("Yamachan Japanese Restaurant").category == "Meals"

    def test_transport_vendor(self):
        assert classify_category("Grab Philippines").category == "Transportation"
        assert classify_category("Petron Gas Station").category == "Transportation"

    def test_accommodation_vendor(self):
        assert classify_category("Seda Hotel Makati").category == "Accommodation"

    def test_wholesaler_is_supplies(self):
        result = classify_category("Okyecen Consumer Goods Wholesaling")
        assert result.category == "Supplies"

    def test_utility_falls_to_others(self):
        assert classify_category("Bayan Telecommunications, Inc.").category == "Others"

    def test_unknown_vendor_defaults_to_others(self):
        result = classify_category("Zzyzx Unclassifiable Ltd")
        assert result.category == "Others"
        assert result.method == "default"

    def test_output_is_always_a_canonical_category(self):
        from app.core.schema import EXPENSE_CATEGORIES

        for vendor in ("Jollibee", "Grab", "Seda Hotel", "", None, "???"):
            assert classify_category(vendor).category in EXPENSE_CATEGORIES


class TestCategoryTiebreak:
    def test_embedder_failure_degrades_to_default(self):
        def broken(_texts):
            raise RuntimeError("model unavailable")

        result = classify_category("Unrecognised Vendor XYZ", embedder=broken)
        assert result.category == "Others"

    def test_tiebreaker_choice_must_be_a_canonical_category(self):
        def embedder(texts):
            # Near-tie between the first two labels.
            return [[1.0, 0.0]] + [[0.9, 0.1], [0.88, 0.12]] + [[0.0, 1.0]] * 4

        def bad_tiebreaker(_text, _options):
            return "Groceries"  # not a SERMS category

        result = classify_category(
            "Unrecognised Vendor XYZ", embedder=embedder, tiebreaker=bad_tiebreaker
        )
        assert result.category != "Groceries"
