"""Ground-truth loading and the field registry that drives scoring.

Design notes
------------
A ground-truth file distinguishes three states for every field, which is the
whole point of the format:

* present in ``expected`` with a value  -> we know the correct answer
* present in ``expected`` as ``null``   -> we know the field is genuinely absent
* listed in ``unknown``                 -> we cannot verify it, so it is EXCLUDED
                                           from scoring rather than guessed

Inventing ground truth for the unreadable handwritten samples would make the
accuracy number meaningless, so unverifiable fields are skipped explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = _ROOT / "tests" / "fixtures" / "receipts"
IMAGES_DIR = _ROOT / "docs" / "receipts"
SYNTHETIC_DIR = _ROOT / "tests" / "fixtures" / "synthetic"

# Named corpora: (ground-truth directory, image directory).
#
# 'real' is nine sample images, only two of which are machine printed - too few to
# support an accuracy claim. 'synthetic' is generated with exact known values, so
# it can be sized properly and can exercise failure paths on demand.
CORPORA: dict[str, tuple[Path, Path]] = {
    "real": (FIXTURES_DIR, IMAGES_DIR),
    "synthetic": (SYNTHETIC_DIR, SYNTHETIC_DIR),
}


class FieldKind(str, Enum):
    """How a field must be compared. Money cannot be compared like text."""

    MONEY = "money"
    RATE = "rate"
    TAX_ID = "tax_id"
    DATE = "date"
    FUZZY_TEXT = "fuzzy_text"
    EXACT = "exact"


class Gate(str, Enum):
    """Accuracy threshold class. Consequence of error differs per field."""

    CRITICAL = "critical"      # financial / compliance impact -> 95%
    STANDARD = "standard"      # 90%
    UNTRACKED = "untracked"    # reported but not gated (subjective)


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: FieldKind
    gate: Gate
    transcribed: bool = True
    """True if the value must be READ off the page.

    Derived fields (country, currency, tax_type, ...) are inferred from context
    or computed arithmetically, so asking whether they "appear in the OCR text"
    is meaningless. They are excluded from recoverability so the ceiling figure
    reflects only what OCR actually has to transcribe.
    """


# Single source of truth for the extraction schema agreed in Q3/Q8.
FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec("vendor_name", FieldKind.FUZZY_TEXT, Gate.STANDARD),
    FieldSpec("country", FieldKind.EXACT, Gate.STANDARD, transcribed=False),
    FieldSpec("currency", FieldKind.EXACT, Gate.STANDARD, transcribed=False),
    FieldSpec("vendor_tax_id", FieldKind.TAX_ID, Gate.CRITICAL),
    FieldSpec("vendor_tax_id_type", FieldKind.EXACT, Gate.STANDARD, transcribed=False),
    FieldSpec("transaction_date", FieldKind.DATE, Gate.CRITICAL),
    FieldSpec("net_sales", FieldKind.MONEY, Gate.STANDARD),
    FieldSpec("tax_amount", FieldKind.MONEY, Gate.STANDARD),
    FieldSpec("tax_type", FieldKind.EXACT, Gate.STANDARD, transcribed=False),
    FieldSpec("tax_rate", FieldKind.RATE, Gate.UNTRACKED, transcribed=False),
    FieldSpec("total_sales", FieldKind.MONEY, Gate.STANDARD),
    FieldSpec("service_charge", FieldKind.MONEY, Gate.STANDARD),
    FieldSpec("total_amount", FieldKind.MONEY, Gate.CRITICAL),
    FieldSpec("vat_classification", FieldKind.EXACT, Gate.STANDARD, transcribed=False),
    FieldSpec("invoice_number", FieldKind.EXACT, Gate.STANDARD),
    FieldSpec("expense_category", FieldKind.EXACT, Gate.UNTRACKED, transcribed=False),
)

FIELD_BY_NAME: dict[str, FieldSpec] = {spec.name: spec for spec in FIELD_SPECS}

MONEY_FIELDS: tuple[str, ...] = tuple(
    spec.name for spec in FIELD_SPECS if spec.kind is FieldKind.MONEY
)


@dataclass
class GroundTruth:
    image: str
    print_type: str
    truth_source: str
    expect_manual_review: bool
    expected: dict[str, Any]
    expected_items: list[dict[str, Any]] | None
    unknown: set[str]
    not_expected_values: dict[str, list[str]]
    also_acceptable: dict[str, list[str]]
    """Additional values that are equally correct.

    Some fields have more than one defensible answer. A receipt issued by a
    franchisee prints both the registered legal entity and the branch trade name,
    and 'the vendor' is legitimately either. Recording both is more honest than
    declaring one wrong or tuning the extractor until a preference wins.
    """
    notes: str
    source: str = "real_sample"
    script: str = "latin"
    extra: dict[str, Any] = field(default_factory=dict)
    image_dir: Path = IMAGES_DIR

    @property
    def image_path(self) -> Path:
        return self.image_dir / self.image

    @property
    def degradation(self) -> str:
        return str(self.extra.get("degradation", "none"))

    @property
    def template(self) -> str:
        return str(self.extra.get("template", "real"))

    @property
    def expect_reconciliation_failure(self) -> bool:
        """Deliberately inconsistent arithmetic, used to test the gate fires."""
        return bool(self.extra.get("expect_reconciliation_failure", False))

    @property
    def is_handwritten(self) -> bool:
        return self.print_type == "handwritten_form"

    def scored_fields(self) -> list[FieldSpec]:
        """Fields we can legitimately score: known, and not marked unknown."""
        return [
            spec
            for spec in FIELD_SPECS
            if spec.name not in self.unknown and spec.name in self.expected
        ]

    def truth_for(self, name: str) -> Any:
        return self.expected.get(name)


def load_ground_truth(
    fixtures_dir: Path | None = None,
    images_dir: Path | None = None,
    corpus: str | None = None,
) -> list[GroundTruth]:
    if corpus is not None:
        if corpus not in CORPORA:
            raise ValueError(f"Unknown corpus {corpus!r}; have {sorted(CORPORA)}")
        fixtures_dir, images_dir = CORPORA[corpus]

    directory = fixtures_dir or FIXTURES_DIR
    image_directory = images_dir or IMAGES_DIR
    files = sorted(directory.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No ground-truth fixtures found in {directory}")
    return [
        _parse(json.loads(path.read_text(encoding="utf-8")), image_directory)
        for path in files
    ]


def _parse(raw: dict[str, Any], image_dir: Path = IMAGES_DIR) -> GroundTruth:
    known_keys = {
        "image", "source", "script", "print_type", "truth_source",
        "expect_manual_review", "expected", "expected_items", "unknown",
        "not_expected_values", "also_acceptable", "notes",
    }
    return GroundTruth(
        image=raw["image"],
        source=raw.get("source", "real_sample"),
        script=raw.get("script", "latin"),
        print_type=raw["print_type"],
        truth_source=raw.get("truth_source", "unspecified"),
        expect_manual_review=bool(raw.get("expect_manual_review", False)),
        expected=raw.get("expected", {}),
        expected_items=raw.get("expected_items"),
        unknown=set(raw.get("unknown", [])),
        not_expected_values=raw.get("not_expected_values", {}),
        also_acceptable=raw.get("also_acceptable", {}),
        notes=raw.get("notes", ""),
        image_dir=image_dir,
        extra={k: v for k, v in raw.items() if k not in known_keys},
    )
