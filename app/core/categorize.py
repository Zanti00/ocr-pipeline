"""Expense category classification.

Three tiers, cheapest and most certain first:

1. vendor lexicon      - deterministic, instant, unit-testable
2. embedding similarity - the sentence-transformer is already resident for
                          duplicate detection, so this costs nothing extra
3. LLM tiebreak        - only when the top two labels are close, and only ever
                          choosing between those two

Output is restricted to SERMS' six canonical categories. SERMS calls
``ExpenseCategory::firstOrCreate(['name' => ...])``, so any novel string
permanently creates a category row - which is how 'Meals & Entertainment' and
'Office Supplies' from the original prompt polluted the taxonomy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

from app.core.schema import DEFAULT_EXPENSE_CATEGORY, EXPENSE_CATEGORIES

# Ordered: the first pattern to match wins, so specific beats general.
VENDOR_LEXICON: tuple[tuple[str, str], ...] = (
    (r"hotel|inn\b|resort|lodge|pension|airbnb|apartelle", "Accommodation"),
    (r"air(lines|ways|asia)|cebu\s*pac|philippine\s*air|\bpal\b|boarding\s*pass"
     r"|travel\s*agency|tours?\b", "Travel"),
    (r"grab|uber|taxi|lalamove|angkas|jeep|bus\s*(fare|ticket)|mrt|lrt|toll"
     r"|parking|shell|petron|caltex|seaoil|fuel|gasoline|petrol|diesel",
     "Transportation"),
    (r"jollibee|mcdonald|kfc|chowking|greenwich|mang\s*inasal|starbucks|restaurant"
     r"|resto|cafe|coffee|bakery|bakeshop|grill|diner|eatery|catering|pizza"
     r"|food|kitchen|japanese|ramen|bbq|canteen", "Meals"),
    (r"office\s*warehouse|national\s*book|bookstore|stationery|supplies|hardware"
     r"|\bmart\b|supermarket|grocer|puregold|sm\s*market|wholesaling|trading"
     r"|printing|press|photocopy", "Supplies"),
    (r"pharmacy|drug\s*store|mercury\s*drug|clinic|hospital|laboratory|diagnostic"
     r"|medical|dental", "Others"),
    (r"telecom|communications|globe|smart\b|pldt|converge|electric|meralco|water"
     r"|maynilad|internet|utility", "Others"),
)

# Short descriptions used as the reference text for embedding comparison. Bare
# labels like "Supplies" are too sparse to embed meaningfully.
CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "Meals": "restaurant food and beverage, dining, meals, catering",
    "Travel": "airfare, flights, travel tickets, tour and trip expenses",
    "Transportation": "taxi, ride hailing, fuel, parking, toll, local transport fare",
    "Accommodation": "hotel, lodging, room accommodation and overnight stay",
    "Supplies": "office supplies, stationery, hardware, groceries and materials",
    "Others": "utilities, telecommunications, medical, professional and other services",
}

EMBEDDING_MARGIN = 0.05
MIN_EMBEDDING_SIMILARITY = 0.25

Embedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]
Tiebreaker = Callable[[str, tuple[str, str]], str | None]


@dataclass
class CategoryChoice:
    category: str
    method: str
    runner_up: str | None = None
    similarity: float | None = None


def classify_category(
    vendor_name: str | None,
    item_text: str = "",
    embedder: Embedder | None = None,
    tiebreaker: Tiebreaker | None = None,
) -> CategoryChoice:
    """Assign one of SERMS' six categories."""
    haystack = " ".join(filter(None, [vendor_name or "", item_text or ""])).casefold()

    for pattern, category in VENDOR_LEXICON:
        if re.search(pattern, haystack):
            return CategoryChoice(category=category, method="lexicon")

    if embedder is not None and haystack.strip():
        choice = _by_embedding(haystack, embedder)
        if choice is not None:
            if (
                tiebreaker is not None
                and choice.runner_up is not None
                and choice.similarity is not None
                and choice.similarity < EMBEDDING_MARGIN + MIN_EMBEDDING_SIMILARITY
            ):
                picked = tiebreaker(haystack, (choice.category, choice.runner_up))
                if picked in EXPENSE_CATEGORIES:
                    return CategoryChoice(category=picked, method="llm_tiebreak",
                                          runner_up=choice.runner_up)
            return choice

    return CategoryChoice(category=DEFAULT_EXPENSE_CATEGORY, method="default")


def _by_embedding(text: str, embedder: Embedder) -> CategoryChoice | None:
    labels = list(CATEGORY_DESCRIPTIONS)
    try:
        vectors = embedder([text] + [CATEGORY_DESCRIPTIONS[label] for label in labels])
    except Exception:
        return None  # embeddings are an optional upgrade, never a hard dependency
    if not vectors or len(vectors) != len(labels) + 1:
        return None

    query, references = vectors[0], vectors[1:]
    scored = sorted(
        ((label, _cosine(query, reference)) for label, reference in zip(labels, references)),
        key=lambda pair: -pair[1],
    )
    best_label, best_score = scored[0]
    if best_score < MIN_EMBEDDING_SIMILARITY:
        return None
    return CategoryChoice(
        category=best_label,
        method="embedding",
        runner_up=scored[1][0] if len(scored) > 1 else None,
        similarity=round(best_score, 4),
    )


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)
