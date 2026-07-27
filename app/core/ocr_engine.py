"""Tesseract execution.

Two things changed from the original single-pass implementation:

* several preprocessing variants and page-segmentation modes are tried, and the
  reading that surfaces the most domain anchors wins. Measurement drove this:
  receipt 2's entire amount column is invisible under the default PSM 3 and
  appears in full under PSM 6.
* word bounding boxes are retained. Money values are positional - pairing a
  'TOTAL' label with the number on the same line is a geometry problem, and
  solving it in code removes the need to ask a 1.5B model to read digits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import pytesseract
from PIL import Image

from app.core.anchors import AnchorScore, score_text
from app.core.preprocessing import Variant, build_variants

logger = logging.getLogger(__name__)

DEFAULT_PSMS: tuple[int, ...] = (6, 4, 11)


@dataclass
class Word:
    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int
    block: int
    par: int
    line: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def center_y(self) -> float:
        return self.top + self.height / 2

    @property
    def line_key(self) -> tuple[int, int, int]:
        return (self.block, self.par, self.line)


@dataclass
class OcrReading:
    text: str
    words: list[Word]
    confidence: float
    variant: str
    psm: int
    anchors: AnchorScore | None = None
    lines: list[list[Word]] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.anchors.total if self.anchors else 0.0


def read_variant(image: Image.Image, psm: int, lang: str = "eng") -> OcrReading:
    """Run one OCR pass and keep both text and word geometry."""
    config = f"--psm {psm} --oem 1 --dpi 300"
    data = pytesseract.image_to_data(
        image, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )

    words: list[Word] = []
    confidences: list[float] = []
    for index, token in enumerate(data["text"]):
        if not token.strip():
            continue
        confidence = float(data["conf"][index])
        words.append(
            Word(
                text=token,
                confidence=max(confidence, 0.0) / 100.0,
                left=int(data["left"][index]),
                top=int(data["top"][index]),
                width=int(data["width"][index]),
                height=int(data["height"][index]),
                block=int(data["block_num"][index]),
                par=int(data["par_num"][index]),
                line=int(data["line_num"][index]),
            )
        )
        if confidence >= 0:
            confidences.append(confidence)

    mean = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0
    text = _render_text(words)
    reading = OcrReading(
        text=text, words=words, confidence=mean, variant="?", psm=psm,
        lines=group_lines(words),
    )
    reading.anchors = score_text(text, mean)
    return reading


def group_lines(words: Sequence[Word]) -> list[list[Word]]:
    """Group words into visual lines, ordered left to right."""
    buckets: dict[tuple[int, int, int], list[Word]] = {}
    for word in words:
        buckets.setdefault(word.line_key, []).append(word)
    lines = [sorted(group, key=lambda w: w.left) for group in buckets.values()]
    return sorted(lines, key=lambda group: (group[0].top, group[0].left))


def _render_text(words: Sequence[Word]) -> str:
    """Rebuild text with newlines at line boundaries.

    Line structure is load-bearing: label/value pairing and the proximity checks
    in anchor scoring both rely on it, and the original space-joined output
    destroyed it.
    """
    return "\n".join(
        " ".join(word.text for word in line) for line in group_lines(words)
    )


@dataclass
class OcrBundle:
    """The primary reading plus its close runners-up.

    Measurement showed the top variants score within 0.01 of each other while
    recovering *different* fields, so committing to a single winner discards
    information. Downstream stages use ``primary`` for word geometry and
    ``combined_text`` for candidate discovery and grounding. Agreement across
    independent readings also serves as corroboration: two variants producing the
    same figure is real evidence, where one model's self-reported confidence is
    not.
    """

    primary: OcrReading
    supporting: list[OcrReading] = field(default_factory=list)
    all_readings: list[OcrReading] = field(default_factory=list)

    @property
    def combined_text(self) -> str:
        seen: set[str] = set()
        blocks: list[str] = []
        for reading in [self.primary, *self.supporting]:
            if reading.text not in seen:
                seen.add(reading.text)
                blocks.append(reading.text)
        return "\n".join(blocks)

    @property
    def confidence(self) -> float:
        return self.primary.confidence

    def agreement(self, token: str) -> int:
        """How many independent readings contain this token."""
        needle = token.strip()
        if not needle:
            return 0
        return sum(1 for r in self.all_readings if needle in r.text)


def read_best(
    image: Image.Image,
    lang: str = "eng",
    psms: Sequence[int] = DEFAULT_PSMS,
    variants: Sequence[Variant] | None = None,
) -> tuple[OcrReading, list[OcrReading]]:
    """Try every variant x PSM combination and return the strongest reading.

    Returns ``(best, all_candidates)``. Roughly 15 CPU passes per receipt, which
    fits comfortably inside the async Celery budget - nothing is waiting on a
    synchronous response.
    """
    candidates: list[OcrReading] = []
    for variant in variants or build_variants(image):
        for psm in psms:
            try:
                reading = read_variant(variant.image, psm=psm, lang=lang)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("OCR failed for %s/psm%s: %s", variant.label, psm, exc)
                continue
            reading.variant = variant.label
            candidates.append(reading)

    if not candidates:
        raise RuntimeError("All OCR variants failed")

    best = max(candidates, key=lambda r: (r.score, r.confidence, len(r.text)))
    logger.info(
        "OCR selected variant=%s psm=%s score=%.3f conf=%.3f",
        best.variant, best.psm, best.score, best.confidence,
    )
    return best, candidates


def read_pooled(
    image: Image.Image,
    lang: str = "eng",
    psms: Sequence[int] = DEFAULT_PSMS,
    pool_size: int = 4,
) -> OcrBundle:
    """Read a receipt and keep the top ``pool_size`` distinct readings."""
    best, candidates = read_best(image, lang=lang, psms=psms)
    ranked = sorted(candidates, key=lambda r: (-r.score, -r.confidence))

    supporting: list[OcrReading] = []
    seen_variants: set[str] = {best.variant}
    for reading in ranked:
        if len(supporting) >= pool_size - 1:
            break
        if reading is best:
            continue
        # Prefer diversity: a different preprocessing variant is more likely to
        # recover a field the primary missed than the same variant at another PSM.
        if reading.variant in seen_variants:
            continue
        seen_variants.add(reading.variant)
        supporting.append(reading)

    return OcrBundle(primary=best, supporting=supporting, all_readings=candidates)


def extract_text(image: Image.Image, lang: str = "eng") -> tuple[str, float]:
    """Backward-compatible single-pass helper: returns ``(text, confidence)``."""
    reading = read_variant(image, psm=6, lang=lang)
    return reading.text, reading.confidence
