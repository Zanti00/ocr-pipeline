"""OCR execution paths used by the harness.

``baseline`` deliberately calls the *existing* production code path so the
"before" number is measured rather than asserted. Every later optimisation is
reported as a delta against it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytesseract
from PIL import Image

from app.core.ocr_engine import read_best, read_pooled, read_variant
from app.eval.legacy import legacy_preprocess_image


@dataclass
class OcrResult:
    label: str
    text: str
    confidence: float
    anchor_score: float = 0.0
    engine: str | None = None
    candidate_count: int = 0

    @property
    def char_count(self) -> int:
        return len(self.text.strip())


def run_baseline(image_path: Path) -> OcrResult:
    """The ORIGINAL production path, frozen: Gaussian blur + adaptive threshold, PSM 3."""
    with Image.open(image_path) as img:
        processed = legacy_preprocess_image(img)
        data = pytesseract.image_to_data(
            processed, lang="eng", output_type=pytesseract.Output.DICT
        )
    collected = _collect(data)
    return OcrResult(label="baseline(psm3+adaptive)", **collected)


def run_optimized(image_path: Path) -> OcrResult:
    """New path: preprocessing variants x page-segmentation modes, anchor-selected."""
    with Image.open(image_path) as img:
        best, _ = read_best(img, lang="eng")
    return OcrResult(
        label=f"optimized({best.engine}/{best.variant}/psm{best.psm})",
        text=best.text,
        confidence=best.confidence,
        anchor_score=best.score,
        engine=best.engine,
        candidate_count=1,
    )


def run_pooled(image_path: Path, pool_size: int = 4) -> OcrResult:
    """Run the production candidate-selected OCR path."""
    with Image.open(image_path) as img:
        bundle = read_pooled(img, lang="eng", pool_size=pool_size)
    labels = "+".join(
        [bundle.primary.label] + [reading.label for reading in bundle.supporting]
    )
    return OcrResult(
        label=f"candidate({labels})",
        text=bundle.combined_text,
        confidence=bundle.confidence,
        anchor_score=bundle.primary.score,
        engine=bundle.primary.engine,
        candidate_count=len(bundle.all_readings),
    )


def run_single(image_path: Path, psm: int = 6, lang: str = "eng") -> OcrResult:
    """One pass over the untouched image at a given PSM."""
    with Image.open(image_path) as img:
        reading = read_variant(img, psm=psm, lang=lang)
    return OcrResult(
        label=f"raw(psm{psm})", text=reading.text,
        confidence=reading.confidence, anchor_score=reading.score,
    )


def run_raw(image_path: Path, psm: int = 6, lang: str = "eng") -> OcrResult:
    """No preprocessing, explicit page segmentation mode."""
    config = f"--psm {psm} --dpi 300"
    with Image.open(image_path) as img:
        data = pytesseract.image_to_data(
            img, lang=lang, config=config, output_type=pytesseract.Output.DICT
        )
    return OcrResult(label=f"raw(psm{psm})", **_collect(data))


def run_variant_psm(image_path: Path, variant_label: str, psm: int) -> OcrResult:
    """Run one named preprocessing variant at one PSM, for ablation."""
    from app.core.preprocessing import build_variants

    with Image.open(image_path) as img:
        variants = {v.label: v for v in build_variants(img)}
    variant = variants.get(variant_label)
    if variant is None:
        raise KeyError(f"Unknown variant {variant_label!r}; have {sorted(variants)}")
    reading = read_variant(variant.image, psm=psm)
    return OcrResult(
        label=f"{variant_label}/psm{psm}", text=reading.text,
        confidence=reading.confidence, anchor_score=reading.score,
    )


def _collect(data: dict) -> dict:
    words, confidences = [], []
    for index, word in enumerate(data["text"]):
        if not word.strip():
            continue
        words.append(word)
        confidence = int(data["conf"][index])
        if confidence >= 0:
            confidences.append(confidence)
    mean = sum(confidences) / len(confidences) if confidences else 0.0
    return {"text": " ".join(words), "confidence": mean / 100.0}
