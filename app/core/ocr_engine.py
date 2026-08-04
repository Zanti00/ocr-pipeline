"""Receipt OCR candidate generation and selection.

The OCR stage runs PaddleOCR once and Tesseract across preprocessing variants and
page-segmentation modes. Their readings are scored using receipt-aware anchors;
the strongest existing candidate becomes primary while all successful readings
remain available for downstream reconciliation.
"""

from __future__ import annotations

import logging
import re
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

import pytesseract
from PIL import Image

from app.core.anchors import AnchorScore, score_text
from app.core.preprocessing import Variant, build_variants

logger = logging.getLogger(__name__)

# Paddle is optional at import time so the service can still start and use the
# Tesseract fallback when an image has not yet been rebuilt with PaddleOCR.
try:
    from paddleocr import PaddleOCR
except Exception as exc:  # pragma: no cover - depends on the runtime image
    PaddleOCR = None  # type: ignore[assignment]
    logger.warning("PaddleOCR unavailable; using Tesseract fallback: %s", exc)

paddle_ocr = None
_paddle_init_attempted = False

# Global Thread Pool Executor for CPU-bound OCR and preprocessing tasks.
# Capped to avoid resource starvation under concurrent celery jobs.
MAX_OCR_WORKERS = max(1, min(os.cpu_count() or 4, 6))
ocr_executor = ThreadPoolExecutor(max_workers=MAX_OCR_WORKERS, thread_name_prefix="ocr")


def _get_paddle_ocr():
    """Initialize Paddle lazily so a missing model cannot break module import."""
    global _paddle_init_attempted, paddle_ocr
    if paddle_ocr is not None:
        return paddle_ocr
    if _paddle_init_attempted:
        return None
    _paddle_init_attempted = True
    if PaddleOCR is None:
        return None
    try:
        try:
            paddle_ocr = PaddleOCR(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
            )
        except (TypeError, ValueError):
            # PaddleOCR 2.x does not know the 3.x pipeline options.
            paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")
    except Exception as exc:  # pragma: no cover - depends on runtime/model
        logger.warning("PaddleOCR initialization failed; using Tesseract fallback: %s", exc)
    return paddle_ocr

DEFAULT_PSMS: tuple[int, ...] = (6, 4, 11)

# Early-exit fast path: one cheap Tesseract pass over the untouched image can
# satisfy most receipts. When its anchor score is too weak, a second cheap pass
# over the ``flat`` rendering at PSM 11 gets one more try before the full
# Tesseract pool (5 variants x PSMs, see OCR_POOL_VARIANTS/OCR_POOL_PSMS) plus
# Paddle runs, so hard cases keep the accuracy of the pooled path while easy
# ones finish in seconds.
FAST_PATH_PSM = 6
FAST_PATH_ANCHOR_SCORE = 6.0
FAST_PATH_MAX_EDGE = 1600


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
    engine: Literal["paddle", "tesseract"] = "tesseract"
    anchors: AnchorScore | None = None
    lines: list[list[Word]] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.anchors.total if self.anchors else 0.0

    @property
    def label(self) -> str:
        """Stable, engine-aware label for logs and audit metadata."""
        return f"{self.engine}/{self.variant}/psm{self.psm}"


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
        engine="tesseract", lines=group_lines(words),
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
    early_exit: bool = False

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


async def read_best(
    image: Image.Image,
    lang: str = "eng",
    psms: Sequence[int] = DEFAULT_PSMS,
    variants: Sequence[Variant] | None = None,
) -> tuple[OcrReading, list[OcrReading]]:
    """Try every configured Tesseract variant x PSM combination concurrently.

    Runs variants in the global ocr_executor to parallelize CPU passes. When
    ``variants`` is omitted, the full built-in variant set is filtered down to
    ``settings.ocr_pool_variants`` ("all" keeps everything). The pool size is
    therefore tunable via ``OCR_POOL_VARIANTS``/``OCR_POOL_PSMS`` without code
    changes, letting operators trade accuracy for speed on hard receipts.
    """
    candidates: list[OcrReading] = []
    loop = asyncio.get_running_loop()

    def _build():
        if variants is not None:
            return variants
        from app.config import settings

        labels = [
            label.strip()
            for label in settings.ocr_pool_variants.split(",")
            if label.strip()
        ]
        built = build_variants(image)
        if "all" in labels:
            return built
        return [v for v in built if v.label in labels]
        
    actual_variants = await loop.run_in_executor(ocr_executor, _build)

    def _run_variant(v_img, v_label, psm_val):
        reading = read_variant(v_img, psm=psm_val, lang=lang)
        reading.variant = v_label
        return reading

    tasks = []
    for variant in actual_variants:
        for psm in psms:
            tasks.append(
                loop.run_in_executor(ocr_executor, _run_variant, variant.image, variant.label, psm)
            )

    # Failure strategy: if one variant fails, we ignore it and continue.
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            logger.warning("OCR failed for a variant: %s", res)
        else:
            candidates.append(res)

    if not candidates:
        raise RuntimeError("All OCR variants failed")

    best = select_primary(candidates)
    logger.info(
        "Tesseract best candidate variant=%s psm=%s score=%.3f conf=%.3f",
        best.variant, best.psm, best.score, best.confidence,
    )
    return best, candidates


def rank_candidates(candidates: Sequence[OcrReading]) -> list[OcrReading]:
    """Rank OCR candidates without adding another OCR or extraction pass.

    Anchor coverage is the primary signal. Confidence and text length only break
    ties because engine confidence values are not directly calibrated across
    PaddleOCR and Tesseract.
    """
    return sorted(
        (candidate for candidate in candidates if candidate.text.strip()),
        key=lambda candidate: (
            -candidate.score,
            -candidate.confidence,
            -len(candidate.text),
        ),
    )


def select_primary(candidates: Sequence[OcrReading]) -> OcrReading:
    """Select the strongest available OCR result from either engine."""
    ranked = rank_candidates(candidates)
    if not ranked:
        raise RuntimeError("All OCR candidates were empty")
    return ranked[0]


def _box_bounds(box) -> tuple[int, int, int, int]:
    """Normalize Paddle's polygon or ``[left, top, right, bottom]`` box."""
    points = np.asarray(box)
    if points.ndim == 1 and len(points) == 4:
        left, top, right, bottom = points
    else:
        points = points.reshape(-1, 2)
        left = points[:, 0].min()
        top = points[:, 1].min()
        right = points[:, 0].max()
        bottom = points[:, 1].max()
    return int(left), int(top), int(right), int(bottom)


def _append_paddle_line(
    words: list[Word],
    text: str,
    confidence: float,
    box,
    line_number: int,
    word_texts: Sequence[str] | None = None,
    word_boxes=None,
) -> None:
    """Append one Paddle line, using native word boxes where available."""
    text = str(text).strip()
    if not text:
        return
    confidence = max(0.0, min(float(confidence), 1.0))

    if word_texts and word_boxes is not None and len(word_texts) == len(word_boxes):
        for token, token_box in zip(word_texts, word_boxes):
            left, top, right, bottom = _box_bounds(token_box)
            words.append(Word(
                text=str(token),
                confidence=confidence,
                left=left,
                top=top,
                width=max(1, right - left),
                height=max(1, bottom - top),
                block=0,
                par=0,
                line=line_number,
            ))
        return

    left, top, right, bottom = _box_bounds(box)
    line_width = max(1, right - left)
    for match in re.finditer(r"\S+", text):
        token_left = left + round(line_width * match.start() / len(text))
        token_right = left + round(line_width * match.end() / len(text))
        words.append(Word(
            text=match.group(),
            confidence=confidence,
            left=token_left,
            top=top,
            width=max(1, token_right - token_left),
            height=max(1, bottom - top),
            block=0,
            par=0,
            line=line_number,
        ))


def read_paddle(image: Image.Image) -> OcrReading:
    """Run PaddleOCR across its legacy 2.x and pipeline 3.x APIs."""
    engine = _get_paddle_ocr()
    if engine is None:
        raise RuntimeError("PaddleOCR is not initialized")

    img_array = np.array(image.convert("RGB"))
    words: list[Word] = []
    confidences: list[float] = []

    # Tall/narrow receipts (aspect >= 2.0) get a smaller detector resize limit:
    # the default 960px side produces a ~3820px internal image and Paddle spends
    # a minute per call on it. 512 was measured to cut that time by ~30% with
    # identical line/character output on long grocery receipts, so the reduced
    # detector context does not lose lines.
    height, width = image.size
    det_limit = 512 if height >= 2.0 * width else None

    if hasattr(engine, "predict"):
        # PaddleOCR 3.x returns OCRResult mappings. Disabling optional document
        # stages avoids the CPU PIR/oneDNN path that is not supported by all
        # Paddle runtime builds and is unnecessary for receipt crops.
        kwargs = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "return_word_box": True,
        }
        if det_limit is not None:
            kwargs["text_det_limit_side_len"] = det_limit
        result = engine.predict(img_array, **kwargs)
        first = result[0] if result else None
        if first is not None and hasattr(first, "get"):
            texts = first.get("rec_texts", [])
            scores = first.get("rec_scores", [])
            boxes = first.get("rec_boxes", [])
            word_texts = first.get("text_word", [])
            word_boxes = first.get("text_word_boxes", [])
            for idx, text in enumerate(texts):
                if idx >= len(boxes):
                    continue
                score = scores[idx] if idx < len(scores) else 0.0
                line_words = word_texts[idx] if idx < len(word_texts) else None
                line_boxes = word_boxes[idx] if idx < len(word_boxes) else None
                _append_paddle_line(
                    words, text, score, boxes[idx], idx, line_words, line_boxes
                )
                confidences.append(max(0.0, min(float(score), 1.0)))
    else:
        # PaddleOCR 2.x returns ``[[[polygon, (text, confidence)], ...]]``.
        img_bgr = img_array[:, :, ::-1]
        result = engine.ocr(img_bgr, cls=True)
        for idx, line in enumerate(result[0] if result and result[0] else []):
            try:
                box, (text, confidence) = line
            except (TypeError, ValueError):
                logger.warning("PaddleOCR returned an unsupported line shape")
                continue
            _append_paddle_line(words, text, confidence, box, idx)
            confidences.append(max(0.0, min(float(confidence), 1.0)))

    if not words:
        # A successful model call with no detections is not successful OCR. Let
        # read_pooled run the complete Tesseract search instead.
        raise RuntimeError("PaddleOCR returned no text")

    mean = sum(confidences) / len(confidences)
    text_content = _render_text(words)
    reading = OcrReading(
        text=text_content,
        words=words,
        confidence=mean,
        variant="source",
        psm=0,
        engine="paddle",
        lines=group_lines(words),
    )
    reading.anchors = score_text(text_content, mean)
    return reading


def _supporting_readings(
    candidates: Sequence[OcrReading],
    primary: OcrReading,
    pool_size: int,
) -> list[OcrReading]:
    """Choose strong, diverse readings excluding the selected primary.

    Readings that score far below the primary are excluded outright: on
    tall/narrow receipts Tesseract degrades to near-zero anchor scores while
    Paddle reads the slip cleanly, and concatenating that garbage into
    ``combined_text`` pollutes the LLM prompt (wrong vendor, wrong country)
    and the item/region scanners. A floor of half the primary's score keeps
    genuinely independent second opinions while dropping engine failure.
    """
    supporting: list[OcrReading] = []
    seen_variants: set[str] = {primary.variant}
    seen_text: set[str] = {primary.text}
    ranked = rank_candidates(candidates)
    target = max(0, pool_size - 1)
    floor = primary.score / 2.0

    # Prefer different preprocessing variants/engines first, then fill remaining
    # slots with another candidate. Identical OCR text is pooled only once.
    for distinct_variants_only in (True, False):
        for reading in ranked:
            if len(supporting) >= target:
                return supporting
            if reading is primary or reading.text in seen_text:
                continue
            if reading.score < floor:
                continue
            if distinct_variants_only and reading.variant in seen_variants:
                continue
            supporting.append(reading)
            seen_variants.add(reading.variant)
            seen_text.add(reading.text)
    return supporting


def read_fast(image: Image.Image, lang: str = "eng") -> OcrReading:
    """Single cheap Tesseract pass for the early-exit path.

    Runs the standard best-effort rendering once at one PSM. The full 16-pass pool
    is only worth its cost when this pass cannot satisfy the receipt.
    """
    from app.core.preprocessing import preprocess_image

    prepared = preprocess_image(image)
    reading = read_variant(prepared, psm=FAST_PATH_PSM, lang=lang)
    reading.variant = "fast"
    return reading


def read_fast_fallback(image: Image.Image, lang: str = "eng") -> OcrReading:
    """Second cheap pass for the early-exit path.

    Used only when ``read_fast``'s anchor score fails the gate: the ``flat``
    rendering at PSM 11 rescues clean receipts whose single best-effort
    rendering happens to degrade (e.g. deskew/denoise hurting that particular
    image), without paying for the full pool. Receipts the primary fast pass
    already satisfies never reach this pass, so behavior is unchanged there.
    """
    from app.core.preprocessing import build_variants

    variants = {v.label: v.image for v in build_variants(image)}
    reading = read_variant(variants["flat"], psm=11, lang=lang)
    reading.variant = "fast_alt"
    return reading


def _fast_path_eligible(reading: OcrReading) -> bool:
    """Anchor score alone decides eligibility; extraction confirms afterwards."""
    return reading.score >= FAST_PATH_ANCHOR_SCORE


async def read_pooled(
    image: Image.Image,
    lang: str = "eng",
    psms: Sequence[int] | None = None,
    pool_size: int = 4,
    fast_path: bool = True,
) -> OcrBundle:
    """Run the early-exit fast pass, escalating to the full pool when needed.

    Fast path: one Tesseract pass over the preprocessed image. When its anchor
    score clears ``FAST_PATH_ANCHOR_SCORE`` the bundle is returned immediately
    with ``early_exit=True`` - the caller still runs extraction and must escalate
    via ``read_pooled(fast_path=False)`` if extraction does not reconcile. When
    the first pass is too weak, a second cheap pass over the ``flat`` rendering
    at PSM 11 gets one more try before the pool.

    Escalation: PaddleOCR and the Tesseract variant x PSM pool run concurrently,
    exactly as before, so hard receipts keep the full accuracy of the pooled path.
    The pool's PSM modes and variants come from ``settings`` (env
    ``OCR_POOL_PSMS``/``OCR_POOL_VARIANTS``) unless overridden via ``psms``.
    """
    if psms is None:
        from app.config import settings

        psms = tuple(
            int(mode)
            for mode in settings.ocr_pool_psms.split(",")
            if mode.strip()
        ) or DEFAULT_PSMS
    loop = asyncio.get_running_loop()

    if fast_path:
        try:
            fast = await loop.run_in_executor(ocr_executor, read_fast, image, lang)
        except Exception as exc:
            logger.warning("Fast path failed (%s); using pooled OCR", exc)
            fast = None
        if fast is not None and _fast_path_eligible(fast):
            logger.info(
                "Fast path: single pass variant=%s psm=%d score=%.3f conf=%.3f",
                fast.variant, fast.psm, fast.score, fast.confidence,
            )
            return OcrBundle(
                primary=fast, supporting=[], all_readings=[fast], early_exit=True
            )
        if fast is not None:
            logger.info(
                "Fast path score %.3f below %.2f; trying fallback rendering",
                fast.score, FAST_PATH_ANCHOR_SCORE,
            )
            try:
                fallback = await loop.run_in_executor(
                    ocr_executor, read_fast_fallback, image, lang
                )
            except Exception as exc:
                logger.warning("Fast-path fallback failed (%s); using pooled OCR", exc)
                fallback = None
            if fallback is not None and _fast_path_eligible(fallback):
                logger.info(
                    "Fast path fallback variant=%s psm=%d score=%.3f conf=%.3f",
                    fallback.variant, fallback.psm, fallback.score, fallback.confidence,
                )
                return OcrBundle(
                    primary=fallback, supporting=[], all_readings=[fallback],
                    early_exit=True,
                )
            if fallback is not None:
                logger.info(
                    "Fast path fallback score %.3f below %.2f; escalating to pooled OCR",
                    fallback.score, FAST_PATH_ANCHOR_SCORE,
                )

    candidates: list[OcrReading] = []

    # We can run Paddle and Tesseract best concurrently
    paddle_task = loop.run_in_executor(ocr_executor, read_paddle, image)
    tesseract_task = asyncio.create_task(read_best(image, lang=lang, psms=psms))

    # Wait for both. If one fails, the other might succeed.
    results = await asyncio.gather(paddle_task, tesseract_task, return_exceptions=True)
    
    paddle_res, tesseract_res = results[0], results[1]

    if isinstance(paddle_res, Exception):
        logger.warning("PaddleOCR failed; continuing with Tesseract: %s", paddle_res)
    else:
        candidates.append(paddle_res)
        logger.info(
            "PaddleOCR succeeded (score=%.3f, conf=%.3f)",
            paddle_res.score,
            paddle_res.confidence,
        )

    if isinstance(tesseract_res, Exception):
        logger.warning("Tesseract OCR failed; continuing with PaddleOCR: %s", tesseract_res)
    else:
        best_tess, tess_cands = tesseract_res
        candidates.extend(c for c in tess_cands if c.text.strip())

    if not candidates:
        raise RuntimeError("PaddleOCR and Tesseract produced no OCR candidates")

    primary = select_primary(candidates)
    supporting = _supporting_readings(candidates, primary, pool_size)
    runner_up = next(
        (candidate for candidate in rank_candidates(candidates) if candidate is not primary),
        None,
    )
    margin = primary.score - runner_up.score if runner_up else 0.0
    logger.info(
        "OCR selected engine=%s variant=%s psm=%s score=%.3f conf=%.3f margin=%.3f",
        primary.engine,
        primary.variant,
        primary.psm,
        primary.score,
        primary.confidence,
        margin,
    )
    return OcrBundle(
        primary=primary,
        supporting=supporting,
        all_readings=candidates,
    )


def extract_text(image: Image.Image, lang: str = "eng") -> tuple[str, float]:
    """Backward-compatible single-pass helper: returns ``(text, confidence)``."""
    reading = read_variant(image, psm=6, lang=lang)
    return reading.text, reading.confidence
