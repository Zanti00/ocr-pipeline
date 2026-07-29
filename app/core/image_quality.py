"""Pre-OCR image quality evaluation.

Evaluates blur, brightness, and resolution using OpenCV heuristics before
running the resource-heavy OCR pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from app.core.preprocessing import correct_orientation, to_gray

logger = logging.getLogger(__name__)


@dataclass
class QualityCheckResult:
    """Structured output from check_image_quality()."""

    passed: bool
    rejection_code: Optional[str] = None       # "blurry" | "too_dark" | "too_small" | None
    rejection_reason: Optional[str] = None     # Human-readable sentence for frontend
    blur_score: float = 0.0                    # Laplacian variance (higher = sharper)
    brightness: float = 0.0                    # Mean pixel value 0-255
    resolution: tuple[int, int] = (0, 0)       # (width, height) after orientation correction


def check_image_quality(image: Image.Image) -> QualityCheckResult:
    """Run size, blur, and brightness checks on a PIL receipt image.

    Checks are performed in priority order: size -> blur -> brightness.

    Size is evaluated on the shorter/longer sides rather than fixed width/height
    axes. Thermal receipts are often tall and narrow (e.g. 225x571); a fixed
    min-width of 300 rejected those even when OCR read them cleanly.
    """
    upright = correct_orientation(image)
    width, height = upright.size
    resolution = (width, height)

    # 1. Size / Resolution check (orientation-independent)
    short_side, long_side = min(width, height), max(width, height)
    min_short = settings.image_quality_min_short_side
    min_long = settings.image_quality_min_long_side
    if short_side < min_short or long_side < min_long:
        reason = (
            f"Image dimensions ({width}x{height}px) are too small for accurate OCR. "
            f"Minimum required is {min_short}px on the short side and "
            f"{min_long}px on the long side."
        )
        logger.info("Quality check failed [too_small]: %s", reason)
        return QualityCheckResult(
            passed=False,
            rejection_code="too_small",
            rejection_reason=reason,
            resolution=resolution,
        )

    gray = to_gray(upright)

    # 2. Blur check via Laplacian variance
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_thresh = settings.image_quality_blur_threshold
    if blur_score < blur_thresh:
        reason = (
            "Image is too blurry for accurate OCR. "
            "Please retake the photo with better focus and steady lighting."
        )
        logger.info("Quality check failed [blurry]: score %.2f < threshold %.2f", blur_score, blur_thresh)
        return QualityCheckResult(
            passed=False,
            rejection_code="blurry",
            rejection_reason=reason,
            blur_score=round(blur_score, 2),
            resolution=resolution,
        )

    # 3. Brightness / Darkness check via mean pixel intensity
    brightness = float(np.mean(gray))
    brightness_floor = settings.image_quality_brightness_floor
    if brightness < brightness_floor:
        reason = (
            "Image is too dark or underexposed for clear OCR reading. "
            "Please capture under better lighting conditions."
        )
        logger.info("Quality check failed [too_dark]: mean %.2f < floor %.2f", brightness, brightness_floor)
        return QualityCheckResult(
            passed=False,
            rejection_code="too_dark",
            rejection_reason=reason,
            blur_score=round(blur_score, 2),
            brightness=round(brightness, 2),
            resolution=resolution,
        )

    return QualityCheckResult(
        passed=True,
        blur_score=round(blur_score, 2),
        brightness=round(brightness, 2),
        resolution=resolution,
    )
