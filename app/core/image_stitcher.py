"""Receipt segment stitching for long receipts.

Concatenates multiple image segments vertically into a single unified image
prior to running OCR.
"""

from __future__ import annotations

import logging
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)


def stitch_segments(images: list[Image.Image]) -> Image.Image:
    """Stitch 2 to MAX_SEGMENTS PIL images vertically into a single image.

    Pads narrower segments with white space to match the widest segment width.

    Raises:
        ValueError: If len(images) < 2 or exceeds image_quality_max_segments.
    """
    max_segments = settings.image_quality_max_segments
    if not images or len(images) < 2:
        raise ValueError("At least 2 image segments are required for stitching")
    if len(images) > max_segments:
        raise ValueError(f"Exceeded maximum segment count of {max_segments} (got {len(images)})")

    # Standardize image modes to RGB
    rgb_images = [img.convert("RGB") for img in images]

    max_width = max(img.width for img in rgb_images)
    total_height = sum(img.height for img in rgb_images)

    stitched = Image.new("RGB", (max_width, total_height), color=(255, 255, 255))

    current_y = 0
    for idx, img in enumerate(rgb_images):
        if img.width < max_width:
            # Center or left-align on white background
            padded = Image.new("RGB", (max_width, img.height), color=(255, 255, 255))
            padded.paste(img, (0, 0))
            stitched.paste(padded, (0, current_y))
        else:
            stitched.paste(img, (0, current_y))
        current_y += img.height

    logger.info(
        "Stitched %d receipt segments into unified image (%dx%d px)",
        len(images), max_width, total_height,
    )
    return stitched
