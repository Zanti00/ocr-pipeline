"""Unit tests for 3-zone vertical blur check on tall receipts."""

import cv2
import numpy as np
from PIL import Image

from app.core.image_quality import check_image_quality


def _create_sharp_tall_receipt(width: int = 400, height: int = 1200) -> Image.Image:
    arr = np.full((height, width), 220, dtype=np.uint8)
    # Add sharp horizontal lines throughout
    for y in range(40, height - 40, 30):
        arr[y : y + 3, 30 : width - 30] = 0
        arr[y + 6 : y + 9, 30 : width - 30] = 255
    return Image.fromarray(arr)


def test_sharp_tall_receipt_passes_all_zones():
    img = _create_sharp_tall_receipt(400, 1200)
    res = check_image_quality(img)
    assert res.passed is True
    assert res.rejection_code is None
    assert res.blur_score > 80.0


def test_tall_receipt_with_blurry_bottom_third_rejected():
    # Top 2/3 sharp, bottom 1/3 heavily blurred (simulating mobile phone camera tilt)
    base_img = _create_sharp_tall_receipt(400, 1200)
    arr = np.array(base_img)

    # Blur the bottom third (from y=800 to y=1200)
    bottom_zone = arr[800:, :]
    blurred_bottom = cv2.GaussianBlur(bottom_zone, (41, 41), 0)
    arr[800:, :] = blurred_bottom

    tilted_img = Image.fromarray(arr)
    res = check_image_quality(tilted_img)

    # Should be rejected as blurry because the bottom third fails the zone gate
    assert res.passed is False
    assert res.rejection_code == "blurry"
    assert "too blurry" in res.rejection_reason.lower()
