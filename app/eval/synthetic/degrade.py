"""Image degradation, applied in reverse of the preprocessing chain.

Three severity tiers so accuracy can be reported *per* degradation level. That
tells you where the pipeline breaks rather than merely that it does - a single
blended figure hides whether failures are concentrated in severe photographs.
"""

from __future__ import annotations

import random
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

DEGRADATIONS: tuple[str, ...] = ("clean", "moderate", "severe")
TIERS = DEGRADATIONS  # backwards-compatible alias

# rotation degrees, perspective shift as a fraction of width, blur sigma,
# illumination strength, noise sigma, JPEG quality
TIER_SETTINGS = {
    "clean": (0.6, 0.004, 0.0, 0.10, 2.0, 92),
    "moderate": (2.5, 0.020, 0.6, 0.32, 5.0, 72),
    "severe": (6.0, 0.055, 1.3, 0.55, 11.0, 45),
}


def degrade(image: Image.Image, tier: str, rng: random.Random | int) -> Image.Image:
    """Apply one degradation tier.

    Accepts either a ``Random`` instance or a bare seed, so callers can either
    share a generator across a corpus or reproduce a single image on demand.
    """
    if tier not in TIER_SETTINGS:
        raise ValueError(f"Unknown degradation tier {tier!r}")
    if not isinstance(rng, random.Random):
        rng = random.Random(rng)
    rotation, shift, blur, light, noise, quality = TIER_SETTINGS[tier]

    array = np.array(image.convert("L"))
    array = _pad(array, rng)
    array = _perspective(array, shift, rng)
    array = _rotate(array, rotation, rng)
    array = _illuminate(array, light, rng)
    if blur:
        array = cv2.GaussianBlur(array, (0, 0), blur)
    array = _noise(array, noise, rng)
    return _jpeg(Image.fromarray(array), quality)


def _pad(array: np.ndarray, rng: random.Random) -> np.ndarray:
    """Surround the page with a darker background, as a phone photo would."""
    margin = rng.randint(20, 70)
    return cv2.copyMakeBorder(
        array, margin, margin, margin, margin,
        cv2.BORDER_CONSTANT, value=rng.randint(90, 160),
    )


def _perspective(array: np.ndarray, shift: float, rng: random.Random) -> np.ndarray:
    if shift <= 0:
        return array
    height, width = array.shape[:2]
    offset = shift * width

    def jitter() -> float:
        return rng.uniform(-offset, offset)

    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    target = np.float32([
        [jitter(), jitter()],
        [width + jitter(), jitter()],
        [width + jitter(), height + jitter()],
        [jitter(), height + jitter()],
    ])
    matrix = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(array, matrix, (width, height),
                               borderMode=cv2.BORDER_REPLICATE)


def _rotate(array: np.ndarray, degrees: float, rng: random.Random) -> np.ndarray:
    angle = rng.uniform(-degrees, degrees)
    if abs(angle) < 0.05:
        return array
    height, width = array.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(array, matrix, (width, height),
                          borderMode=cv2.BORDER_REPLICATE)


def _illuminate(array: np.ndarray, strength: float, rng: random.Random) -> np.ndarray:
    """Impose a directional gradient plus a soft shadow blob.

    This is the condition the shadow-removal stage exists for, so the corpus has
    to contain it or that stage is never actually exercised.
    """
    if strength <= 0:
        return array
    height, width = array.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    ys /= height
    xs /= width

    direction = rng.choice(("left", "right", "top", "bottom"))
    ramp = {"left": xs, "right": 1 - xs, "top": ys, "bottom": 1 - ys}[direction]
    field = 1.0 - strength * ramp

    cx, cy = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
    radius = rng.uniform(0.25, 0.5)
    blob = np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * radius ** 2)))
    field -= strength * 0.5 * blob

    result = array.astype(np.float32) * np.clip(field, 0.25, 1.15)
    return np.clip(result, 0, 255).astype(np.uint8)


def _noise(array: np.ndarray, sigma: float, rng: random.Random) -> np.ndarray:
    if sigma <= 0:
        return array
    generator = np.random.default_rng(rng.randint(0, 2**32 - 1))
    noisy = array.astype(np.float32) + generator.normal(0, sigma, array.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.convert("L").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy()
