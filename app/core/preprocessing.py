"""Image preparation for OCR.

Rather than one fixed chain, this module produces several candidate renderings
of the same receipt. The settings that rescue a faded thermal print will blow out
a crisp scan, so the pipeline generates variants and lets measured OCR quality
pick the winner (see ``app.core.ocr_engine.read_best``).

Stage order matters and is deliberate:

1. orientation      - must run first; every later stage assumes upright text
2. perspective      - guarded; declines to warp when no confident document quad
3. shadow removal   - divide by a blurred background estimate ("flatbed" look)
4. upscale          - Tesseract needs roughly 300 DPI equivalent x-height
5. denoise          - bilateral, which preserves thin thermal strokes
6. contrast (CLAHE) - after shadow removal, or it amplifies the blotches
7. deskew           - small residual rotation left over after warping
8. binarise         - OPTIONAL. Tesseract 5's LSTM engine often scores higher on
                      clean grayscale, so this is offered as a variant rather
                      than applied unconditionally.

Sharpening is intentionally absent: on thermal and dot-matrix print it amplifies
speckle into character-shaped noise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

TARGET_LONG_EDGE = 1600
MAX_UPSCALE = 3.0
MAX_DESKEW_DEGREES = 15.0
MIN_QUAD_AREA_RATIO = 0.50

# Tesseract's orientation confidence is unbounded; genuine rotations typically
# score well above 1, while noise sits near zero. Measured false positives on
# receipts came in around 0.1-0.2.
MIN_ORIENTATION_CONFIDENCE = 2.0


@dataclass
class Variant:
    """One candidate rendering of a receipt image."""

    label: str
    image: Image.Image


# ---------------------------------------------------------------------------
# individual stages
# ---------------------------------------------------------------------------

def to_gray(image: Image.Image) -> np.ndarray:
    array = np.array(image.convert("RGB"))[:, :, ::-1]
    return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)


def correct_orientation(image: Image.Image) -> Image.Image:
    """Rotate to upright using Tesseract's orientation detection.

    Only acts on a *confident* detection. Tesseract's OSD reports a rotation
    regardless of how weak the evidence is, and on receipts - short lines, sparse
    text, large blank areas - it guesses badly. An upright synthetic slip was
    reported as 180 degrees with ``orientation_conf`` 0.12 (and its script
    identified as Arabic), and rotating on that reading turned a perfectly
    readable receipt upside down, taking every field with it.

    Ignoring rotation when unsure is the safe failure: a wrongly upright page
    still reads, whereas a wrongly inverted one reads as nothing.

    Requires the ``osd`` traineddata; if unavailable the image is returned
    unchanged rather than failing the job.
    """
    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        rotate = int(osd.get("rotate", 0)) % 360
        confidence = float(osd.get("orientation_conf", 0.0))
    except Exception as exc:  # pragma: no cover - depends on tessdata presence
        logger.debug("Orientation detection unavailable: %s", exc)
        return image

    if not rotate:
        return image
    if confidence < MIN_ORIENTATION_CONFIDENCE:
        logger.debug(
            "Ignoring %s deg orientation: confidence %.2f below %.2f",
            rotate, confidence, MIN_ORIENTATION_CONFIDENCE,
        )
        return image

    # PIL rotates counter-clockwise; OSD reports the clockwise correction.
    return image.rotate(-rotate, expand=True, fillcolor="white")


def correct_perspective(gray: np.ndarray) -> tuple[np.ndarray, bool]:
    """Flatten a photographed receipt to a head-on view.

    Declines when no convincing document quadrilateral is found. Several samples
    are cropped fragments with no visible border, and warping on a false
    detection destroys the image - a slightly tilted receipt reads better than a
    confidently mangled one.
    """
    height, width = gray.shape[:2]
    scale = 800.0 / max(height, width)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale < 1 else gray.copy()

    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gray, False

    small_area = small.shape[0] * small.shape[1]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        perimeter = cv2.arcLength(contour, True)
        quad = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(quad) != 4 or not cv2.isContourConvex(quad):
            continue
        if cv2.contourArea(quad) < MIN_QUAD_AREA_RATIO * small_area:
            continue
        points = quad.reshape(4, 2).astype(np.float32) / (scale if scale < 1 else 1.0)
        return _warp(gray, _order_corners(points)), True

    return gray, False


def _order_corners(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype=np.float32)
    total = points.sum(axis=1)
    ordered[0] = points[np.argmin(total)]   # top-left
    ordered[2] = points[np.argmax(total)]   # bottom-right
    diff = np.diff(points, axis=1).ravel()
    ordered[1] = points[np.argmin(diff)]    # top-right
    ordered[3] = points[np.argmax(diff)]    # bottom-left
    return ordered


def _warp(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    (tl, tr, br, bl) = corners
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 50 or height < 50:
        return gray
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(gray, matrix, (width, height), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def remove_shadow(gray: np.ndarray, kernel: int = 41) -> np.ndarray:
    """Normalise uneven illumination by dividing out a blurred background.

    Division rather than subtraction is what produces the flat white page;
    subtraction leaves grey blotches where the shadow was.
    """
    kernel = kernel if kernel % 2 else kernel + 1
    background = cv2.medianBlur(gray, kernel)
    normalized = cv2.divide(gray, background, scale=255)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def upscale(gray: np.ndarray) -> np.ndarray:
    """Scale up small images so character height reaches Tesseract's sweet spot."""
    height, width = gray.shape[:2]
    longest = max(height, width)
    if longest >= TARGET_LONG_EDGE:
        return gray
    factor = min(TARGET_LONG_EDGE / longest, MAX_UPSCALE)
    return cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Bilateral filter: smooths noise while keeping stroke edges intact.

    Replaces the previous 5x5 Gaussian blur, which was thinning and breaking
    thermal-print strokes before they reached the recogniser.
    """
    return cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)


def enhance_contrast(gray: np.ndarray, clip: float = 2.0) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return clahe.apply(gray)


def deskew(gray: np.ndarray) -> np.ndarray:
    """Correct small residual rotation using the dominant text angle."""
    inverted = cv2.bitwise_not(gray)
    _, mask = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    coords = cv2.findNonZero(mask)
    if coords is None or len(coords) < 50:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    if angle > 45:
        angle -= 90
    if abs(angle) < 0.3 or abs(angle) > MAX_DESKEW_DEGREES:
        return gray

    height, width = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (width, height), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def binarize_sauvola(gray: np.ndarray, window: int = 25, k: float = 0.2) -> np.ndarray:
    """Sauvola local thresholding - handles uneven exposure better than Otsu.

    Implemented with box filters to avoid a scikit-image dependency.
    """
    window = window if window % 2 else window + 1
    image = gray.astype(np.float32)
    mean = cv2.boxFilter(image, ddepth=-1, ksize=(window, window), normalize=True)
    mean_sq = cv2.boxFilter(image * image, ddepth=-1, ksize=(window, window),
                            normalize=True)
    std = np.sqrt(np.clip(mean_sq - mean * mean, 0, None))
    threshold = mean * (1.0 + k * ((std / 128.0) - 1.0))
    return np.where(image > threshold, 255, 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# variant construction
# ---------------------------------------------------------------------------

def build_variants(image: Image.Image) -> list[Variant]:
    """Produce the candidate renderings the OCR stage will choose between.

    ``raw`` is included on purpose: measurement showed untouched grayscale beats
    the previous fixed chain by a wide margin, so it earns a place as a genuine
    contender rather than a fallback.
    """
    upright = correct_orientation(image)
    gray = to_gray(upright)

    warped, did_warp = correct_perspective(gray)
    base = upscale(warped)
    if not did_warp:
        base = upscale(gray)

    flattened = remove_shadow(base)
    cleaned = deskew(denoise(flattened))

    variants = [
        Variant("raw", Image.fromarray(gray)),
        Variant("flat", Image.fromarray(flattened)),
        Variant("clean", Image.fromarray(cleaned)),
        Variant("contrast", Image.fromarray(enhance_contrast(cleaned))),
        Variant("sauvola", Image.fromarray(binarize_sauvola(cleaned))),
    ]
    return variants


def preprocess_image(image: Image.Image) -> Image.Image:
    """Single best-effort rendering.

    Retained for callers that want one image rather than a variant set. Returns
    the shadow-flattened, denoised, deskewed grayscale - deliberately NOT
    binarised, since Tesseract 5's LSTM engine generally reads grayscale better.
    """
    upright = correct_orientation(image)
    gray = to_gray(upright)
    warped, did_warp = correct_perspective(gray)
    base = upscale(warped if did_warp else gray)
    return Image.fromarray(deskew(denoise(remove_shadow(base))))
