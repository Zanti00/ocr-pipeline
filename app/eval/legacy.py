"""Frozen copy of the original preprocessing chain.

Preserved verbatim so the "before" figure stays reproducible after
``app.core.preprocessing`` is rewritten. Measured at 43.8% recoverability on the
sample corpus, against 71.9% for no preprocessing at all - the Gaussian blur and
adaptive threshold were erasing legible vendor names and TINs.

Not used in production. Referenced only by the evaluation harness.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def legacy_preprocess_image(image: Image.Image) -> Image.Image:
    cv_img = np.array(image.convert("RGB"))
    cv_img = cv_img[:, :, ::-1].copy()

    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return Image.fromarray(thresh)
