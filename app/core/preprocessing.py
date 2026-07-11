import cv2
import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image) -> Image.Image:
    # Convert PIL Image to OpenCV format
    cv_img = np.array(image.convert('RGB'))
    # Convert RGB to BGR
    cv_img = cv_img[:, :, ::-1].copy()

    # Convert to grayscale
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding (Adaptive)
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Back to PIL
    result = Image.fromarray(thresh)
    return result
