from pdf2image import convert_from_bytes
from PIL import Image

def pdf_to_images(pdf_bytes: bytes) -> list[Image.Image]:
    images = convert_from_bytes(pdf_bytes)
    return images
