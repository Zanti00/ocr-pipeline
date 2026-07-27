FROM python:3.12-slim

# Install system dependencies for OpenCV, Tesseract, and poppler (for pdf2image).
#
# tesseract-ocr-osd supplies the orientation-and-script detection data. Without it
# pytesseract.image_to_osd raises, the preprocessing stage swallows the error, and
# rotated receipts are never straightened - a silent loss of accuracy that does not
# reproduce on hosts whose Tesseract ships OSD data by default.
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-osd \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip install .

# Cache sentence-transformers model
COPY scripts/download_model.py scripts/download_model.py
RUN python scripts/download_model.py

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]
