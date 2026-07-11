FROM python:3.12-slim

# Install system dependencies for OpenCV, Tesseract, and poppler (for pdf2image)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
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
