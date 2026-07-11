# OCR Pipeline

A standalone Python/FastAPI microservice that provides AI-powered receipt OCR extraction, expense categorization, BIR VAT classification, and pgvector-based duplicate detection.

## Architecture
- **FastAPI**: API layer serving webhook triggers
- **Celery & Redis**: Asynchronous task processing queue
- **MongoDB**: Storage for OCR results and job status tracking
- **PostgreSQL & pgvector**: Vector embedding storage and similarity search
- **Ollama**: Local LLM execution for structured data extraction (Qwen2.5 1.5B)
- **Tesseract**: Traditional OCR for raw text extraction
- **Sentence-Transformers**: Text embedding generation (`all-MiniLM-L6-v2`)

## Setup

1. Copy environment variables:
   ```bash
   cp .env.example .env
   ```
2. Make sure you set your API keys for consumers (`SERMS_API_KEY`, `PRS_API_KEY`).
3. Start the Docker composition:
   ```bash
   docker compose up -d
   ```

> [!NOTE]
> **Automatic Model Downloads**
> - The Qwen2.5 1.5B model is automatically pulled by the `ocr_ollama` container's entrypoint script on the first startup. This may take a few minutes depending on your internet connection.
> - The `all-MiniLM-L6-v2` Sentence-Transformers model is downloaded and cached during the `ocr_api` container's build process, so no first-request latency will occur.

## Integration

See `docs/api-contract.md` for the full webhook integration and API spec.
