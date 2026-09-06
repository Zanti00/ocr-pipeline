# Project Summary: OCR Pipeline for SERMS

## Overview
The OCR Pipeline is a highly scalable, asynchronous microservice built to ingest documents (images/PDFs), extract text using OCR, and generate semantic embeddings for duplicate detection and search. It acts as a dedicated worker engine for the main SERMS backend.

## Main Features
- **Asynchronous Ingestion:** FastAPI endpoints that receive documents and immediately offload processing to Celery workers.
- **OCR Processing:** Accurate text extraction via PyTesseract and OpenCV.
- **Semantic Embedding & Search:** Vector generation via `sentence-transformers` and storage in PostgreSQL (`pgvector`) for detecting duplicates within a 90-day window.
- **Webhook Callbacks:** Automated HTTP notifications to upstream services once processing is complete or if it fails.

## Key User Flows
1. **Submission:** An internal service submits a document and receives a `job_id`.
2. **Processing:** Celery workers extract text and generate embeddings in the background.
3. **Storage & Verification:** Data is saved to MongoDB and PostgreSQL; a duplicate check is performed.
4. **Notification:** A webhook callback delivers the final extraction results back to the submitting service.

## Key Requirements
- **High Throughput:** Must handle concurrent processing via Redis and Celery without blocking.
- **Security:** Strict Service-to-Service API Key authentication for all endpoints and callbacks.
- **Accuracy:** Reliable text extraction even on complex document formats.
