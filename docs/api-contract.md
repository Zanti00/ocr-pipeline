# API Contract

## Authentication
All endpoints (except health) require a Bearer token matching the respective consumer's API key.
```
Authorization: Bearer <API_KEY>
```

## Endpoints

### POST `/api/ocr/process`
Submits a receipt for async processing.

**Request:**
```json
{
  "receipt_id": 42,
  "file_url": "https://...",
  "callback_url": "https://...",
  "source_service": "serms"
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "uuid",
  "status": "queued",
  "message": "Receipt queued for OCR processing."
}
```

**Response (422 Unprocessable Entity) — pre-OCR quality rejection:**

Returned *synchronously* when the image fails blur / brightness / size checks
and `force_process` is not set. **No Celery job is queued and no callback is
sent.** The client must surface `message` / `rejection_reason` as a toast or
modal. A MongoDB audit row is still written with `status: "rejected"`.

```json
{
  "status": "rejected",
  "message": "Image dimensions (100x100px) are too small for accurate OCR. ...",
  "rejection_code": "too_small",
  "rejection_reason": "Image dimensions (100x100px) are too small for accurate OCR. ...",
  "blur_score": 0.0,
  "brightness": 0.0,
  "resolution": [100, 100],
  "segment_index": null,
  "job_id": "uuid",
  "receipt_id": 42
}
```

`rejection_code` values: `too_small` | `blurry` | `too_dark` | `quality_failed`.

To bypass the gate (e.g. user confirms a retake is impossible), resubmit with
`"force_process": true`.

### POST Callback (Async)
Sent to the provided `callback_url`.
```json
{
  "receipt_id": 42,
  "ocr_confidence_score": 0.85,
  "status": "completed",
  "vendor_name": "...",
  "transaction_date": "...",
  "total_amount": 100.0,
  "vat_amount": 10.0,
  "tin": "...",
  "invoice_number": "...",
  "vat_classification": "vat",
  "expense_category": "Meals",
  "is_duplicate": false,
  "duplicate_similarity": null,
  "items": [
    {
      "name": "Item",
      "quantity": 1,
      "price": 100.0
    }
  ]
}
```

### POST `/api/duplicate-check`
Checks for duplicate receipts.
```json
{
  "receipt_text": "...",
  "source_service": "serms",
  "threshold": 0.85,
  "days_window": 90
}
```

## Financial context and callback additions
`POST /api/ocr/process` accepts optional flat context fields. They are normalized before queueing and propagated unchanged through the job:
```json
{"country":"US","currency":"USD","location":"Las Vegas"}
```
Caller currency/country/location take priority over OCR context. Currency is validated internally as ISO-4217; the outbound callback retains the existing consumer allowlist and omits unsupported codes rather than rejecting the receipt.

Callbacks may additionally include `tax_basis` (`inclusive|exclusive|unknown`), `financial_reconciliation_status`, and `needs_manual_review`. These are optional and legacy fields remain unchanged. Detailed semantics, evidence, reported/computed totals, discrepancy, tax-line breakdown, and review reasons are retained in the Mongo job document.
