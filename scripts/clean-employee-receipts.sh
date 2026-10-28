#!/bin/bash
set -e

TARGET_EMAIL="${1:-employee@example.com}"
echo "=========================================================="
echo " Purging receipts and artifacts for: $TARGET_EMAIL"
echo "=========================================================="

echo "[1/5] Purging receipts, files, and claims in SERMS container..."
RESULT=$(docker exec serms_php php /var/www/scripts/purge_employee_receipts.php "$TARGET_EMAIL")
echo "$RESULT"

# Extract receipt IDs and active IDs from JSON output
RECEIPT_IDS=$(echo "$RESULT" | grep -o '"receipt_ids": *\[[^]]*\]' | grep -o '[0-9]\+' | paste -sd, -)
ACTIVE_IDS=$(echo "$RESULT" | grep -o '"all_active_receipt_ids": *\[[^]]*\]' | grep -o '[0-9]\+' | paste -sd, -)

echo "[2/5] Purging pgvector embeddings in ocr_postgres..."
if [ -n "$ACTIVE_IDS" ]; then
    docker exec ocr_postgres psql -U ocr_user -d ocr_pipeline -c "DELETE FROM receipt_embeddings WHERE receipt_id NOT IN ($ACTIVE_IDS) AND source_service = 'serms';"
else
    docker exec ocr_postgres psql -U ocr_user -d ocr_pipeline -c "TRUNCATE TABLE receipt_embeddings RESTART IDENTITY;"
fi

echo "[3/5] Purging MongoDB jobs in ocr_mongo..."
if [ -n "$RECEIPT_IDS" ]; then
    docker exec ocr_mongo mongosh ocr_pipeline --quiet --eval "db.ocr_jobs.deleteMany({ receipt_id: { \$in: [$RECEIPT_IDS] } })"
elif [ -n "$ACTIVE_IDS" ]; then
    docker exec ocr_mongo mongosh ocr_pipeline --quiet --eval "db.ocr_jobs.deleteMany({ receipt_id: { \$nin: [$ACTIVE_IDS] } })"
else
    docker exec ocr_mongo mongosh ocr_pipeline --quiet --eval "db.ocr_jobs.deleteMany({})"
fi

echo "[4/5] Flushing file hash cache in ocr_redis..."
docker exec ocr_redis sh -c "redis-cli --scan --pattern 'ocr:file_hash:*' | xargs -r redis-cli del" || true

echo "[5/5] Cleaning temporary OCR artifacts in ocr_worker container..."
docker exec ocr_worker sh -c "rm -f /tmp/tess_* /tmp/*.txt" || true

echo "=========================================================="
echo " Cleanup complete! $TARGET_EMAIL is ready for fresh testing."
echo "=========================================================="
