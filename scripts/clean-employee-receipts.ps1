param (
    [string]$TargetEmail = "employee@example.com"
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Purging receipts and artifacts for: $TargetEmail" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

Write-Host "[1/5] Purging receipts, files, and claims in SERMS container..." -ForegroundColor Yellow
$jsonRaw = docker exec serms_php php /var/www/scripts/purge_employee_receipts.php "$TargetEmail"
Write-Host $jsonRaw

try {
    $res = $jsonRaw | ConvertFrom-Json
    $receiptIds = $res.receipt_ids
    $activeIds = $res.all_active_receipt_ids
} catch {
    $receiptIds = @()
    $activeIds = @()
}

Write-Host "[2/5] Purging pgvector embeddings in ocr_postgres..." -ForegroundColor Yellow
if ($activeIds -and $activeIds.Count -gt 0) {
    $activeList = $activeIds -join ', '
    # Remove any ghost embeddings that do not belong to active receipts in SERMS
    docker exec ocr_postgres psql -U ocr_user -d ocr_pipeline -c "DELETE FROM receipt_embeddings WHERE receipt_id NOT IN ($activeList) AND source_service = 'serms';"
} else {
    # No active receipts left in SERMS -> truncate all embeddings
    docker exec ocr_postgres psql -U ocr_user -d ocr_pipeline -c "TRUNCATE TABLE receipt_embeddings RESTART IDENTITY;"
}

Write-Host "[3/5] Purging MongoDB jobs in ocr_mongo..." -ForegroundColor Yellow
if ($receiptIds -and $receiptIds.Count -gt 0) {
    $idsList = $receiptIds -join ', '
    docker exec ocr_mongo mongosh ocr_pipeline --quiet --eval "db.ocr_jobs.deleteMany({ receipt_id: { `$in: [$idsList] } })"
} else {
    if ($activeIds -and $activeIds.Count -gt 0) {
        $activeList = $activeIds -join ', '
        docker exec ocr_mongo mongosh ocr_pipeline --quiet --eval "db.ocr_jobs.deleteMany({ receipt_id: { `$nin: [$activeList] } })"
    } else {
        docker exec ocr_mongo mongosh ocr_pipeline --quiet --eval "db.ocr_jobs.deleteMany({})"
    }
}

Write-Host "[4/5] Flushing file hash cache in ocr_redis..." -ForegroundColor Yellow
docker exec ocr_redis sh -c "redis-cli --scan --pattern 'ocr:file_hash:*' | xargs -r redis-cli del"

Write-Host "[5/5] Cleaning temporary OCR artifacts in ocr_worker container..." -ForegroundColor Yellow
docker exec ocr_worker sh -c "rm -f /tmp/tess_* /tmp/*.txt"

Write-Host "==========================================================" -ForegroundColor Green
Write-Host " Cleanup complete! $TargetEmail is ready for fresh testing." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
