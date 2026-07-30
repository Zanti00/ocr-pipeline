Write-Host "Resetting MongoDB ocr_jobs..." -ForegroundColor Cyan
docker compose exec -T mongodb mongosh ocr_pipeline --eval "db.ocr_jobs.deleteMany({})"

Write-Host "Resetting PostgreSQL receipt_embeddings..." -ForegroundColor Cyan
docker compose exec -T postgres psql -U ocr_user -d ocr_pipeline -c "TRUNCATE TABLE receipt_embeddings RESTART IDENTITY;"

Write-Host "Resetting SERMS MySQL database..." -ForegroundColor Cyan
$mysqlQuery = @"
SET FOREIGN_KEY_CHECKS = 0;
TRUNCATE TABLE receipt_items;
TRUNCATE TABLE reimbursement_receipts;
TRUNCATE TABLE receipts;
TRUNCATE TABLE reimbursements;
SET FOREIGN_KEY_CHECKS = 1;
"@

$mysqlQuery | docker exec -i serms_mysql mysql -u root -psecret serms

Write-Host "Database reset complete!" -ForegroundColor Green
