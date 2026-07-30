#!/bin/bash

echo "Resetting MongoDB ocr_jobs..."
docker compose exec -T mongodb mongosh ocr_pipeline --eval "db.ocr_jobs.deleteMany({})"

echo "Resetting PostgreSQL receipt_embeddings..."
docker compose exec -T postgres psql -U ocr_user -d ocr_pipeline -c "TRUNCATE TABLE receipt_embeddings RESTART IDENTITY;"

echo "Resetting SERMS MySQL database..."
docker exec -i serms_mysql mysql -u root -psecret serms << 'EOF'
SET FOREIGN_KEY_CHECKS = 0;
TRUNCATE TABLE receipt_items;
TRUNCATE TABLE reimbursement_receipts;
TRUNCATE TABLE receipts;
TRUNCATE TABLE reimbursements;
SET FOREIGN_KEY_CHECKS = 1;
EOF

echo "Database reset complete!"
