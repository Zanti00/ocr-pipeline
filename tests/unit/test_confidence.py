from app.core.confidence import compute_composite_score

def test_compute_composite_score():
    # Perfect score
    extracted = {
        "vendor_name": "Test", "transaction_date": "2026", "total_amount": 100, 
        "vat_amount": 10, "tin": "123", "invoice_number": "1", "expense_category": "A"
    }
    score = compute_composite_score(1.0, extracted, True)
    assert score == 1.0
    
    # Missing some fields, 80% tesseract, invalid TIN
    extracted_partial = {
        "vendor_name": "Test", "transaction_date": "2026", "total_amount": 100, 
        "expense_category": "A"
    }
    score = compute_composite_score(0.8, extracted_partial, False)
    assert round(score, 2) == 0.63
