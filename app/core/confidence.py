def compute_composite_score(tesseract_confidence: float, extracted_fields: dict, bir_valid: bool) -> float:
    """
    Computes a composite confidence score (0.0 to 1.0)
    Weighting:
    - 50% Tesseract raw OCR confidence
    - 40% LLM extraction completeness (how many expected fields are not null)
    - 10% BIR Validation success (TIN is valid)
    """
    # Expected fields from the prompt
    expected_fields = [
        "vendor_name", "transaction_date", "total_amount", 
        "vat_amount", "tin", "invoice_number", "expense_category"
    ]
    
    filled_fields = sum(1 for field in expected_fields if extracted_fields.get(field))
    completeness_score = filled_fields / len(expected_fields)
    
    bir_score = 1.0 if bir_valid else 0.0
    
    composite = (tesseract_confidence * 0.5) + (completeness_score * 0.4) + (bir_score * 0.1)
    
    return round(composite, 4)
