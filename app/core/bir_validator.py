import re

def validate_tin(tin: str) -> bool:
    if not tin:
        return False
    # Matches xxx-xxx-xxx-xxx or xxx-xxx-xxx-xxxV
    pattern = r"^\d{3}-\d{3}-\d{3}-\d{3}V?$"
    return bool(re.match(pattern, tin.strip()))

def classify_vat(ocr_text: str, tin: str = None) -> str:
    text_upper = ocr_text.upper()
    
    if "NON-VAT" in text_upper or "NON VAT" in text_upper:
        return "non-vat"
        
    if "VAT-EXEMPT" in text_upper or "VAT EXEMPT" in text_upper:
        return "non-vat"
        
    if "VAT" in text_upper:
        return "vat"
        
    return "non-vat"
