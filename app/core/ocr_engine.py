import pytesseract
from PIL import Image
from typing import Tuple

def extract_text(image: Image.Image, lang: str = "eng") -> Tuple[str, float]:
    """
    Extracts text from image and returns (text, average_confidence).
    """
    data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    
    text_chunks = []
    confidences = []
    
    for i, word in enumerate(data['text']):
        if word.strip():
            text_chunks.append(word)
            # Confidence is out of 100
            conf = int(data['conf'][i])
            if conf >= 0:
                confidences.append(conf)
                
    full_text = " ".join(text_chunks)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    
    # Return confidence as a float between 0 and 1
    return full_text, avg_conf / 100.0
