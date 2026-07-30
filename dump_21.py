from pathlib import Path
from PIL import Image
from app.core.ocr_engine import read_pooled

RECEIPTS = Path('docs/receipts')
bundle = read_pooled(Image.open(RECEIPTS / 'receipt 21.jpg'), lang='eng')
print(bundle.combined_text)
