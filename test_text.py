from pathlib import Path
from PIL import Image
from app.core.ocr_engine import read_pooled
from app.core.locale import detect_locale
from app.core.financial_semantics import infer_tax_basis

RECEIPTS = Path('docs/receipts')

def test(name):
    bundle = read_pooled(Image.open(RECEIPTS / name), lang='eng')
    loc = detect_locale(bundle.combined_text)
    basis = infer_tax_basis(bundle.combined_text, loc.country)
    print(f"{name}: locale={loc.country}, basis={basis}")

test('receipt 21.jpg')
test('receipt 22.jpg')
