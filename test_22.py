import sys
import asyncio
from pathlib import Path
from PIL import Image
from app.core.ocr_engine import read_pooled
from app.core.pipeline import _extract_with_assists

RECEIPTS = Path('docs/receipts')

async def main():
    bundle21 = read_pooled(Image.open(RECEIPTS / 'receipt 21.jpg'), lang='eng')
    ext21 = await _extract_with_assists(bundle21)
    print("Receipt 21:")
    print("  tax_basis:", ext21.fields.get('tax_basis'))
    print("  total_amount:", ext21.fields.get('total_amount'))
    print("  notes:", ext21.money.reconciliation_notes if ext21.money else '')

    bundle22 = read_pooled(Image.open(RECEIPTS / 'receipt 22.jpg'), lang='eng')
    ext22 = await _extract_with_assists(bundle22)
    print("Receipt 22:")
    print("  tax_basis:", ext22.fields.get('tax_basis'))
    print("  total_amount:", ext22.fields.get('total_amount'))
    print("  notes:", ext22.money.reconciliation_notes if ext22.money else '')

asyncio.run(main())
