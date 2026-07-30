import sys
import asyncio
from app.core.pipeline import extract
from tests.unit.conftest import _bundles

def test_receipt(name):
    print(f'\\n--- Testing {name} ---')
    bundle = _bundles(name)
    res = extract(bundle)
    for f in ['tax_basis', 'net_sales', 'tax_amount', 'computed_total', 'reported_total', 'total_amount', 'financial_reconciliation_status']:
        print(f'{f}: {res.fields.get(f)}')
    print(f'Notes: {res.fields.get("reconciliation_notes")}')

test_receipt('receipt 21.jpg')
test_receipt('receipt 22.jpg')
