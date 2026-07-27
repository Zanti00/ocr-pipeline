"""Synthetic receipt corpus generation.

The sample corpus contains only two machine-printed receipts, and neither is
Philippine, so no real image exercises the BIR path with a readable TIN *and*
readable amounts at once. A 90% gate measured on two receipts is not measurement.

Generating receipts solves that: because the generator chooses the values, ground
truth is exact and free - no keying, no verification, no ambiguity. It is also the
only way to test failure paths deliberately, such as a receipt whose VAT identity
does not close.

The honest limitation is that this measures the pipeline against *this
generator's* fonts and layouts. Randomised fonts, spacing and label wording
mitigate it, and the two real printed receipts remain as a transfer check.
"""

from app.eval.synthetic.corpus import CorpusStats, generate_corpus
from app.eval.synthetic.degrade import DEGRADATIONS, degrade
from app.eval.synthetic.render import render_receipt
from app.eval.synthetic.spec import TEMPLATES, LineItem, ReceiptSpec, build_spec

__all__ = [
    "CorpusStats", "generate_corpus", "DEGRADATIONS", "degrade",
    "render_receipt", "TEMPLATES", "LineItem", "ReceiptSpec", "build_spec",
]
