"""Prompts for the narrowed language-model role.

The model no longer reads the receipt. Amounts, tax ids, dates and invoice numbers
are extracted deterministically from OCR geometry, because asking a 1.5B model to
transcribe digits is where fabricated totals came from.

What remains is selection: choose one option from a closed list, or return null.
An answer outside the list is rejected by the caller, so a wrong answer is bounded
and a hallucinated one is impossible.
"""

VENDOR_SELECTION_PROMPT = """You are given candidate text lines from the printed header of a receipt.
Choose which line is the NAME OF THE BUSINESS that issued the receipt.

Rules:
- You MUST choose exactly one option from the CANDIDATES list, copied character for character.
- Do NOT invent, correct, complete or reformat a name.
- Do NOT choose a street address, phone number, tax number, or a person acting as proprietor.
- The DO_NOT_CHOOSE list contains the customer or payer. Never select those.
- If no candidate is the issuing business, return null.

Respond with JSON only: {{"vendor_name": "<exact candidate or null>"}}

CANDIDATES:
{candidates}

DO_NOT_CHOOSE:
{excluded}
"""

LOCATION_SELECTION_PROMPT = """You are given candidate text lines from a receipt header.
Choose which line best describes the LOCATION (city/area/address) of the business.

Rules:
- Choose EXACTLY one option from CANDIDATES, copied character for character.
- Prefer lines containing city, municipality, barangay, province or country names.
- Do NOT choose the business name, phone number, or tax ID.
- If no candidate is a location, return null.

Respond with JSON only: {{"location": "<exact candidate or null>"}}

CANDIDATES:
{candidates}
"""


CATEGORY_TIEBREAK_PROMPT = """Classify a business expense into exactly one category.

Rules:
- Choose ONLY from the two OPTIONS below.
- Respond with JSON only: {{"category": "<one of the options>"}}

OPTIONS: {option_a} | {option_b}

RECEIPT DETAILS:
{details}
"""

# Retained for the legacy single-shot path. Kept strict about abstention so that,
# if it is ever used, a missing value comes back as null rather than a guess.
RECEIPT_EXTRACTION_PROMPT = """Extract structured data from the OCR text of a receipt.

CRITICAL RULES:
- Copy values EXACTLY as they appear in the OCR text. Never reformat or compute.
- If a value is not literally present in the text, return null for it.
- Never guess an amount, a date, or a tax number. A null is correct; a guess is not.

Fields: vendor_name, transaction_date (YYYY-MM-DD), total_amount, vat_amount, tin,
invoice_number, expense_category, items[{{name, quantity, price}}].

expense_category must be exactly one of:
Meals, Travel, Supplies, Accommodation, Transportation, Others

Return the JSON object only.

OCR TEXT:
{ocr_text}
"""


ITEM_ANALYSIS_PROMPT = """Analyze the structural line items from the provided receipt text lines.

Rules:
- Identify each line item that represents a purchased product or service.
- Extract "name" (clean product description), "quantity" (integer, default 1 if omitted), and "price" (number).
- Do NOT include subtotal, tax, total, payment, change, or store discount lines.
- Respond with JSON only: {{"items": [{{"name": "<clean item name>", "quantity": 1, "price": 0.00}}]}}

RECEIPT LINES:
{lines}
"""
