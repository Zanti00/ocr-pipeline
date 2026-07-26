"""Randomised receipt content.

Amounts are constructed so the identities close exactly - VAT computed from net,
service charge from net, total as the sum. That is what makes the reconciliation
gate measurable: a generated receipt is arithmetically sound by construction, so
any reconciliation failure the pipeline reports on one is a false positive.

Deliberately inconsistent receipts are produced separately, by setting
``corrupt_arithmetic`` on a spec, so the gate has true positives to catch. Waiting
for a real receipt with an arithmetic error is not a test strategy.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta

PH_VAT_RATE = 0.12
MY_SST_RATE = 0.06
US_TAX_RATES = (0.0725, 0.0825, 0.0838, 0.06, 0.10)

TEMPLATES: tuple[str, ...] = (
    "ph_vat_or",     # VAT-registered official receipt
    "ph_nonvat_si",  # non-VAT sales invoice
    "ph_pos",        # thermal POS slip, Philippines
    "us_pos",        # US thermal slip, sales tax, no TIN
    "my_pos",        # Malaysian POS slip with SST
)
PH_TEMPLATES = ("ph_vat_or", "ph_nonvat_si", "ph_pos")

PH_VENDORS = (
    ("Yamachan Japanese Restaurant", "Meals", "Mark Anthony R. Alvaro"),
    ("Kanto Freestyle Breakfast", "Meals", "Jerome T. Villanueva"),
    ("Bayanihan Grill and Restobar", "Meals", "Rosalie M. Cruz"),
    ("Okyecen Consumer Goods Wholesaling", "Supplies", "Jade B. Parativo"),
    ("Metro Hardware and Construction Supply", "Supplies", "Danilo S. Reyes"),
    ("Sunrise Office Solutions Inc.", "Supplies", None),
    ("Seda Residences Makati", "Accommodation", None),
    ("Palawan Sunset Inn", "Accommodation", "Maria L. Santos"),
    ("Detoxicare Molecular Diagnostics Laboratory Inc.", "Others", None),
    ("United Daily Press Inc.", "Others", None),
    ("Bayan Telecommunications, Inc.", "Others", None),
    ("Quezon Transport Services Corp.", "Transportation", None),
)

PH_ADDRESSES = (
    ("Unit 6-1 Makati Cinema Square, Fernando St.", "Pio Del Pilar, Makati City"),
    ("2/F Ever Gotesco Commonwealth, Commonwealth Ave.", "Quezon City, NCR"),
    ("812 Benavidez Street Barangay 295", "Binondo Manila, Philippines"),
    ("P-5 B-14 L-11 Eastwood Residences", "Brgy. San Isidro Rodriguez, Rizal"),
    ("145 Salcedo Street, Legaspi Village", "Makati City, Fourth District"),
)

US_VENDORS = (
    ("Jollibee Las Vegas", "Meals",
     ("3890 S Maryland Parkway Suite 137", "Las Vegas NV 89119")),
    ("Blue Bottle Coffee", "Meals", ("66 Mint St", "San Francisco CA 94103")),
    ("Office Depot Store 2241", "Supplies",
     ("1200 N Federal Hwy", "Boca Raton FL 33432")),
    ("Hilton Garden Inn Seattle", "Accommodation",
     ("1821 Boren Ave", "Seattle WA 98101")),
)

MY_VENDORS = (
    ("Jollibee Yayasan Complex", "Meals",
     ("Jollibee Yayasan Complex BO6", "Kuala Lumpur, Malaysia")),
    ("Secret Recipe Sdn. Bhd.", "Meals", ("Lot 12 Jalan Ampang", "Kuala Lumpur")),
)

MENU_ITEMS = (
    ("Chicken Teriyaki Set", 285.0), ("Salmon Sashimi", 420.0),
    ("Beef Gyudon", 265.0), ("California Maki", 180.0),
    ("Iced Tea", 75.0), ("Bottled Water", 35.0),
    ("Bond Paper A4 Ream", 245.0), ("Ballpen Box", 120.0),
    ("Stapler Heavy Duty", 340.0), ("Printer Ink Cartridge", 890.0),
    ("Deluxe Room 1 Night", 3200.0), ("Airport Transfer", 850.0),
    ("Laboratory Test Panel", 1450.0), ("Newspaper Subscription", 300.0),
)

DATE_FORMATS = ("%m-%d-%Y", "%m/%d/%Y", "%d-%m-%Y", "%b. %d, %Y", "%B %d, %Y",
                "%Y-%m-%d", "%d %b %Y")
FONTS = ("arial", "calibri", "verdana", "tahoma", "consola", "cour", "times", "lucon")

CUSTOMERS = (
    "Scientific Biotech Specialties, Inc.", "Kumho Trading Incorporated",
    "Rey Nimfa Enterprises", "Pacific Rim Logistics Corp.",
)


@dataclass
class LineItem:
    name: str
    quantity: int
    unit_price: float

    @property
    def amount(self) -> float:
        return round(self.quantity * self.unit_price, 2)


@dataclass
class ReceiptSpec:
    """The printed content and the ground truth, identical by construction."""

    template: str
    vendor_name: str
    country: str
    currency: str
    transaction_date: date
    date_text: str
    items: list[LineItem]

    net_sales: float
    tax_amount: float | None
    total_sales: float
    service_charge: float | None
    total_amount: float

    tax_type: str | None
    tax_rate: float | None
    vat_classification: str | None

    vendor_tax_id: str | None = None
    vendor_tax_id_type: str | None = None
    customer_name: str | None = None
    customer_tax_id: str | None = None
    invoice_number: str | None = None
    expense_category: str = "Others"

    address_lines: list[str] = field(default_factory=list)
    proprietor: str | None = None
    font_name: str = "arial"
    index: int = 0

    corrupt_arithmetic: bool = False
    """When set, the rendered tax figure is inflated so the identities cannot close.

    The spec's own numbers stay consistent; only the printed tax differs, which is
    what the reconciliation gate has to notice.
    """

    @property
    def printed_tax_amount(self) -> float | None:
        if self.tax_amount is None:
            return None
        if not self.corrupt_arithmetic:
            return self.tax_amount
        return round(self.tax_amount * 1.75 + 3.0, 2)


def build_spec(index: int, rng: random.Random, template: str | None = None) -> ReceiptSpec:
    template = template or rng.choice(TEMPLATES)
    if template in PH_TEMPLATES:
        return _build_ph(index, rng, template)
    if template == "us_pos":
        return _build_us(index, rng)
    if template == "my_pos":
        return _build_my(index, rng)
    raise ValueError(f"Unknown template {template!r}")


def _items(rng: random.Random, count: int) -> list[LineItem]:
    chosen = rng.sample(MENU_ITEMS, k=min(count, len(MENU_ITEMS)))
    return [
        LineItem(name=name, quantity=rng.randint(1, 3), unit_price=price)
        for name, price in chosen
    ]


def _date(rng: random.Random) -> tuple[date, str]:
    value = date.today() - timedelta(days=rng.randint(1, 900))
    return value, value.strftime(rng.choice(DATE_FORMATS))


def _tax_id(rng: random.Random) -> str:
    """A PH TIN with a three- or five-digit branch code.

    Both widths occur in practice and the original validator accepted only the
    narrower one, so the corpus deliberately mixes them.
    """
    groups = [f"{rng.randint(0, 999):03d}" for _ in range(3)]
    branch = (f"{rng.randint(0, 999):03d}" if rng.random() < 0.55
              else f"{rng.randint(0, 99999):05d}")
    return "-".join(groups + [branch])


def _build_ph(index: int, rng: random.Random, template: str) -> ReceiptSpec:
    vendor, category, proprietor = rng.choice(PH_VENDORS)
    items = _items(rng, rng.randint(1, 4))
    gross = round(sum(item.amount for item in items), 2)
    transaction_date, date_text = _date(rng)

    if template == "ph_nonvat_si":
        net_sales = total_sales = gross
        tax_amount, tax_type, tax_rate = None, None, None
        vat_classification = "non-vat"
        service_charge = None
    else:
        # PH VAT is inclusive: the printed sales figure already contains the tax.
        total_sales = gross
        net_sales = round(total_sales / (1 + PH_VAT_RATE), 2)
        tax_amount = round(total_sales - net_sales, 2)
        tax_type, tax_rate = "VAT", PH_VAT_RATE
        vat_classification = "vat"
        service_charge = (round(net_sales * 0.10, 2)
                          if template == "ph_vat_or" and rng.random() < 0.45 else None)

    total_amount = round(total_sales + (service_charge or 0.0), 2)
    has_customer = rng.random() < 0.40

    return ReceiptSpec(
        template=template,
        vendor_name=vendor,
        country="PH",
        currency="PHP",
        transaction_date=transaction_date,
        date_text=date_text,
        items=items,
        net_sales=net_sales,
        tax_amount=tax_amount,
        total_sales=total_sales,
        service_charge=service_charge,
        total_amount=total_amount,
        tax_type=tax_type,
        tax_rate=tax_rate,
        vat_classification=vat_classification,
        vendor_tax_id=_tax_id(rng),
        vendor_tax_id_type="PH_TIN",
        customer_name=rng.choice(CUSTOMERS) if has_customer else None,
        customer_tax_id=_tax_id(rng) if has_customer else None,
        invoice_number=f"{rng.randint(1000, 99999)}",
        expense_category=category,
        address_lines=list(rng.choice(PH_ADDRESSES)),
        proprietor=proprietor,
        font_name=rng.choice(FONTS),
        index=index,
    )


def _build_us(index: int, rng: random.Random) -> ReceiptSpec:
    vendor, category, address = rng.choice(US_VENDORS)
    items = _items(rng, rng.randint(1, 3))
    transaction_date, date_text = _date(rng)

    # US sales tax is exclusive and varies by jurisdiction, so only the additive
    # identity can be asserted - the rate itself is not predictable.
    net_sales = round(sum(item.amount for item in items) / 10, 2)
    rate = rng.choice(US_TAX_RATES)
    tax_amount = round(net_sales * rate, 2)
    total = round(net_sales + tax_amount, 2)

    return ReceiptSpec(
        template="us_pos",
        vendor_name=vendor,
        country="US",
        currency="USD",
        transaction_date=transaction_date,
        date_text=date_text,
        items=items,
        net_sales=net_sales,
        tax_amount=tax_amount,
        total_sales=total,
        service_charge=None,
        total_amount=total,
        tax_type="SALES_TAX",
        tax_rate=rate,
        # SERMS validates vat_classification with in:vat,non-vat, so a foreign
        # receipt must send null rather than a third value.
        vat_classification=None,
        invoice_number=f"{rng.randint(100, 999)}",
        expense_category=category,
        address_lines=list(address),
        font_name=rng.choice(FONTS),
        index=index,
    )


def _build_my(index: int, rng: random.Random) -> ReceiptSpec:
    vendor, category, address = rng.choice(MY_VENDORS)
    items = _items(rng, rng.randint(1, 3))
    transaction_date, date_text = _date(rng)

    net_sales = round(sum(item.amount for item in items) / 20, 2)
    tax_amount = round(net_sales * MY_SST_RATE, 2)
    total = round(net_sales + tax_amount, 2)

    return ReceiptSpec(
        template="my_pos",
        vendor_name=vendor,
        country="MY",
        currency="MYR",
        transaction_date=transaction_date,
        date_text=date_text,
        items=items,
        net_sales=net_sales,
        tax_amount=tax_amount,
        total_sales=total,
        service_charge=None,
        total_amount=total,
        tax_type="TAX",
        tax_rate=MY_SST_RATE,
        vat_classification=None,
        invoice_number=f"{rng.randint(10000, 99999)}",
        expense_category=category,
        address_lines=list(address),
        font_name=rng.choice(FONTS),
        index=index,
    )
