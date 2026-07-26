"""Money resolution and arithmetic reconciliation.

Money has internal structure a model cannot fake consistently, which makes it the
strongest hallucination detector available. A guessed total will not satisfy
``net + tax + charges = total``; a correctly read one will.

Reconciliation is locale-parameterised. PH VAT is a fixed 12% and inclusive, so
the identity is a hard assertion. US sales tax is exclusive and varies by state,
so only the weaker additive identity is checked and the rate is derived rather
than validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.layout import LayoutScan
from app.core.locale import LocaleGuess
from app.core.numbers import normalize_money

PH_VAT_RATE = 0.12
TOLERANCE = 0.02
PLAUSIBLE_RATE_RANGE = (0.0, 0.30)

MONEY_FIELDS = (
    "net_sales", "tax_amount", "total_sales", "service_charge", "total_amount",
)


@dataclass
class MoneyResult:
    values: dict[str, float | None] = field(default_factory=dict)
    derived: set[str] = field(default_factory=set)
    evidence: dict[str, str] = field(default_factory=dict)
    reconciled: bool = False
    reconciliation_notes: list[str] = field(default_factory=list)
    tax_rate: float | None = None
    tax_type: str | None = None
    derivations: set[str] = field(default_factory=set)
    """Which gaps were filled arithmetically, so tautologies can be excluded.

    An identity that compares a derived value against the values it was derived
    from always passes and proves nothing. Counting those as successful
    reconciliation inflates confidence on receipts whose figures were never
    actually cross-checked - which is the opposite of what the gate is for.
    """

    def get(self, name: str) -> float | None:
        return self.values.get(name)


def resolve_money(scan: LayoutScan, locale: LocaleGuess) -> MoneyResult:
    country = locale.country
    result = MoneyResult()

    for name in MONEY_FIELDS:
        labeled = scan.first(name)
        if labeled is None:
            result.values[name] = None
            continue
        value = normalize_money(labeled.raw_token, country)
        result.values[name] = value
        if value is not None:
            result.evidence[name] = labeled.evidence

    _fill_gaps(result, scan, country)
    _reconcile(result, country)
    return result


def _fill_gaps(result: MoneyResult, scan: LayoutScan, country: str | None) -> None:
    """Derive missing figures arithmetically instead of letting a model guess.

    Deterministic arithmetic beats a 1.5B model at maths every time, and a derived
    value is auditable in a way a generated one is not.
    """
    net = result.get("net_sales")
    tax = result.get("tax_amount")
    total_sales = result.get("total_sales")
    total = result.get("total_amount")
    service = result.get("service_charge")

    is_ph = country == "PH"

    if is_ph and net is None and total_sales is not None:
        net = round(total_sales / (1 + PH_VAT_RATE), 2)
        result.values["net_sales"] = net
        result.derived.add("net_sales")
        result.derivations.add("net_from_sales")
        result.evidence["net_sales"] = f"derived: {total_sales} / 1.12"

    if is_ph and tax is None and net is not None:
        tax = round(net * PH_VAT_RATE, 2)
        result.values["tax_amount"] = tax
        result.derived.add("tax_amount")
        result.derivations.add("tax_from_net")
        result.evidence["tax_amount"] = f"derived: {net} x 0.12"

    if total_sales is None and net is not None and tax is not None:
        total_sales = round(net + tax, 2)
        result.values["total_sales"] = total_sales
        result.derived.add("total_sales")
        result.derivations.add("sales_from_net_tax")
        result.evidence["total_sales"] = f"derived: {net} + {tax}"

    if total_sales is None and tax is None and total is not None and service is None:
        # No tax and no service charge, so sales equal the amount paid. Common on
        # POS receipts that print a single 'Total' line and nothing else.
        result.values["total_sales"] = total
        result.derived.add("total_sales")
        result.derivations.add("sales_from_total")
        result.evidence["total_sales"] = f"derived: equals total {total} (no tax/charges)"
        total_sales = total

    if total is None and total_sales is not None:
        candidate = round(total_sales + (service or 0.0), 2)
        # Only accept a derived total if the figure is actually printed somewhere,
        # otherwise we would be inventing the single most consequential field.
        if _token_present(candidate, scan, country):
            result.values["total_amount"] = candidate
            result.derived.add("total_amount")
            result.derivations.add("total_from_sales")
            result.evidence["total_amount"] = (
                f"derived: {total_sales} + service {service or 0.0}, confirmed on page"
            )


def _token_present(value: float, scan: LayoutScan, country: str | None) -> bool:
    target = round(value, 2)
    for token in scan.money_tokens:
        parsed = normalize_money(token, country)
        if parsed is not None and abs(parsed - target) <= TOLERANCE:
            return True
    return False


def _reconcile(result: MoneyResult, country: str | None) -> None:
    net = result.get("net_sales")
    tax = result.get("tax_amount")
    total_sales = result.get("total_sales")
    total = result.get("total_amount")
    service = result.get("service_charge") or 0.0

    checks: list[bool] = []
    derivations = result.derivations

    if country == "PH":
        # tax_type is set later from the VAT classification, because it describes
        # the vendor's registration rather than whether an amount was legible. A
        # VAT-registered OR is still a VAT document when its handwritten figures
        # cannot be read.
        if net is not None and tax is not None:
            result.tax_rate = PH_VAT_RATE
            if "tax_from_net" in derivations:
                # tax was computed as net x 0.12, so re-checking it is circular.
                result.reconciliation_notes.append(
                    "PH VAT identity skipped: tax was derived from net"
                )
            else:
                ok = abs(net * PH_VAT_RATE - tax) <= max(TOLERANCE, net * 0.001)
                checks.append(ok)
                result.reconciliation_notes.append(
                    f"PH VAT identity net*0.12={net * PH_VAT_RATE:.2f} vs tax={tax:.2f}"
                    f" -> {'ok' if ok else 'FAIL'}"
                )
    elif country is not None:
        if tax is None:
            result.tax_type = None
        else:
            result.tax_type = "SALES_TAX" if country == "US" else "TAX"
        if net and tax is not None and net > 0:
            rate = tax / net
            plausible = PLAUSIBLE_RATE_RANGE[0] <= rate <= PLAUSIBLE_RATE_RANGE[1]
            checks.append(plausible)
            result.tax_rate = round(rate, 4)
            result.reconciliation_notes.append(
                f"derived rate {rate:.4f} -> {'plausible' if plausible else 'IMPLAUSIBLE'}"
            )

    additive_is_circular = (
        "sales_from_net_tax" in derivations
        or {"net_from_sales", "tax_from_net"} <= derivations
    )
    if net is not None and tax is not None and total_sales is not None:
        if additive_is_circular:
            result.reconciliation_notes.append(
                "net+tax identity skipped: figures derived from one another"
            )
        else:
            ok = abs(net + tax - total_sales) <= TOLERANCE
            checks.append(ok)
            result.reconciliation_notes.append(
                f"net+tax={net + tax:.2f} vs total_sales={total_sales:.2f}"
                f" -> {'ok' if ok else 'FAIL'}"
            )

    total_is_circular = bool({"sales_from_total", "total_from_sales"} & derivations)
    if total_sales is not None and total is not None:
        if total_is_circular:
            result.reconciliation_notes.append(
                "total identity skipped: total and sales derived from one another"
            )
        else:
            ok = abs(total_sales + service - total) <= TOLERANCE
            checks.append(ok)
            result.reconciliation_notes.append(
                f"total_sales+service={total_sales + service:.2f} vs total={total:.2f}"
                f" -> {'ok' if ok else 'FAIL'}"
            )

    # A receipt printing no tax should still add up: the net figure and the amount
    # paid must agree. This is the only identity available on tax-free slips, and
    # it catches digit misreads that would otherwise pass unchecked.
    if tax is None and net is not None and total is not None and not service:
        if "net_sales" in result.derived:
            # net came from the total, so the comparison is circular. When both are
            # printed the comparison is genuine evidence, even if they agree.
            result.reconciliation_notes.append(
                "no-tax identity skipped: net was derived from the total"
            )
        else:
            ok = abs(net - total) <= TOLERANCE
            checks.append(ok)
            result.reconciliation_notes.append(
                f"no-tax receipt: net={net:.2f} vs total={total:.2f}"
                f" -> {'ok' if ok else 'FAIL'}"
            )

    if not checks:
        result.reconciled = False
        result.reconciliation_notes.append("no identity could be checked")
    else:
        result.reconciled = all(checks)


def resolve_tax_type(
    country: str | None, vat_classification: str | None, tax_amount: float | None
) -> str | None:
    """Name the tax regime.

    For PH this follows the vendor's registration, so a VAT-registered receipt is
    typed ``VAT`` even when its amounts are illegible, and a non-VAT sales invoice
    is typed ``None``. Elsewhere there is no registration marker to read, so the
    type is only asserted when a tax figure was actually found.
    """
    if country == "PH":
        return "VAT" if vat_classification == "vat" else None
    if tax_amount is None:
        return None
    return "SALES_TAX" if country == "US" else "TAX"


def classify_tax(text: str, locale: LocaleGuess, tax_amount: float | None) -> str | None:
    """VAT classification, locale aware.

    Returns ``vat`` / ``non-vat`` for PH and ``None`` elsewhere - SERMS validates
    this field with ``in:vat,non-vat``, so any third value would 422 the whole
    callback. Non-PH semantics are carried by ``tax_type`` internally instead.
    """
    if locale.country != "PH":
        return None

    lowered = text.casefold()
    if "non-vat reg" in lowered or "non vat reg" in lowered:
        return "non-vat"
    if "not valid for claiming input tax" in lowered:
        return "non-vat"
    if "vat reg" in lowered:
        return "vat"
    if "vat-exempt sales" in lowered or "vat exempt sales" in lowered:
        # A printed breakdown label alone does not make a receipt VAT-exempt.
        return "vat" if tax_amount else "non-vat"
    if tax_amount:
        return "vat"
    return "non-vat"
