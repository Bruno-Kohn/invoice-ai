"""Regex-based parser to extract structured receipt data from OCR text.

Parses OCR output into a Receipt schema using pattern matching for:
- Menu items (name, quantity, price)
- Subtotal, tax, service charges
- Total amount
"""

import re
from typing import Optional

from src.parsing.schema import LineItem, SubTotal, Total, Receipt


def parse_receipt(ocr_text: str) -> Receipt:
    """Parse raw OCR text into a structured Receipt.

    Args:
        ocr_text: Raw text from OCR engine (lines joined by newline).

    Returns:
        Receipt with extracted menu items, subtotal, and total.
    """
    lines = [line.strip() for line in ocr_text.split("\n") if line.strip()]

    menu_items = _extract_menu_items(lines)
    sub_total = _extract_subtotal(lines)
    total = _extract_total(lines)

    return Receipt(
        menu=menu_items if menu_items else [],
        sub_total=sub_total,
        total=total,
    )


def _extract_menu_items(lines: list[str]) -> list[LineItem]:
    """Extract menu items from OCR lines.

    Looks for patterns like:
    - "Item Name  2 x  10,000"
    - "Item Name  10,000"
    - "Item Name  2x  @5,000  10,000"
    """
    items = []

    # Pattern: name + optional quantity + price at end of line
    pattern = re.compile(
        r"^(.+?)\s+"           # item name (non-greedy)
        r"(?:(\d+)\s*x\s+)?"  # optional quantity (e.g. "2 x" or "2x")
        r"(?:@?\s*[\d.,]+\s+)?"  # optional unit price
        r"([\d.,]+)\s*$",      # price at end
        re.IGNORECASE,
    )

    # Keywords that indicate non-menu lines
    skip_keywords = {
        "total", "subtotal", "sub total", "tax", "service", "change",
        "cash", "credit", "disc", "discount", "ppn", "pbr", "charge",
        "payment", "member", "date", "time", "table", "cashier",
        "receipt", "invoice", "order", "bill", "thank", "terima",
        # PT-BR keywords
        "valor", "troco", "dinheiro", "cartao", "cartão", "debito", "débito",
        "credito", "crédito", "cpf", "cnpj", "nf-e", "nfce", "cupom",
        "fiscal", "icms", "pis", "cofins", "operador", "caixa",
        "obrigado", "volte", "sempre", "consumidor", "emissao", "emissão",
        "chave", "acesso", "protocolo", "serie", "série", "numero", "número",
    }

    for line in lines:
        lower = line.lower()
        if any(kw in lower for kw in skip_keywords):
            continue

        match = pattern.match(line)
        if match:
            name = match.group(1).strip()
            qty = match.group(2)
            price = match.group(3)

            # Skip if name is too short or looks like a number
            if len(name) < 2 or re.match(r"^[\d.,\s]+$", name):
                continue

            items.append(LineItem(
                nm=name,
                cnt=f"{qty} x" if qty else None,
                price=price,
            ))

    return items


def _extract_subtotal(lines: list[str]) -> Optional[SubTotal]:
    """Extract subtotal, tax, service from OCR lines."""
    subtotal_price = _find_price_for_keyword(lines, ["subtotal", "sub total", "sub-total", "valor produtos"])
    tax_price = _find_price_for_keyword(lines, ["tax", "ppn", "pajak", "vat", "icms", "imposto", "tributos"])
    service_price = _find_price_for_keyword(lines, ["service", "servis", "svc", "taxa", "servico", "serviço"])
    discount_price = _find_price_for_keyword(lines, ["discount", "disc", "diskon", "desconto", "desc"])

    if any([subtotal_price, tax_price, service_price, discount_price]):
        return SubTotal(
            subtotal_price=subtotal_price,
            tax_price=tax_price,
            service_price=service_price,
            discount_price=discount_price,
        )
    return None


def _extract_total(lines: list[str]) -> Total:
    """Extract total, cash, and change from OCR lines."""
    total_price = _find_price_for_keyword(lines, ["total", "grand total", "jumlah", "valor total", "vlr total"])
    cash_price = _find_price_for_keyword(lines, ["cash", "tunai", "bayar", "dinheiro", "especie", "espécie"])
    change_price = _find_price_for_keyword(lines, ["change", "kembalian", "kembali", "troco"])

    return Total(
        total_price=total_price,
        cashprice=cash_price,
        changeprice=change_price,
    )


def _find_price_for_keyword(lines: list[str], keywords: list[str]) -> Optional[str]:
    """Find a price value on the same line as a keyword.

    Args:
        lines: OCR text lines.
        keywords: Keywords to search for (case-insensitive).

    Returns:
        The price string if found, else None.
    """
    price_pattern = re.compile(r"([\d]+[.,\d]*[\d]+)")

    for line in lines:
        lower = line.lower()
        for kw in keywords:
            if kw in lower:
                # Find rightmost price on the line
                prices = price_pattern.findall(line)
                if prices:
                    return prices[-1]
    return None
