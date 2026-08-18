"""Unit tests for parsing modules."""

import pytest

from src.parsing.schema import LineItem, SubTotal, Total, Receipt
from src.parsing.regex_parser import parse_receipt


# ─── Schema Tests ───────────────────────────────────────

class TestSchema:
    def test_line_item_creation(self):
        item = LineItem(nm="Nasi Goreng", cnt="2 x", price="70,000")
        assert item.nm == "Nasi Goreng"
        assert item.cnt == "2 x"
        assert item.price == "70,000"

    def test_line_item_optional_fields(self):
        item = LineItem(nm="Item")
        assert item.cnt is None
        assert item.price is None
        assert item.unitprice is None

    def test_subtotal_creation(self):
        sub = SubTotal(subtotal_price="88,000", tax_price="8,800")
        assert sub.subtotal_price == "88,000"
        assert sub.tax_price == "8,800"
        assert sub.service_price is None

    def test_total_creation(self):
        total = Total(total_price="96,800")
        assert total.total_price == "96,800"

    def test_receipt_creation(self):
        receipt = Receipt(
            menu=[LineItem(nm="Item 1", price="10,000")],
            total=Total(total_price="10,000"),
        )
        assert len(receipt.menu) == 1
        assert receipt.total.total_price == "10,000"

    def test_receipt_menu_single_dict_to_list(self):
        """Menu as single dict should be converted to list."""
        receipt = Receipt(
            menu={"nm": "Item", "price": "5000"},
            total=Total(total_price="5000"),
        )
        assert isinstance(receipt.menu, list)
        assert len(receipt.menu) == 1


# ─── Regex Parser Tests ─────────────────────────────────

class TestRegexParser:
    def test_parse_simple_receipt(self):
        text = """Nasi Goreng  2 x  70,000
Ice Tea  1 x  18,000
Sub Total  88,000
Tax 10%  8,800
TOTAL  96,800"""
        receipt = parse_receipt(text)
        assert receipt.total.total_price is not None
        assert len(receipt.menu) >= 1

    def test_parse_extracts_total(self):
        text = "TOTAL  150,000"
        receipt = parse_receipt(text)
        assert receipt.total.total_price == "150,000"

    def test_parse_extracts_subtotal(self):
        text = """Item  10,000
Sub Total  10,000
Tax  1,000
TOTAL  11,000"""
        receipt = parse_receipt(text)
        assert receipt.sub_total is not None
        assert receipt.sub_total.subtotal_price == "10,000"
        assert receipt.sub_total.tax_price == "1,000"

    def test_parse_empty_text(self):
        receipt = parse_receipt("")
        assert receipt.total.total_price is None
        assert receipt.menu == []

    def test_parse_skips_non_menu_lines(self):
        text = """Thank you for visiting
Date: 2024-01-01
Cashier: John
Nasi Goreng  50,000
TOTAL  50,000"""
        receipt = parse_receipt(text)
        # Should not have "Thank you", "Date:", "Cashier:" as menu items
        for item in receipt.menu:
            assert "thank" not in (item.nm or "").lower()
            assert "date" not in (item.nm or "").lower()
            assert "cashier" not in (item.nm or "").lower()

    def test_parse_handles_no_total(self):
        text = "Nasi Goreng  50,000"
        receipt = parse_receipt(text)
        # Should still parse without crashing
        assert receipt is not None
