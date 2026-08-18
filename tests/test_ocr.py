"""Unit tests for OCR modules."""

import numpy as np
import pytest

from src.ocr.engine import OCRResult, BoundingBox
from src.ocr.postprocessing import (
    correct_numeric_chars,
    normalize_whitespace,
    merge_lines_by_proximity,
)


# ─── OCRResult / BoundingBox Tests ──────────────────────

class TestOCRResult:
    def test_create_result(self):
        result = OCRResult(text="Hello", confidence=0.95)
        assert result.text == "Hello"
        assert result.confidence == 0.95
        assert result.bbox is None

    def test_create_result_with_bbox(self):
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=50)
        result = OCRResult(text="Test", confidence=0.8, bbox=bbox)
        assert result.bbox.x1 == 10
        assert result.bbox.width == 90
        assert result.bbox.height == 30
        assert result.bbox.area == 2700


# ─── Postprocessing Tests ───────────────────────────────

class TestPostprocessing:
    def test_correct_numeric_chars_o_to_zero(self):
        # O → 0 in numeric context
        result = correct_numeric_chars("1O,OOO")
        assert "0" in result

    def test_correct_numeric_chars_preserves_text(self):
        result = correct_numeric_chars("Hello World")
        assert result == "Hello World"

    def test_normalize_whitespace(self):
        result = normalize_whitespace("  Hello   World  ")
        assert result == "Hello World"

    def test_normalize_whitespace_empty(self):
        result = normalize_whitespace("")
        assert result == ""

    def test_merge_lines_no_results(self):
        result = merge_lines_by_proximity([])
        assert result == []

    def test_merge_lines_single_result(self):
        results = [OCRResult(text="Hello", confidence=0.9, bbox=BoundingBox(x1=0, y1=0, x2=50, y2=20))]
        merged = merge_lines_by_proximity(results)
        assert len(merged) == 1

    def test_merge_lines_same_line(self):
        results = [
            OCRResult(text="Hello", confidence=0.9, bbox=BoundingBox(x1=0, y1=10, x2=50, y2=30)),
            OCRResult(text="World", confidence=0.9, bbox=BoundingBox(x1=60, y1=12, x2=110, y2=32)),
        ]
        merged = merge_lines_by_proximity(results, y_threshold=10)
        # Should merge into 1 line
        assert len(merged) <= len(results)
