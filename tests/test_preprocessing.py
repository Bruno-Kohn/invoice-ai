"""Unit tests for preprocessing modules."""

import numpy as np
import pytest

from src.preprocessing.enhancement import (
    to_grayscale,
    apply_clahe,
    apply_gaussian_blur,
    apply_adaptive_threshold,
    enhance,
)
from src.preprocessing.transforms import apply_morphological_closing, apply_bilateral_filter
from src.preprocessing.deskew import compute_skew_angle_hough, deskew
from src.preprocessing.document_detector import order_points, detect_document


# ─── Fixtures ───────────────────────────────────────────

@pytest.fixture
def sample_color_image():
    """Create a 100x100 BGR test image."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_gray_image():
    """Create a 100x100 grayscale test image."""
    return np.random.randint(0, 255, (100, 100), dtype=np.uint8)


# ─── Enhancement Tests ──────────────────────────────────

class TestEnhancement:
    def test_to_grayscale_from_color(self, sample_color_image):
        result = to_grayscale(sample_color_image)
        assert result.ndim == 2
        assert result.shape == (100, 100)

    def test_to_grayscale_already_gray(self, sample_gray_image):
        result = to_grayscale(sample_gray_image)
        assert result.ndim == 2
        np.testing.assert_array_equal(result, sample_gray_image)

    def test_apply_clahe(self, sample_gray_image):
        result = apply_clahe(sample_gray_image)
        assert result.shape == sample_gray_image.shape
        assert result.dtype == np.uint8

    def test_apply_gaussian_blur(self, sample_gray_image):
        result = apply_gaussian_blur(sample_gray_image, kernel_size=3)
        assert result.shape == sample_gray_image.shape

    def test_apply_adaptive_threshold(self, sample_gray_image):
        result = apply_adaptive_threshold(sample_gray_image)
        assert result.shape == sample_gray_image.shape
        # Should be binary
        unique = np.unique(result)
        assert len(unique) <= 2

    def test_enhance_returns_image(self, sample_color_image):
        result = enhance(sample_color_image, grayscale=True, clahe=True)
        assert result.ndim == 2
        assert result.shape[:2] == sample_color_image.shape[:2]

    def test_enhance_no_grayscale(self, sample_color_image):
        result = enhance(sample_color_image, grayscale=False, clahe=False, adaptive_threshold=False)
        assert result.ndim == 3


# ─── Transforms Tests ───────────────────────────────────

class TestTransforms:
    def test_bilateral_filter(self, sample_color_image):
        result = apply_bilateral_filter(sample_color_image)
        assert result.shape == sample_color_image.shape

    def test_morphological_closing(self, sample_gray_image):
        result = apply_morphological_closing(sample_gray_image)
        assert result.shape == sample_gray_image.shape


# ─── Deskew Tests ───────────────────────────────────────

class TestDeskew:
    def test_deskew_returns_image(self, sample_color_image):
        result = deskew(sample_color_image)
        assert result.ndim == 3

    def test_deskew_preserves_dtype(self, sample_color_image):
        result = deskew(sample_color_image)
        assert result.dtype == np.uint8


# ─── Document Detector Tests ────────────────────────────

class TestDocumentDetector:
    def test_order_points(self):
        pts = np.array([[10, 10], [100, 10], [100, 100], [10, 100]], dtype=np.float32)
        ordered = order_points(pts)
        assert ordered.shape == (4, 2)
        # Top-left should have smallest sum
        assert ordered[0].sum() <= ordered[2].sum()

    def test_detect_document_returns_image(self, sample_color_image):
        result = detect_document(sample_color_image)
        assert result.ndim == 3
        assert result.dtype == np.uint8
