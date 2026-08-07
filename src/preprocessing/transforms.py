"""Morphological transforms: closing, bilateral filter."""

import cv2
import numpy as np


def apply_morphological_closing(
    image: np.ndarray, kernel_size: tuple = (2, 2), iterations: int = 1
) -> np.ndarray:
    """Apply morphological closing to reconnect broken characters.

    Args:
        image: Input image (grayscale or binary).
        kernel_size: Size of the structuring element.
        iterations: Number of times to apply the operation.

    Returns:
        Image after morphological closing.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def apply_bilateral_filter(
    image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75
) -> np.ndarray:
    """Apply bilateral filter to reduce noise while preserving edges.

    Args:
        image: Input image.
        d: Diameter of each pixel neighborhood.
        sigma_color: Filter sigma in the color space.
        sigma_space: Filter sigma in the coordinate space.

    Returns:
        Filtered image.
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
