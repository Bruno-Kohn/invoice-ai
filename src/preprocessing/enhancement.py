"""Image enhancement: CLAHE, Gaussian Blur, Adaptive Threshold."""

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        image: Grayscale image.
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization.

    Returns:
        Enhanced image.
    """
    gray = to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply Gaussian blur to reduce noise.

    Args:
        image: Input image.
        kernel_size: Kernel size (must be odd).

    Returns:
        Blurred image.
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_adaptive_threshold(
    image: np.ndarray,
    block_size: int = 11,
    constant: int = 2,
    method: str = "gaussian",
) -> np.ndarray:
    """Apply adaptive thresholding for binarization.

    Args:
        image: Grayscale image.
        block_size: Size of neighbourhood area (must be odd).
        constant: Constant subtracted from mean.
        method: "gaussian" or "mean".

    Returns:
        Binary image.
    """
    gray = to_grayscale(image)

    adaptive_method = (
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "gaussian"
        else cv2.ADAPTIVE_THRESH_MEAN_C
    )

    return cv2.adaptiveThreshold(
        gray, 255, adaptive_method, cv2.THRESH_BINARY, block_size, constant
    )


def enhance(
    image: np.ndarray,
    grayscale: bool = True,
    clahe: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: tuple = (8, 8),
    gaussian_blur: bool = True,
    gaussian_kernel_size: int = 3,
    adaptive_threshold: bool = True,
    threshold_block_size: int = 11,
    threshold_constant: int = 2,
    threshold_method: str = "gaussian",
) -> np.ndarray:
    """Apply full enhancement pipeline.

    Args:
        image: Input BGR image.
        Other args: Configuration for each step.

    Returns:
        Enhanced image (grayscale or binary depending on config).
    """
    result = image

    if grayscale:
        result = to_grayscale(result)

    if clahe:
        result = apply_clahe(result, clahe_clip_limit, clahe_tile_grid_size)

    if gaussian_blur:
        result = apply_gaussian_blur(result, gaussian_kernel_size)

    if adaptive_threshold:
        result = apply_adaptive_threshold(
            result, threshold_block_size, threshold_constant, threshold_method
        )

    return result
