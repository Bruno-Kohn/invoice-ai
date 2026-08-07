"""Deskew: correct image rotation using Hough Transform or minAreaRect."""

import cv2
import numpy as np


def compute_skew_angle_hough(image: np.ndarray) -> float:
    """Estimate skew angle using Hough Line Transform.

    Args:
        image: Input BGR or grayscale image.

    Returns:
        Estimated skew angle in degrees.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        if line.ndim == 2:
            x1, y1, x2, y2 = line[0]
        else:
            x1, y1, x2, y2 = line
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only consider near-horizontal lines
        if abs(angle) < 45:
            angles.append(angle)

    if not angles:
        return 0.0

    return float(np.median(angles))


def compute_skew_angle_minarea(image: np.ndarray) -> float:
    """Estimate skew angle using minAreaRect on non-zero pixels.

    Args:
        image: Input BGR or grayscale image.

    Returns:
        Estimated skew angle in degrees.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))

    if len(coords) < 10:
        return 0.0

    angle = cv2.minAreaRect(coords)[-1]

    # Normalize angle
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    return float(angle)


def deskew(image: np.ndarray, method: str = "hough", max_angle: float = 45.0) -> np.ndarray:
    """Correct image rotation.

    Args:
        image: Input BGR image.
        method: "hough" or "minAreaRect".
        max_angle: Maximum angle to correct (ignore larger rotations).

    Returns:
        Deskewed image.
    """
    if method == "hough":
        angle = compute_skew_angle_hough(image)
    elif method == "minAreaRect":
        angle = compute_skew_angle_minarea(image)
    else:
        raise ValueError(f"Unknown method: {method}")

    if abs(angle) < 5.0 or abs(angle) > max_angle:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute new bounding dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
