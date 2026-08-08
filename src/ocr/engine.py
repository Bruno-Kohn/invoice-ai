"""Abstract OCR engine interface and shared data structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BoundingBox:
    """Bounding box for detected text region."""

    x1: int  # top-left x
    y1: int  # top-left y
    x2: int  # bottom-right x
    y2: int  # bottom-right y

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class OCRResult:
    """Single text detection + recognition result."""

    text: str
    confidence: float  # 0.0 to 1.0
    bbox: Optional[BoundingBox] = None


class OCREngine(ABC):
    """Abstract base class for OCR engines.

    All OCR implementations must follow this interface,
    allowing interchangeable use in the pipeline.
    """

    @abstractmethod
    def recognize(self, image: np.ndarray) -> list[OCRResult]:
        """Run OCR on an image and return detected text regions.

        Args:
            image: Input image (BGR or grayscale numpy array).

        Returns:
            List of OCRResult objects, each containing text, confidence, and bbox.
        """
        ...

    def recognize_text(self, image: np.ndarray) -> str:
        """Convenience method: run OCR and return full text as a single string.

        Args:
            image: Input image.

        Returns:
            All detected text concatenated with newlines (ordered top-to-bottom).
        """
        results = self.recognize(image)

        # Sort by vertical position (top to bottom)
        results_sorted = sorted(
            results, key=lambda r: r.bbox.y1 if r.bbox else 0
        )

        return "\n".join(r.text for r in results_sorted if r.text.strip())
