"""Tesseract OCR engine wrapper."""

import numpy as np
import pytesseract

from src.ocr.engine import BoundingBox, OCREngine, OCRResult


class TesseractOCREngine(OCREngine):
    """OCR engine using Tesseract (baseline for comparison)."""

    def __init__(self, language: str = "eng", psm: int = 6, oem: int = 3):
        """Initialize Tesseract engine.

        Args:
            language: Tesseract language code.
            psm: Page segmentation mode (6 = uniform block of text).
            oem: OCR engine mode (3 = default, based on available).
        """
        self.language = language
        self.config = f"--psm {psm} --oem {oem}"

    def recognize(self, image: np.ndarray) -> list[OCRResult]:
        """Run Tesseract OCR on an image.

        Args:
            image: Input image (BGR or grayscale numpy array).

        Returns:
            List of OCRResult with text, confidence, and bounding boxes.
        """
        # Tesseract works best with grayscale/binary images
        if len(image.shape) == 3:
            import cv2
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Get detailed output with bounding boxes
        data = pytesseract.image_to_data(
            gray, lang=self.language, config=self.config, output_type=pytesseract.Output.DICT
        )

        ocr_results = []
        n_boxes = len(data["text"])

        for i in range(n_boxes):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])

            # Skip empty text and low-confidence noise
            if not text or conf < 0:
                continue

            bbox = BoundingBox(
                x1=data["left"][i],
                y1=data["top"][i],
                x2=data["left"][i] + data["width"][i],
                y2=data["top"][i] + data["height"][i],
            )

            ocr_results.append(OCRResult(
                text=text,
                confidence=conf / 100.0,  # normalize to 0-1
                bbox=bbox,
            ))

        return ocr_results
