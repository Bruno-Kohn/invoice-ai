"""PaddleOCR engine wrapper."""

import numpy as np
from paddleocr import PaddleOCR as _PaddleOCR

from src.ocr.engine import BoundingBox, OCREngine, OCRResult


class PaddleOCREngine(OCREngine):
    """OCR engine using PaddleOCR (detection + recognition)."""

    def __init__(
        self,
        language: str = "en",
        use_angle_cls: bool = True,
        use_gpu: bool = False,
        det_model_dir: str | None = None,
        rec_model_dir: str | None = None,
    ):
        kwargs = {"lang": language}
        if det_model_dir:
            kwargs["text_detection_model_dir"] = det_model_dir
        if rec_model_dir:
            kwargs["text_recognition_model_dir"] = rec_model_dir
        if use_angle_cls:
            kwargs["use_textline_orientation"] = True

        self.ocr = _PaddleOCR(**kwargs)

    def recognize(self, image: np.ndarray) -> list[OCRResult]:
        """Run PaddleOCR on an image.

        Args:
            image: Input image (BGR numpy array).

        Returns:
            List of OCRResult with text, confidence, and bounding boxes.
        """
        results = self.ocr.predict(image)

        ocr_results = []
        for result in results:
            for item in result.get("rec_texts", []):
                # New API: iterate over detection results
                pass

            # Handle different result formats
            if "dt_polys" in result and "rec_texts" in result:
                polys = result["dt_polys"]
                texts = result["rec_texts"]
                scores = result["rec_scores"]

                for poly, text, score in zip(polys, texts, scores):
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    bbox = BoundingBox(
                        x1=int(min(xs)),
                        y1=int(min(ys)),
                        x2=int(max(xs)),
                        y2=int(max(ys)),
                    )
                    ocr_results.append(OCRResult(
                        text=text,
                        confidence=float(score),
                        bbox=bbox,
                    ))

        return ocr_results
