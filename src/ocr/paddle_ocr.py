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
        self.ocr = _PaddleOCR(
            lang=language,
            use_angle_cls=use_angle_cls,
            use_gpu=use_gpu,
            det_model_dir=det_model_dir,
            rec_model_dir=rec_model_dir,
            show_log=False,
        )

    def recognize(self, image: np.ndarray) -> list[OCRResult]:
        """Run PaddleOCR on an image.

        Args:
            image: Input image (BGR numpy array).

        Returns:
            List of OCRResult with text, confidence, and bounding boxes.
        """
        results = self.ocr.ocr(image, cls=True)

        if not results or results[0] is None:
            return []

        ocr_results = []
        for line in results[0]:
            # line format: [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, confidence)]
            points, (text, confidence) = line

            # Convert 4-point polygon to axis-aligned bounding box
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            bbox = BoundingBox(
                x1=int(min(xs)),
                y1=int(min(ys)),
                x2=int(max(xs)),
                y2=int(max(ys)),
            )

            ocr_results.append(OCRResult(
                text=text,
                confidence=float(confidence),
                bbox=bbox,
            ))

        return ocr_results
