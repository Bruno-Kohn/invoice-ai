"""End-to-end pipeline: Image → Preprocessing → Quality Gate → OCR → Parsing → JSON.

Orchestrates all components into a single configurable pipeline.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.preprocessing.document_detector import detect_document
from src.preprocessing.deskew import deskew
from src.preprocessing.enhancement import enhance
from src.preprocessing.transforms import apply_bilateral_filter
from src.ocr.engine import OCREngine, OCRResult
from src.ocr.paddle_ocr import PaddleOCREngine
from src.ocr.tesseract_ocr import TesseractOCREngine
from src.ocr.postprocessing import correct_numeric_chars, normalize_whitespace, merge_lines_by_proximity
from src.parsing.schema import Receipt
from src.parsing.regex_parser import parse_receipt as regex_parse
from src.parsing.llm_parser import parse_receipt as llm_parse


@dataclass
class PipelineConfig:
    """Configuration for the extraction pipeline."""

    # Preprocessing
    preprocessing: bool = True
    detect_document: bool = True
    deskew: bool = True
    clahe: bool = True
    adaptive_threshold: bool = False
    bilateral_filter: bool = True

    # Quality gate
    quality_gate: bool = False
    quality_model_path: Optional[str] = None
    quality_threshold: float = 0.5
    quality_device: str = "cpu"

    # OCR
    ocr_engine: str = "paddleocr"  # "paddleocr" or "tesseract"
    ocr_language: str = "en"

    # Parsing
    parser: str = "regex"  # "regex" or "llm"
    llm_model: str = "gemini-3.5-flash"
    llm_few_shot: bool = False
    llm_api_key: Optional[str] = None


@dataclass
class PipelineResult:
    """Result from a single pipeline run."""

    receipt: Optional[Receipt] = None
    ocr_text: str = ""
    quality_label: Optional[str] = None
    quality_score: Optional[float] = None
    rejected: bool = False
    timings: dict = field(default_factory=dict)
    error: Optional[str] = None


class InvoicePipeline:
    """End-to-end document extraction pipeline.

    Usage:
        pipeline = InvoicePipeline(config)
        result = pipeline.process(image)
        print(result.receipt.model_dump_json())
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self._ocr_engine: Optional[OCREngine] = None
        self._quality_model = None

    def _get_ocr_engine(self) -> OCREngine:
        """Lazy-initialize OCR engine."""
        if self._ocr_engine is None:
            if self.config.ocr_engine == "paddleocr":
                self._ocr_engine = PaddleOCREngine(language=self.config.ocr_language)
            elif self.config.ocr_engine == "tesseract":
                self._ocr_engine = TesseractOCREngine(language=self.config.ocr_language)
            else:
                raise ValueError(f"Unknown OCR engine: {self.config.ocr_engine}")
        return self._ocr_engine

    def _get_quality_model(self):
        """Lazy-initialize quality CNN."""
        if self._quality_model is None and self.config.quality_gate:
            import torch
            from src.quality.model import QualityClassifier
            from src.quality.dataset import default_transform, IDX_TO_LABEL

            self._quality_model = QualityClassifier(num_classes=3, pretrained=False)
            state = torch.load(
                self.config.quality_model_path,
                map_location=self.config.quality_device,
                weights_only=True,
            )
            self._quality_model.load_state_dict(state)
            self._quality_model.to(self.config.quality_device)
            self._quality_model.eval()
            self._transform = default_transform()
            self._idx_to_label = IDX_TO_LABEL
        return self._quality_model

    def process(self, image: np.ndarray) -> PipelineResult:
        """Process a single image through the full pipeline.

        Args:
            image: BGR image (as loaded by cv2.imread).

        Returns:
            PipelineResult with extracted receipt, timings, etc.
        """
        result = PipelineResult()
        timings = {}

        try:
            # 1. Preprocessing
            t0 = time.time()
            processed = self._preprocess(image) if self.config.preprocessing else image
            timings["preprocessing"] = time.time() - t0

            # 2. Quality gate
            if self.config.quality_gate:
                t0 = time.time()
                label, score = self._assess_quality(image)
                timings["quality_gate"] = time.time() - t0
                result.quality_label = label
                result.quality_score = score

                if label == "not_ready" and score > self.config.quality_threshold:
                    result.rejected = True
                    result.timings = timings
                    return result

            # 3. OCR
            t0 = time.time()
            ocr_results = self._run_ocr(processed)
            ocr_text = self._postprocess_ocr(ocr_results)
            timings["ocr"] = time.time() - t0
            result.ocr_text = ocr_text

            # 4. Parsing
            t0 = time.time()
            receipt = self._parse(ocr_text)
            timings["parsing"] = time.time() - t0
            result.receipt = receipt

        except Exception as e:
            result.error = str(e)

        result.timings = timings
        return result

    def process_file(self, image_path: str | Path) -> PipelineResult:
        """Process an image file."""
        image = cv2.imread(str(image_path))
        if image is None:
            return PipelineResult(error=f"Could not read image: {image_path}")
        return self.process(image)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline."""
        img = image.copy()

        if self.config.detect_document:
            img = detect_document(img)

        if self.config.deskew:
            img = deskew(img)

        if self.config.bilateral_filter:
            img = apply_bilateral_filter(img)

        img = enhance(
            img,
            grayscale=True,
            clahe=self.config.clahe,
            adaptive_threshold=self.config.adaptive_threshold,
        )

        return img

    def _assess_quality(self, image: np.ndarray) -> tuple[str, float]:
        """Run quality CNN on the original image."""
        import torch

        model = self._get_quality_model()
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self._transform(img_rgb).unsqueeze(0).to(self.config.quality_device)

        probs = model.predict_proba(tensor)[0]
        pred_idx = torch.argmax(probs).item()
        label = self._idx_to_label[pred_idx]
        score = probs[pred_idx].item()

        return label, score

    def _run_ocr(self, image: np.ndarray) -> list[OCRResult]:
        """Run OCR engine."""
        engine = self._get_ocr_engine()
        return engine.recognize(image)

    def _postprocess_ocr(self, results: list[OCRResult]) -> str:
        """Post-process OCR results into clean text."""
        merged = merge_lines_by_proximity(results)
        lines = []
        for r in merged:
            text = correct_numeric_chars(r.text)
            text = normalize_whitespace(text)
            if text:
                lines.append(text)
        return "\n".join(lines)

    def _parse(self, ocr_text: str) -> Receipt:
        """Parse OCR text into structured Receipt."""
        if self.config.parser == "llm":
            return llm_parse(
                ocr_text,
                few_shot=self.config.llm_few_shot,
                api_key=self.config.llm_api_key,
                model_name=self.config.llm_model,
            )
        else:
            return regex_parse(ocr_text)
