"""OCR post-processing: text correction and bounding box merging."""

import re
from src.ocr.engine import OCRResult


# Common OCR character substitution errors
CHAR_CORRECTIONS = {
    "O": "0",  # letter O → digit zero (in numeric context)
    "o": "0",
    "l": "1",  # lowercase L → digit one (in numeric context)
    "I": "1",  # uppercase I → digit one (in numeric context)
    "S": "5",  # S → 5 (in numeric context)
    "B": "8",  # B → 8 (in numeric context)
    "Z": "2",  # Z → 2 (in numeric context)
}


def correct_numeric_chars(text: str) -> str:
    """Fix common OCR errors in numeric strings.

    Only applies corrections to tokens that look like they should be numbers
    (contain at least one digit or currency symbol).

    Args:
        text: Input text string.

    Returns:
        Corrected text.
    """
    tokens = text.split()
    corrected = []

    for token in tokens:
        # If token contains digits or price-like patterns, apply corrections
        if re.search(r"\d", token) or re.match(r"^[.,\d\sOolISBZ]+$", token):
            for wrong, right in CHAR_CORRECTIONS.items():
                token = token.replace(wrong, right)
        corrected.append(token)

    return " ".join(corrected)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace: collapse multiple spaces, strip lines."""
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def merge_lines_by_proximity(
    results: list[OCRResult], y_threshold: int = 10
) -> list[OCRResult]:
    """Merge OCR results that are on the same line (similar y-coordinate).

    Args:
        results: List of OCR results.
        y_threshold: Maximum y-distance to consider same line.

    Returns:
        Merged results (one per line).
    """
    if not results:
        return []

    # Sort by y then x
    sorted_results = sorted(results, key=lambda r: (r.bbox.y1 if r.bbox else 0, r.bbox.x1 if r.bbox else 0))

    merged = []
    current_line: list[OCRResult] = [sorted_results[0]]

    for result in sorted_results[1:]:
        prev = current_line[-1]

        # Check if on the same line
        if result.bbox and prev.bbox:
            y_diff = abs(result.bbox.y1 - prev.bbox.y1)
            if y_diff <= y_threshold:
                current_line.append(result)
                continue

        # Flush current line
        merged.append(_merge_line(current_line))
        current_line = [result]

    # Flush last line
    if current_line:
        merged.append(_merge_line(current_line))

    return merged


def _merge_line(results: list[OCRResult]) -> OCRResult:
    """Merge multiple OCR results into a single line result."""
    # Sort left to right
    results = sorted(results, key=lambda r: r.bbox.x1 if r.bbox else 0)

    text = " ".join(r.text for r in results)
    confidence = sum(r.confidence for r in results) / len(results)

    # Compute bounding box spanning all results
    bboxes = [r.bbox for r in results if r.bbox]
    if bboxes:
        from src.ocr.engine import BoundingBox
        bbox = BoundingBox(
            x1=min(b.x1 for b in bboxes),
            y1=min(b.y1 for b in bboxes),
            x2=max(b.x2 for b in bboxes),
            y2=max(b.y2 for b in bboxes),
        )
    else:
        bbox = None

    return OCRResult(text=text, confidence=confidence, bbox=bbox)


def postprocess(results: list[OCRResult], merge_lines: bool = True) -> list[OCRResult]:
    """Apply full post-processing pipeline to OCR results.

    Args:
        results: Raw OCR results.
        merge_lines: Whether to merge results on the same line.

    Returns:
        Post-processed OCR results.
    """
    if merge_lines:
        results = merge_lines_by_proximity(results)

    # Apply text corrections
    processed = []
    for r in results:
        text = normalize_whitespace(r.text)
        text = correct_numeric_chars(text)
        processed.append(OCRResult(text=text, confidence=r.confidence, bbox=r.bbox))

    return [r for r in processed if r.text.strip()]
