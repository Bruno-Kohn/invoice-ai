"""Generate remaining report images: OCR bboxes and failure examples."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.document_detector import detect_document
from src.preprocessing.enhancement import enhance
from src.preprocessing.transforms import apply_bilateral_filter
from src.ocr.paddle_ocr import PaddleOCREngine

img_dir = Path("data/raw/test/images")
test_imgs = sorted(img_dir.glob("*.png"))
if not test_imgs:
    test_imgs = sorted(img_dir.glob("*.jpg"))

FIGURES_DIR = Path("results/figures")

# === 4. BOUNDING BOXES OCR ===
print("Initializing PaddleOCR...")
engine = PaddleOCREngine(language="en")
img = cv2.imread(str(test_imgs[0]))
processed = detect_document(img)
processed = apply_bilateral_filter(processed)
enhanced = enhance(processed, grayscale=True, clahe=True, adaptive_threshold=False)
enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

results = engine.recognize(enhanced_bgr)
vis_img = enhanced_bgr.copy()
for r in results:
    if r.bbox:
        cv2.rectangle(vis_img, (int(r.bbox.x1), int(r.bbox.y1)),
                      (int(r.bbox.x2), int(r.bbox.y2)), (0, 255, 0), 2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
ax1.imshow(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB))
ax1.set_title("Preprocessed Image", fontweight="bold")
ax1.axis("off")
ax2.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
ax2.set_title(f"OCR Bounding Boxes ({len(results)} detected)", fontweight="bold")
ax2.axis("off")
plt.suptitle("PaddleOCR: Detected Text Regions", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "report_ocr_bboxes.png", bbox_inches="tight")
print("4. OCR bboxes saved!")

# === 5. FAILURE EXAMPLE ===
img_bad = cv2.imread(str(test_imgs[7]))
results_bad = engine.recognize(img_bad)
ocr_text_bad = " ".join(r.text for r in results_bad)

ann_path = Path("data/raw/test/annotations") / test_imgs[7].name.replace(".png", ".json").replace(".jpg", ".json")
with open(ann_path) as f:
    ann = json.load(f)
gt_texts = []
for line in ann.get("valid_line", []):
    for word in line.get("words", []):
        t = word.get("text", "").strip()
        if t:
            gt_texts.append(t)
gt_text = " ".join(gt_texts)

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off")
text_content = f"""FAILURE EXAMPLE — Image: {test_imgs[7].name} (CER=0.743)

GROUND TRUTH:
{gt_text[:200]}...

OCR OUTPUT:
{ocr_text_bad[:200]}...

ANALYSIS:
- High CER indicates OCR failed significantly
- Likely causes: low contrast, small text, unusual font
"""
ax.text(0.05, 0.95, text_content, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
ax.set_title("Failure Example: OCR Error", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "report_failure_example.png", bbox_inches="tight")
print("5. Failure example saved!")
