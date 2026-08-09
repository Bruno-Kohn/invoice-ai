"""Assign quality labels to synthetic dataset images using real CER from OCR.

Runs PaddleOCR on each degraded image, compares with ground truth text,
calculates Character Error Rate (CER), and assigns labels:
- "ready": CER < 5%
- "marginal": 5% <= CER <= 20%
- "not_ready": CER > 20%
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_cer(predicted: str, ground_truth: str) -> float:
    """Compute Character Error Rate using edit distance.

    CER = edit_distance(pred, gt) / len(gt)

    Args:
        predicted: OCR output text.
        ground_truth: Reference text.

    Returns:
        CER as a float (0.0 = perfect, 1.0+ = very bad).
    """
    if not ground_truth:
        return 0.0 if not predicted else 1.0

    import editdistance
    distance = editdistance.eval(predicted, ground_truth)
    return distance / len(ground_truth)


def get_ground_truth_text(annotation_path: Path) -> str:
    """Extract all text from a CORD-v2 annotation as a single string.

    Args:
        annotation_path: Path to annotation JSON.

    Returns:
        Concatenated ground truth text.
    """
    with open(annotation_path) as f:
        data = json.load(f)

    # Extract text from valid_line (contains actual OCR ground truth)
    texts = []
    for line in data.get("valid_line", []):
        for word in line.get("words", []):
            text = word.get("text", "").strip()
            if text:
                texts.append(text)

    return " ".join(texts)


def assign_labels_with_cer(
    synthetic_dir: Path,
    annotations_dir: Path,
    output_path: Path,
    checkpoint_path: Path = None,
    checkpoint_interval: int = 50,
    cer_ready_threshold: float = 0.05,
    cer_not_ready_threshold: float = 0.20,
    limit: int = None,
):
    """Assign quality labels using real OCR CER.

    Supports checkpointing — saves progress every N images and resumes
    from where it left off if interrupted.

    Args:
        synthetic_dir: Directory with synthetic images.
        annotations_dir: Directory with CORD-v2 annotations.
        output_path: Path to save labels JSON.
        checkpoint_path: Path for checkpoint file. If None, uses output_path + ".checkpoint"
        checkpoint_interval: Save checkpoint every N images.
        cer_ready_threshold: CER below this = "ready".
        cer_not_ready_threshold: CER above this = "not_ready".
    """
    from src.ocr.paddle_ocr import PaddleOCREngine

    if checkpoint_path is None:
        checkpoint_path = output_path.with_suffix(".checkpoint.json")

    # Load existing checkpoint if available
    labels = []
    processed_paths = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            labels = json.load(f)
        processed_paths = {l["image_path"] for l in labels}
        print(f"Resuming from checkpoint: {len(labels)} images already processed")

    # Initialize OCR engine once
    print("Initializing PaddleOCR...")
    ocr = PaddleOCREngine(language="en")

    metadata_path = synthetic_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Build ground truth cache
    print("Loading ground truth texts...")
    gt_cache = {}
    for ann_path in sorted(annotations_dir.glob("*.json")):
        gt_text = get_ground_truth_text(ann_path)
        gt_cache[ann_path.stem] = gt_text

    # Build full task list
    tasks = []

    # Originals
    original_dir = synthetic_dir / "original"
    if original_dir.exists():
        for img_path in sorted(original_dir.glob("*.png")):
            rel_path = f"original/{img_path.name}"
            if rel_path not in processed_paths:
                tasks.append({
                    "img_path": img_path,
                    "rel_path": rel_path,
                    "degradation": "original",
                    "stem": img_path.stem,
                })

    # Degraded
    for entry in metadata:
        rel_path = entry["output"]
        if rel_path not in processed_paths:
            stem = Path(entry["original"]).stem
            tasks.append({
                "img_path": synthetic_dir / entry["output"],
                "rel_path": rel_path,
                "degradation": entry["degradation"],
                "stem": stem,
            })

    print(f"\nRemaining images to process: {len(tasks)}")
    print(f"Checkpoint saves every {checkpoint_interval} images")

    # Process with checkpointing
    new_count = 0
    for task in tqdm(tasks[:limit] if limit else tasks, desc="Computing CER"):
        gt_text = gt_cache.get(task["stem"], "")
        if not gt_text:
            continue

        img = cv2.imread(str(task["img_path"]))
        if img is None:
            continue

        results = ocr.recognize(img)
        predicted_text = " ".join(r.text for r in results)

        cer = compute_cer(predicted_text, gt_text)
        label = _cer_to_label(cer, cer_ready_threshold, cer_not_ready_threshold)

        labels.append({
            "image_path": task["rel_path"],
            "degradation": task["degradation"],
            "cer": round(cer, 4),
            "label": label,
        })

        new_count += 1

        # Save checkpoint periodically
        if new_count % checkpoint_interval == 0:
            checkpoint_path.write_text(json.dumps(labels, indent=2))
            tqdm.write(f"  [Checkpoint] Saved {len(labels)} labels ({new_count} new)")

    # Final save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(labels, indent=2))

    # Remove checkpoint file (no longer needed)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Print summary
    from collections import Counter
    label_counts = Counter(l["label"] for l in labels)
    cer_values = [l["cer"] for l in labels]

    print(f"\n{'='*50}")
    print(f"LABELING COMPLETE")
    print(f"{'='*50}")
    print(f"Total labeled images: {len(labels)}")
    print(f"  ready:     {label_counts['ready']:4d} ({100*label_counts['ready']/len(labels):.1f}%)")
    print(f"  marginal:  {label_counts['marginal']:4d} ({100*label_counts['marginal']/len(labels):.1f}%)")
    print(f"  not_ready: {label_counts['not_ready']:4d} ({100*label_counts['not_ready']/len(labels):.1f}%)")
    print(f"\nCER statistics:")
    print(f"  Mean: {np.mean(cer_values):.3f}")
    print(f"  Median: {np.median(cer_values):.3f}")
    print(f"  Min: {np.min(cer_values):.3f}, Max: {np.max(cer_values):.3f}")
    print(f"\nLabels saved to {output_path}")


def _cer_to_label(cer: float, ready_thresh: float, not_ready_thresh: float) -> str:
    """Convert CER value to quality label."""
    if cer < ready_thresh:
        return "ready"
    elif cer > not_ready_thresh:
        return "not_ready"
    else:
        return "marginal"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assign quality labels using real OCR CER")
    parser.add_argument("--synthetic-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--annotations-dir", type=Path, default=Path("data/raw/train/annotations"))
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/labels.json"))
    parser.add_argument("--cer-ready", type=float, default=0.05)
    parser.add_argument("--cer-not-ready", type=float, default=0.20)
    parser.add_argument("--limit", type=int, default=None,
                        help="Max images to process per run (for batch execution)")
    args = parser.parse_args()

    assign_labels_with_cer(
        synthetic_dir=args.synthetic_dir,
        annotations_dir=args.annotations_dir,
        output_path=args.output,
        cer_ready_threshold=args.cer_ready,
        cer_not_ready_threshold=args.cer_not_ready,
        limit=args.limit,
    )
