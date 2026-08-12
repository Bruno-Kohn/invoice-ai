"""Run experiments comparing different pipeline configurations.

Usage:
    python scripts/run_experiments.py --exp 1 --samples 20
    python scripts/run_experiments.py --exp 1,2,3 --samples 20
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import character_error_rate, word_error_rate
from src.evaluation.visualization import plot_metric_comparison, plot_grouped_bar


def get_ground_truth_text(annotation_path: Path) -> str:
    """Extract ground truth text from CORD-v2 annotation."""
    with open(annotation_path) as f:
        data = json.load(f)

    texts = []
    for line in data.get("valid_line", []):
        line_texts = []
        for word in line.get("words", []):
            text = word.get("text", "").strip()
            if text:
                line_texts.append(text)
        if line_texts:
            texts.append(" ".join(line_texts))
    return "\n".join(texts)


def run_ocr_on_image(image: np.ndarray, engine) -> str:
    """Run OCR and return concatenated text."""
    from src.ocr.postprocessing import correct_numeric_chars, normalize_whitespace, merge_lines_by_proximity

    results = engine.recognize(image)
    merged = merge_lines_by_proximity(results)
    lines = []
    for r in merged:
        text = correct_numeric_chars(r.text)
        text = normalize_whitespace(text)
        if text:
            lines.append(text)
    return "\n".join(lines)


def preprocess_image(image: np.ndarray, config: dict) -> np.ndarray:
    """Apply preprocessing based on config flags."""
    from src.preprocessing.document_detector import detect_document
    from src.preprocessing.deskew import deskew
    from src.preprocessing.enhancement import enhance
    from src.preprocessing.transforms import apply_bilateral_filter

    img = image.copy()

    if config.get("detect_document", True):
        img = detect_document(img)
    if config.get("deskew", True):
        img = deskew(img)
    if config.get("bilateral_filter", True):
        img = apply_bilateral_filter(img)

    img = enhance(
        img,
        grayscale=True,
        clahe=config.get("clahe", True),
        adaptive_threshold=config.get("adaptive_threshold", False),
    )
    return img


def load_test_data(num_samples: int = 20):
    """Load test images and ground truth."""
    ann_dir = Path("data/raw/test/annotations")
    img_dir = Path("data/raw/test/images")

    samples = []
    for ann_path in sorted(ann_dir.glob("*.json"))[:num_samples]:
        img_path = img_dir / ann_path.name.replace(".json", ".png")
        if not img_path.exists():
            # Try jpg
            img_path = img_dir / ann_path.name.replace(".json", ".jpg")
        if not img_path.exists():
            continue

        gt_text = get_ground_truth_text(ann_path)
        if not gt_text.strip():
            continue

        samples.append({
            "ann_path": ann_path,
            "img_path": img_path,
            "gt_text": gt_text,
        })

    return samples


def run_config(samples, engine, config: dict, config_name: str) -> dict:
    """Run a pipeline config on all samples and compute metrics."""
    cer_scores = []
    wer_scores = []
    times = []

    for i, sample in enumerate(samples):
        img = cv2.imread(str(sample["img_path"]))
        if img is None:
            continue

        t0 = time.time()

        if config.get("preprocessing", True):
            processed = preprocess_image(img, config)
        else:
            processed = img

        # Ensure 3-channel image for OCR engines
        if len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        ocr_text = run_ocr_on_image(processed, engine)
        elapsed = time.time() - t0

        cer = character_error_rate(ocr_text, sample["gt_text"])
        wer = word_error_rate(ocr_text, sample["gt_text"])

        cer_scores.append(cer)
        wer_scores.append(wer)
        times.append(elapsed)

        print(f"    [{i+1}/{len(samples)}] CER={cer:.3f} WER={wer:.3f} [{elapsed:.1f}s]")

    return {
        "config": config_name,
        "cer_mean": float(np.mean(cer_scores)),
        "cer_median": float(np.median(cer_scores)),
        "wer_mean": float(np.mean(wer_scores)),
        "wer_median": float(np.median(wer_scores)),
        "avg_time": float(np.mean(times)),
        "cer_scores": [float(c) for c in cer_scores],
        "wer_scores": [float(w) for w in wer_scores],
    }


# ─────────────────────────────────────────────
# Experiment definitions
# ─────────────────────────────────────────────

def exp1_preprocessing(samples, engine):
    """Exp 1: With preprocessing vs without."""
    print("\n" + "=" * 60)
    print("EXP 1: Preprocessing vs No Preprocessing")
    print("=" * 60)

    configs = {
        "No Preprocessing": {"preprocessing": False},
        "Full Preprocessing": {
            "preprocessing": True, "detect_document": True,
            "deskew": True, "clahe": True, "bilateral_filter": True,
        },
    }

    results = {}
    for name, config in configs.items():
        print(f"\n  Running: {name}")
        results[name] = run_config(samples, engine, config, name)

    # Visualize
    labels = list(results.keys())
    cer_vals = [results[l]["cer_mean"] for l in labels]
    wer_vals = [results[l]["wer_mean"] for l in labels]

    plot_grouped_bar(
        labels,
        {"CER": cer_vals, "WER": wer_vals},
        title="Exp 1: Preprocessing vs No Preprocessing",
        ylabel="Error Rate",
        save_name="exp1_preprocessing",
    )

    return results


def exp2_clahe(samples, engine):
    """Exp 2: CLAHE vs no CLAHE."""
    print("\n" + "=" * 60)
    print("EXP 2: CLAHE vs No CLAHE")
    print("=" * 60)

    configs = {
        "Without CLAHE": {
            "preprocessing": True, "detect_document": True,
            "deskew": True, "clahe": False, "bilateral_filter": True,
        },
        "With CLAHE": {
            "preprocessing": True, "detect_document": True,
            "deskew": True, "clahe": True, "bilateral_filter": True,
        },
    }

    results = {}
    for name, config in configs.items():
        print(f"\n  Running: {name}")
        results[name] = run_config(samples, engine, config, name)

    labels = list(results.keys())
    cer_vals = [results[l]["cer_mean"] for l in labels]

    plot_metric_comparison(labels, cer_vals, "CER",
                           title="Exp 2: Effect of CLAHE", save_name="exp2_clahe")
    return results


def exp3_adaptive_threshold(samples, engine):
    """Exp 3: Adaptive Threshold vs without."""
    print("\n" + "=" * 60)
    print("EXP 3: Adaptive Threshold vs Without")
    print("=" * 60)

    configs = {
        "Without Adaptive Threshold": {
            "preprocessing": True, "detect_document": True,
            "deskew": True, "clahe": True, "bilateral_filter": True,
            "adaptive_threshold": False,
        },
        "With Adaptive Threshold": {
            "preprocessing": True, "detect_document": True,
            "deskew": True, "clahe": True, "bilateral_filter": True,
            "adaptive_threshold": True,
        },
    }

    results = {}
    for name, config in configs.items():
        print(f"\n  Running: {name}")
        results[name] = run_config(samples, engine, config, name)

    labels = list(results.keys())
    cer_vals = [results[l]["cer_mean"] for l in labels]

    plot_metric_comparison(labels, cer_vals, "CER",
                           title="Exp 3: Effect of Adaptive Threshold",
                           save_name="exp3_adaptive_threshold")
    return results


def exp4_deskew(samples, engine):
    """Exp 4: Deskew on vs off."""
    print("\n" + "=" * 60)
    print("EXP 4: Deskew On vs Off")
    print("=" * 60)

    configs = {
        "Without Deskew": {
            "preprocessing": True, "detect_document": True,
            "deskew": False, "clahe": True, "bilateral_filter": True,
        },
        "With Deskew": {
            "preprocessing": True, "detect_document": True,
            "deskew": True, "clahe": True, "bilateral_filter": True,
        },
    }

    results = {}
    for name, config in configs.items():
        print(f"\n  Running: {name}")
        results[name] = run_config(samples, engine, config, name)

    labels = list(results.keys())
    cer_vals = [results[l]["cer_mean"] for l in labels]

    plot_metric_comparison(labels, cer_vals, "CER",
                           title="Exp 4: Effect of Deskew",
                           save_name="exp4_deskew")
    return results


def exp5_ablation(samples, engine):
    """Exp 5: Ablation — full pipeline vs subsets."""
    print("\n" + "=" * 60)
    print("EXP 5: Ablation Study")
    print("=" * 60)

    configs = {
        "None": {"preprocessing": False},
        "Document Detection Only": {
            "preprocessing": True, "detect_document": True,
            "deskew": False, "clahe": False, "bilateral_filter": False,
        },
        "Det + Deskew": {
            "preprocessing": True, "detect_document": True,
            "deskew": True, "clahe": False, "bilateral_filter": False,
        },
        "Det + Deskew + CLAHE": {
            "preprocessing": True, "detect_document": True,
            "deskew": True, "clahe": True, "bilateral_filter": False,
        },
        "Full Pipeline": {
            "preprocessing": True, "detect_document": True,
            "deskew": True, "clahe": True, "bilateral_filter": True,
        },
    }

    results = {}
    for name, config in configs.items():
        print(f"\n  Running: {name}")
        results[name] = run_config(samples, engine, config, name)

    labels = list(results.keys())
    cer_vals = [results[l]["cer_mean"] for l in labels]

    plot_metric_comparison(labels, cer_vals, "CER",
                           title="Exp 5: Ablation — Cumulative Preprocessing",
                           save_name="exp5_ablation")
    return results


def exp6_ocr_comparison(samples):
    """Exp 6: PaddleOCR vs Tesseract."""
    print("\n" + "=" * 60)
    print("EXP 6: PaddleOCR vs Tesseract")
    print("=" * 60)

    from src.ocr.paddle_ocr import PaddleOCREngine
    from src.ocr.tesseract_ocr import TesseractOCREngine

    config = {
        "preprocessing": True, "detect_document": True,
        "deskew": True, "clahe": True, "bilateral_filter": True,
    }

    results = {}
    for name, engine_cls in [("PaddleOCR", PaddleOCREngine), ("Tesseract", TesseractOCREngine)]:
        print(f"\n  Running: {name}")
        engine = engine_cls(language="en" if name == "PaddleOCR" else "eng")
        results[name] = run_config(samples, engine, config, name)

    labels = list(results.keys())
    cer_vals = [results[l]["cer_mean"] for l in labels]
    wer_vals = [results[l]["wer_mean"] for l in labels]
    time_vals = [results[l]["avg_time"] for l in labels]

    plot_grouped_bar(
        labels,
        {"CER": cer_vals, "WER": wer_vals},
        title="Exp 6: PaddleOCR vs Tesseract",
        ylabel="Error Rate",
        save_name="exp6_ocr_comparison",
    )

    return results


EXPERIMENTS = {
    1: ("Preprocessing vs No Preprocessing", exp1_preprocessing),
    2: ("CLAHE vs No CLAHE", exp2_clahe),
    3: ("Adaptive Threshold", exp3_adaptive_threshold),
    4: ("Deskew On vs Off", exp4_deskew),
    5: ("Ablation Study", exp5_ablation),
    6: ("PaddleOCR vs Tesseract", exp6_ocr_comparison),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--exp", type=str, default="1",
                        help="Experiment numbers, comma-separated (e.g. '1,2,3')")
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    exp_nums = [int(x.strip()) for x in args.exp.split(",")]

    print(f"Loading {args.samples} test samples...")
    samples = load_test_data(args.samples)
    print(f"Loaded {len(samples)} samples\n")

    # Initialize PaddleOCR engine (shared across experiments except exp6)
    needs_paddle = any(e != 6 for e in exp_nums)
    engine = None
    if needs_paddle:
        print("Initializing PaddleOCR...")
        from src.ocr.paddle_ocr import PaddleOCREngine
        engine = PaddleOCREngine(language="en")

    all_results = {}
    for exp_num in exp_nums:
        if exp_num not in EXPERIMENTS:
            print(f"Unknown experiment: {exp_num}")
            continue

        name, func = EXPERIMENTS[exp_num]
        if exp_num == 6:
            results = func(samples)
        else:
            results = func(samples, engine)

        all_results[f"exp{exp_num}"] = results

    # Save all results
    output_path = Path("results/metrics/experiments.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results if any
    existing = {}
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
    existing.update(all_results)

    with open(output_path, "w") as f:
        json.dump(existing, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
    print("Figures saved to results/figures/")
