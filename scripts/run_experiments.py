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


def exp7_cnn_filter(samples, engine):
    """Exp 7: Pipeline with CNN quality filter vs without."""
    print("\n" + "=" * 60)
    print("EXP 7: CNN Quality Filter vs No Filter")
    print("=" * 60)

    import torch
    from src.quality.model import QualityClassifier
    from src.quality.dataset import default_transform, IDX_TO_LABEL

    model_path = Path("models/quality_cnn/best_model.pth")
    if not model_path.exists():
        print("  ERROR: CNN model not found. Run training first.")
        return {}

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = QualityClassifier(num_classes=3, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    transform = default_transform()

    config = {
        "preprocessing": True, "detect_document": True,
        "deskew": True, "clahe": True, "bilateral_filter": True,
    }

    # Without filter — process all
    print("\n  Running: No CNN Filter (process all)")
    no_filter = run_config(samples, engine, config, "No Filter")

    # With filter — skip not_ready
    print("\n  Running: With CNN Filter (skip not_ready)")
    cer_scores = []
    wer_scores = []
    times = []
    rejected = 0

    for i, sample in enumerate(samples):
        img = cv2.imread(str(sample["img_path"]))
        if img is None:
            continue

        t0 = time.time()

        # Quality check
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transform(img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = model.predict_proba(tensor)[0]
        pred_idx = torch.argmax(probs).item()
        label = IDX_TO_LABEL[pred_idx]

        if label == "not_ready":
            rejected += 1
            elapsed = time.time() - t0
            times.append(elapsed)
            cer_scores.append(1.0)  # Worst case for rejected
            wer_scores.append(1.0)
            print(f"    [{i+1}/{len(samples)}] REJECTED (not_ready) [{elapsed:.1f}s]")
            continue

        processed = preprocess_image(img, config)
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

    with_filter = {
        "config": "With CNN Filter",
        "cer_mean": float(np.mean(cer_scores)),
        "wer_mean": float(np.mean(wer_scores)),
        "avg_time": float(np.mean(times)),
        "rejected": rejected,
        "rejected_pct": rejected / len(samples),
    }

    results = {"No Filter": no_filter, "With CNN Filter": with_filter}

    labels = list(results.keys())
    cer_vals = [results[l]["cer_mean"] for l in labels]
    plot_metric_comparison(labels, cer_vals, "CER",
                           title="Exp 7: CNN Quality Filter",
                           save_name="exp7_cnn_filter")
    return results


def exp8_cnn_thresholds(samples, engine):
    """Exp 8: CNN rejection thresholds (0.3, 0.5, 0.7)."""
    print("\n" + "=" * 60)
    print("EXP 8: CNN Rejection Thresholds")
    print("=" * 60)

    import torch
    from src.quality.model import QualityClassifier
    from src.quality.dataset import default_transform, IDX_TO_LABEL

    model_path = Path("models/quality_cnn/best_model.pth")
    if not model_path.exists():
        print("  ERROR: CNN model not found.")
        return {}

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = QualityClassifier(num_classes=3, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    transform = default_transform()

    config = {
        "preprocessing": True, "detect_document": True,
        "deskew": True, "clahe": True, "bilateral_filter": True,
    }

    # Pre-compute quality scores and OCR for all samples
    sample_data = []
    for i, sample in enumerate(samples):
        img = cv2.imread(str(sample["img_path"]))
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transform(img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = model.predict_proba(tensor)[0]
        not_ready_prob = probs[2].item()  # P(not_ready)

        processed = preprocess_image(img, config)
        if len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        ocr_text = run_ocr_on_image(processed, engine)

        cer = character_error_rate(ocr_text, sample["gt_text"])
        sample_data.append({"cer": cer, "not_ready_prob": not_ready_prob})
        print(f"    [{i+1}/{len(samples)}] CER={cer:.3f} P(not_ready)={not_ready_prob:.3f}")

    # Evaluate at different thresholds
    thresholds = [0.0, 0.3, 0.5, 0.7]
    results = {}
    for thresh in thresholds:
        accepted = [s for s in sample_data if s["not_ready_prob"] < thresh]
        rejected = len(sample_data) - len(accepted)
        avg_cer = float(np.mean([s["cer"] for s in accepted])) if accepted else 0.0

        name = f"Threshold={thresh}" if thresh > 0 else "No Filter"
        results[name] = {
            "config": name,
            "threshold": thresh,
            "cer_mean": avg_cer,
            "rejected": rejected,
            "rejected_pct": rejected / len(sample_data) if sample_data else 0,
            "accepted": len(accepted),
        }
        print(f"  {name}: CER={avg_cer:.3f}, rejected={rejected}/{len(sample_data)}")

    from src.evaluation.visualization import plot_line
    thresh_labels = [str(t) for t in thresholds]
    cer_vals = [results[n]["cer_mean"] for n in results]
    rejected_pcts = [results[n]["rejected_pct"] for n in results]

    plot_line(
        thresholds,
        {"CER (accepted only)": cer_vals, "Rejection Rate": rejected_pcts},
        title="Exp 8: CER vs CNN Rejection Threshold",
        xlabel="Threshold",
        ylabel="Score",
        save_name="exp8_cnn_thresholds",
    )
    return results


def exp9_regex_vs_llm(samples, engine):
    """Exp 9: Regex vs LLM parser (F1, latency)."""
    print("\n" + "=" * 60)
    print("EXP 9: Regex vs LLM Parser")
    print("=" * 60)

    from src.parsing.regex_parser import parse_receipt as regex_parse
    from src.parsing.llm_parser import parse_receipt as llm_parse
    from src.evaluation.metrics import evaluate_receipt

    config = {
        "preprocessing": True, "detect_document": True,
        "deskew": True, "clahe": True, "bilateral_filter": True,
    }

    # Get OCR text for all samples first
    print("\n  Running OCR on all samples...")
    ocr_texts = []
    gt_parses = []
    for i, sample in enumerate(samples):
        img = cv2.imread(str(sample["img_path"]))
        if img is None:
            continue
        processed = preprocess_image(img, config)
        if len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        ocr_text = run_ocr_on_image(processed, engine)
        ocr_texts.append(ocr_text)

        with open(sample["ann_path"]) as f:
            data = json.load(f)
        gt = data.get("gt_parse", {})
        if isinstance(gt, str):
            gt = json.loads(gt)
        gt_parses.append(gt)
        print(f"    [{i+1}/{len(samples)}] OCR done")

    # Regex parser
    print("\n  Running: Regex Parser")
    regex_metrics = {"f1": [], "anls": [], "total_em": [], "time": []}
    for i, (ocr_text, gt) in enumerate(zip(ocr_texts, gt_parses)):
        t0 = time.time()
        receipt = regex_parse(ocr_text)
        elapsed = time.time() - t0
        metrics = evaluate_receipt(receipt.model_dump(), gt)
        regex_metrics["f1"].append(metrics["f1"])
        regex_metrics["anls"].append(metrics["anls"])
        regex_metrics["total_em"].append(1.0 if metrics["total_exact_match"] else 0.0)
        regex_metrics["time"].append(elapsed)
        print(f"    [{i+1}/{len(ocr_texts)}] F1={metrics['f1']:.3f} ANLS={metrics['anls']:.3f}")

    # LLM parser
    print("\n  Running: LLM Parser")
    llm_metrics = {"f1": [], "anls": [], "total_em": [], "time": []}
    for i, (ocr_text, gt) in enumerate(zip(ocr_texts, gt_parses)):
        t0 = time.time()
        try:
            receipt = llm_parse(ocr_text)
            elapsed = time.time() - t0
            metrics = evaluate_receipt(receipt.model_dump(), gt)
            llm_metrics["f1"].append(metrics["f1"])
            llm_metrics["anls"].append(metrics["anls"])
            llm_metrics["total_em"].append(1.0 if metrics["total_exact_match"] else 0.0)
            llm_metrics["time"].append(elapsed)
            print(f"    [{i+1}/{len(ocr_texts)}] F1={metrics['f1']:.3f} ANLS={metrics['anls']:.3f}")
        except Exception as e:
            if "429" in str(e):
                print(f"    [{i+1}/{len(ocr_texts)}] Rate limited, waiting 65s...")
                time.sleep(65)
                try:
                    t0 = time.time()
                    receipt = llm_parse(ocr_text)
                    elapsed = time.time() - t0
                    metrics = evaluate_receipt(receipt.model_dump(), gt)
                    llm_metrics["f1"].append(metrics["f1"])
                    llm_metrics["anls"].append(metrics["anls"])
                    llm_metrics["total_em"].append(1.0 if metrics["total_exact_match"] else 0.0)
                    llm_metrics["time"].append(elapsed)
                    print(f"    [{i+1}/{len(ocr_texts)}] F1={metrics['f1']:.3f} (retry)")
                except Exception:
                    llm_metrics["f1"].append(0)
                    llm_metrics["anls"].append(0)
                    llm_metrics["total_em"].append(0)
                    llm_metrics["time"].append(0)
                    print(f"    [{i+1}/{len(ocr_texts)}] FAILED")
            else:
                llm_metrics["f1"].append(0)
                llm_metrics["anls"].append(0)
                llm_metrics["total_em"].append(0)
                llm_metrics["time"].append(0)
                print(f"    [{i+1}/{len(ocr_texts)}] ERROR: {e}")
        time.sleep(5)

    results = {
        "Regex": {
            "f1_mean": float(np.mean(regex_metrics["f1"])),
            "anls_mean": float(np.mean(regex_metrics["anls"])),
            "total_em": float(np.mean(regex_metrics["total_em"])),
            "avg_time_ms": float(np.mean(regex_metrics["time"])) * 1000,
        },
        "LLM (Gemini)": {
            "f1_mean": float(np.mean(llm_metrics["f1"])),
            "anls_mean": float(np.mean(llm_metrics["anls"])),
            "total_em": float(np.mean(llm_metrics["total_em"])),
            "avg_time_ms": float(np.mean(llm_metrics["time"])) * 1000,
        },
    }

    labels = ["Regex", "LLM (Gemini)"]
    plot_grouped_bar(
        labels,
        {
            "F1": [results[l]["f1_mean"] for l in labels],
            "ANLS": [results[l]["anls_mean"] for l in labels],
            "Total EM": [results[l]["total_em"] for l in labels],
        },
        title="Exp 9: Regex vs LLM Parser",
        ylabel="Score",
        save_name="exp9_regex_vs_llm",
    )
    return results


def exp10_llm_zeroshot_fewshot(samples, engine):
    """Exp 10: LLM zero-shot vs few-shot."""
    print("\n" + "=" * 60)
    print("EXP 10: LLM Zero-shot vs Few-shot")
    print("=" * 60)

    from src.parsing.llm_parser import parse_receipt as llm_parse
    from src.evaluation.metrics import evaluate_receipt

    config = {
        "preprocessing": True, "detect_document": True,
        "deskew": True, "clahe": True, "bilateral_filter": True,
    }

    # Get OCR text
    print("\n  Running OCR...")
    ocr_data = []
    for i, sample in enumerate(samples):
        img = cv2.imread(str(sample["img_path"]))
        if img is None:
            continue
        processed = preprocess_image(img, config)
        if len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        ocr_text = run_ocr_on_image(processed, engine)

        with open(sample["ann_path"]) as f:
            data = json.load(f)
        gt = data.get("gt_parse", {})
        if isinstance(gt, str):
            gt = json.loads(gt)
        ocr_data.append({"ocr_text": ocr_text, "gt": gt})
        print(f"    [{i+1}/{len(samples)}] OCR done")

    results = {}
    for mode_name, few_shot in [("Zero-shot", False), ("Few-shot", True)]:
        print(f"\n  Running: {mode_name}")
        f1_scores = []
        for i, item in enumerate(ocr_data):
            try:
                receipt = llm_parse(item["ocr_text"], few_shot=few_shot)
                metrics = evaluate_receipt(receipt.model_dump(), item["gt"])
                f1_scores.append(metrics["f1"])
                print(f"    [{i+1}/{len(ocr_data)}] F1={metrics['f1']:.3f}")
            except Exception as e:
                if "429" in str(e):
                    print(f"    [{i+1}/{len(ocr_data)}] Rate limited, waiting 65s...")
                    time.sleep(65)
                    try:
                        receipt = llm_parse(item["ocr_text"], few_shot=few_shot)
                        metrics = evaluate_receipt(receipt.model_dump(), item["gt"])
                        f1_scores.append(metrics["f1"])
                        print(f"    [{i+1}/{len(ocr_data)}] F1={metrics['f1']:.3f} (retry)")
                    except Exception:
                        f1_scores.append(0)
                        print(f"    [{i+1}/{len(ocr_data)}] FAILED")
                else:
                    f1_scores.append(0)
                    print(f"    [{i+1}/{len(ocr_data)}] ERROR")
            time.sleep(5)

        results[mode_name] = {"f1_mean": float(np.mean(f1_scores)), "f1_scores": f1_scores}

    labels = list(results.keys())
    f1_vals = [results[l]["f1_mean"] for l in labels]
    plot_metric_comparison(labels, f1_vals, "F1",
                           title="Exp 10: Zero-shot vs Few-shot",
                           save_name="exp10_zeroshot_fewshot")
    return results


def exp11_gt_vs_real_ocr(samples, engine):
    """Exp 11: Ground truth text + Parser vs Real OCR + Parser."""
    print("\n" + "=" * 60)
    print("EXP 11: GT Text vs Real OCR → Parser")
    print("=" * 60)

    from src.parsing.regex_parser import parse_receipt as regex_parse
    from src.evaluation.metrics import evaluate_receipt

    config = {
        "preprocessing": True, "detect_document": True,
        "deskew": True, "clahe": True, "bilateral_filter": True,
    }

    results = {}
    for source_name in ["Ground Truth Text", "Real OCR"]:
        print(f"\n  Running: {source_name}")
        f1_scores = []

        for i, sample in enumerate(samples):
            with open(sample["ann_path"]) as f:
                data = json.load(f)
            gt = data.get("gt_parse", {})
            if isinstance(gt, str):
                gt = json.loads(gt)

            if source_name == "Ground Truth Text":
                text = sample["gt_text"]
            else:
                img = cv2.imread(str(sample["img_path"]))
                if img is None:
                    continue
                processed = preprocess_image(img, config)
                if len(processed.shape) == 2:
                    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                text = run_ocr_on_image(processed, engine)

            receipt = regex_parse(text)
            metrics = evaluate_receipt(receipt.model_dump(), gt)
            f1_scores.append(metrics["f1"])
            print(f"    [{i+1}/{len(samples)}] F1={metrics['f1']:.3f}")

        results[source_name] = {
            "f1_mean": float(np.mean(f1_scores)),
            "f1_scores": [float(f) for f in f1_scores],
        }

    labels = list(results.keys())
    f1_vals = [results[l]["f1_mean"] for l in labels]
    plot_metric_comparison(labels, f1_vals, "F1",
                           title="Exp 11: GT Text vs Real OCR → Regex Parser",
                           save_name="exp11_gt_vs_ocr")
    return results


def exp12_full_vs_naive(samples, engine):
    """Exp 12: Full pipeline vs naive baseline (no preprocessing, no postprocessing)."""
    print("\n" + "=" * 60)
    print("EXP 12: Full Pipeline vs Naive Baseline")
    print("=" * 60)

    from src.parsing.regex_parser import parse_receipt as regex_parse
    from src.evaluation.metrics import evaluate_receipt

    results = {}
    configs = {
        "Naive (raw OCR)": {"preprocessing": False},
        "Full Pipeline": {
            "preprocessing": True, "detect_document": True,
            "deskew": True, "clahe": True, "bilateral_filter": True,
        },
    }

    for name, config in configs.items():
        print(f"\n  Running: {name}")
        f1_scores = []
        cer_scores = []

        for i, sample in enumerate(samples):
            img = cv2.imread(str(sample["img_path"]))
            if img is None:
                continue

            if config.get("preprocessing", True):
                processed = preprocess_image(img, config)
            else:
                processed = img
            if len(processed.shape) == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

            ocr_text = run_ocr_on_image(processed, engine)
            cer = character_error_rate(ocr_text, sample["gt_text"])
            cer_scores.append(cer)

            with open(sample["ann_path"]) as f:
                data = json.load(f)
            gt = data.get("gt_parse", {})
            if isinstance(gt, str):
                gt = json.loads(gt)

            receipt = regex_parse(ocr_text)
            metrics = evaluate_receipt(receipt.model_dump(), gt)
            f1_scores.append(metrics["f1"])
            print(f"    [{i+1}/{len(samples)}] CER={cer:.3f} F1={metrics['f1']:.3f}")

        results[name] = {
            "cer_mean": float(np.mean(cer_scores)),
            "f1_mean": float(np.mean(f1_scores)),
        }

    labels = list(results.keys())
    plot_grouped_bar(
        labels,
        {
            "CER": [results[l]["cer_mean"] for l in labels],
            "F1": [results[l]["f1_mean"] for l in labels],
        },
        title="Exp 12: Full Pipeline vs Naive Baseline",
        ylabel="Score",
        save_name="exp12_full_vs_naive",
    )
    return results


def exp13_resolutions(samples, engine):
    """Exp 13: Different image resolutions."""
    print("\n" + "=" * 60)
    print("EXP 13: Different Resolutions")
    print("=" * 60)

    config = {
        "preprocessing": True, "detect_document": True,
        "deskew": True, "clahe": True, "bilateral_filter": True,
    }

    results = {}
    for res_name, scale in [("25% (low)", 0.25), ("50% (medium)", 0.5), ("100% (original)", 1.0)]:
        print(f"\n  Running: {res_name}")
        cer_scores = []
        times = []

        for i, sample in enumerate(samples):
            img = cv2.imread(str(sample["img_path"]))
            if img is None:
                continue

            t0 = time.time()

            if scale < 1.0:
                h, w = img.shape[:2]
                img = cv2.resize(img, (int(w * scale), int(h * scale)))

            processed = preprocess_image(img, config)
            if len(processed.shape) == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            ocr_text = run_ocr_on_image(processed, engine)
            elapsed = time.time() - t0

            cer = character_error_rate(ocr_text, sample["gt_text"])
            cer_scores.append(cer)
            times.append(elapsed)
            print(f"    [{i+1}/{len(samples)}] CER={cer:.3f} [{elapsed:.1f}s]")

        results[res_name] = {
            "cer_mean": float(np.mean(cer_scores)),
            "avg_time": float(np.mean(times)),
            "cer_scores": [float(c) for c in cer_scores],
        }

    labels = list(results.keys())
    cer_vals = [results[l]["cer_mean"] for l in labels]
    plot_metric_comparison(labels, cer_vals, "CER",
                           title="Exp 13: CER by Image Resolution",
                           save_name="exp13_resolutions")
    return results


EXPERIMENTS = {
    1: ("Preprocessing vs No Preprocessing", exp1_preprocessing),
    2: ("CLAHE vs No CLAHE", exp2_clahe),
    3: ("Adaptive Threshold", exp3_adaptive_threshold),
    4: ("Deskew On vs Off", exp4_deskew),
    5: ("Ablation Study", exp5_ablation),
    6: ("PaddleOCR vs Tesseract", exp6_ocr_comparison),
    7: ("CNN Quality Filter", exp7_cnn_filter),
    8: ("CNN Rejection Thresholds", exp8_cnn_thresholds),
    9: ("Regex vs LLM", exp9_regex_vs_llm),
    10: ("LLM Zero-shot vs Few-shot", exp10_llm_zeroshot_fewshot),
    11: ("GT Text vs Real OCR", exp11_gt_vs_real_ocr),
    12: ("Full Pipeline vs Naive", exp12_full_vs_naive),
    13: ("Different Resolutions", exp13_resolutions),
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

    # Initialize PaddleOCR engine (shared across most experiments)
    standalone_exps = {6}  # These init their own engines
    needs_paddle = any(e not in standalone_exps for e in exp_nums)
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
        if exp_num in standalone_exps:
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
