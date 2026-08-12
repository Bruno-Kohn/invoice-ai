"""Test both parsers (regex and LLM) against CORD-v2 ground truth.

Runs on a subset of test annotations, compares extracted fields
with ground truth, and reports F1 scores per field.
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsing.regex_parser import parse_receipt as regex_parse
from src.parsing.llm_parser import parse_receipt as llm_parse
from src.parsing.schema import Receipt


def load_ground_truth(annotation_path: Path) -> dict:
    """Load ground truth from a CORD-v2 annotation file."""
    with open(annotation_path) as f:
        data = json.load(f)

    gt = data.get("gt_parse", {})
    if isinstance(gt, str):
        gt = json.loads(gt)
    return gt


def get_ocr_text(annotation_path: Path) -> str:
    """Extract OCR text (concatenated words) from annotation."""
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


def extract_total_price(receipt: Receipt) -> str | None:
    """Get total price from parsed receipt."""
    if receipt.total:
        return receipt.total.total_price
    return None


def extract_menu_items(receipt: Receipt) -> list[dict]:
    """Get simplified menu items from parsed receipt."""
    items = []
    menu = receipt.menu if isinstance(receipt.menu, list) else [receipt.menu]
    for item in menu:
        items.append({
            "nm": (item.nm or "").strip().lower(),
            "price": (item.price or "").strip(),
        })
    return items


def compute_field_f1(predicted: list[str], ground_truth: list[str]) -> dict:
    """Compute precision, recall, F1 for a set of field values."""
    pred_set = set(v for v in predicted if v)
    gt_set = set(v for v in ground_truth if v)

    if not gt_set and not pred_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not gt_set:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    if not pred_set:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0}

    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(gt_set) if gt_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}


def test_parsers(
    annotations_dir: Path = Path("data/raw/test/annotations"),
    num_samples: int = 20,
    use_llm: bool = True,
):
    """Test regex and LLM parsers on a subset of ground truth.

    Args:
        annotations_dir: Path to CORD-v2 annotations.
        num_samples: Number of samples to test.
        use_llm: Whether to also test the LLM parser.
    """
    ann_files = sorted(annotations_dir.glob("*.json"))[:num_samples]
    print(f"Testing on {len(ann_files)} samples from {annotations_dir}\n")

    regex_results = {"total_exact": 0, "menu_f1": [], "time": []}
    llm_results = {"total_exact": 0, "menu_f1": [], "time": []}

    for i, ann_path in enumerate(ann_files):
        gt = load_ground_truth(ann_path)
        ocr_text = get_ocr_text(ann_path)

        if not ocr_text.strip():
            continue

        # Ground truth values
        gt_total = gt.get("total", {}).get("total_price", "")
        gt_menu = gt.get("menu", [])
        if isinstance(gt_menu, dict):
            gt_menu = [gt_menu]
        gt_prices = [item.get("price", "") for item in gt_menu if item.get("price")]

        # Regex parser
        t0 = time.time()
        regex_receipt = regex_parse(ocr_text)
        regex_time = time.time() - t0
        regex_results["time"].append(regex_time)

        regex_total = extract_total_price(regex_receipt) or ""
        if regex_total.replace(",", "").replace(".", "") == gt_total.replace(",", "").replace(".", ""):
            regex_results["total_exact"] += 1

        regex_prices = [item.get("price", "") for item in extract_menu_items(regex_receipt) if item.get("price")]
        f1_data = compute_field_f1(regex_prices, gt_prices)
        regex_results["menu_f1"].append(f1_data["f1"])

        # LLM parser
        if use_llm:
            t0 = time.time()
            try:
                llm_receipt = llm_parse(ocr_text)
                llm_time = time.time() - t0
                llm_results["time"].append(llm_time)

                llm_total = extract_total_price(llm_receipt) or ""
                if llm_total.replace(",", "").replace(".", "") == gt_total.replace(",", "").replace(".", ""):
                    llm_results["total_exact"] += 1

                llm_prices = [item.get("price", "") for item in extract_menu_items(llm_receipt) if item.get("price")]
                f1_data = compute_field_f1(llm_prices, gt_prices)
                llm_results["menu_f1"].append(f1_data["f1"])
            except Exception as e:
                # Retry once after waiting if rate limited
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"  Rate limited, waiting 65s...")
                    time.sleep(65)
                    try:
                        t0 = time.time()
                        llm_receipt = llm_parse(ocr_text)
                        llm_time = time.time() - t0
                        llm_results["time"].append(llm_time)

                        llm_total = extract_total_price(llm_receipt) or ""
                        if llm_total.replace(",", "").replace(".", "") == gt_total.replace(",", "").replace(".", ""):
                            llm_results["total_exact"] += 1

                        llm_prices = [item.get("price", "") for item in extract_menu_items(llm_receipt) if item.get("price")]
                        f1_data = compute_field_f1(llm_prices, gt_prices)
                        llm_results["menu_f1"].append(f1_data["f1"])
                    except Exception as e2:
                        print(f"  LLM error on {ann_path.name}: {e2}")
                        llm_results["time"].append(0)
                        llm_results["menu_f1"].append(0)
                else:
                    print(f"  LLM error on {ann_path.name}: {e}")
                    llm_results["time"].append(0)
                    llm_results["menu_f1"].append(0)

            # Rate limit: 20 req/day for free tier, space them out
            time.sleep(5)

        print(f"  [{i+1}/{len(ann_files)}] {ann_path.name} done")

    # Report
    n = len(ann_files)
    print(f"\n{'='*60}")
    print(f"RESULTS ({n} samples)")
    print(f"{'='*60}")

    print(f"\n{'Regex Parser':>20} | {'LLM Parser':>20}")
    print(f"{'-'*20}-+-{'-'*20}")

    regex_total_acc = regex_results["total_exact"] / n if n else 0
    regex_menu_f1 = sum(regex_results["menu_f1"]) / len(regex_results["menu_f1"]) if regex_results["menu_f1"] else 0
    regex_avg_time = sum(regex_results["time"]) / len(regex_results["time"]) if regex_results["time"] else 0

    print(f"{'Total Exact Match':>20}: {regex_total_acc:.1%}  |  ", end="")
    if use_llm:
        llm_total_acc = llm_results["total_exact"] / n if n else 0
        print(f"{llm_total_acc:.1%}")
    else:
        print("N/A")

    print(f"{'Menu Item F1':>20}: {regex_menu_f1:.3f}  |  ", end="")
    if use_llm:
        llm_menu_f1 = sum(llm_results["menu_f1"]) / len(llm_results["menu_f1"]) if llm_results["menu_f1"] else 0
        print(f"{llm_menu_f1:.3f}")
    else:
        print("N/A")

    print(f"{'Avg Latency':>20}: {regex_avg_time*1000:.1f}ms  |  ", end="")
    if use_llm:
        llm_avg_time = sum(llm_results["time"]) / len(llm_results["time"]) if llm_results["time"] else 0
        print(f"{llm_avg_time*1000:.1f}ms")
    else:
        print("N/A")

    # Save results
    results = {
        "num_samples": n,
        "regex": {
            "total_exact_match": regex_total_acc,
            "menu_item_f1": regex_menu_f1,
            "avg_latency_ms": regex_avg_time * 1000,
        },
    }
    if use_llm:
        results["llm"] = {
            "total_exact_match": llm_total_acc,
            "menu_item_f1": llm_menu_f1,
            "avg_latency_ms": llm_avg_time * 1000,
        }

    output_path = Path("results/metrics/parser_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM parser")
    args = parser.parse_args()

    test_parsers(num_samples=args.samples, use_llm=not args.no_llm)
