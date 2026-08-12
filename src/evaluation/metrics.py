"""Evaluation metrics for OCR and parsing quality.

Implements standard metrics for document extraction evaluation:
- CER (Character Error Rate)
- WER (Word Error Rate)
- F1 Score (per-field and macro)
- ANLS (Average Normalized Levenshtein Similarity)
- Exact Match
"""

import editdistance
import numpy as np


def character_error_rate(predicted: str, ground_truth: str) -> float:
    """Character Error Rate — edit distance normalized by GT length.

    CER = edit_distance(pred, gt) / len(gt)

    Args:
        predicted: OCR/parser output text.
        ground_truth: Reference text.

    Returns:
        CER as float (0.0 = perfect, >1.0 = very bad).
    """
    if not ground_truth:
        return 0.0 if not predicted else 1.0
    return editdistance.eval(predicted, ground_truth) / len(ground_truth)


def word_error_rate(predicted: str, ground_truth: str) -> float:
    """Word Error Rate — edit distance at word level.

    WER = edit_distance(pred_words, gt_words) / len(gt_words)

    Args:
        predicted: OCR/parser output text.
        ground_truth: Reference text.

    Returns:
        WER as float (0.0 = perfect).
    """
    gt_words = ground_truth.split()
    pred_words = predicted.split()

    if not gt_words:
        return 0.0 if not pred_words else 1.0

    return editdistance.eval(pred_words, gt_words) / len(gt_words)


def normalized_levenshtein_similarity(predicted: str, ground_truth: str) -> float:
    """Normalized Levenshtein Similarity (NLS).

    NLS = 1 - edit_distance(pred, gt) / max(len(pred), len(gt))

    Returns 1.0 for perfect match, 0.0 for completely different.
    """
    if not predicted and not ground_truth:
        return 1.0
    max_len = max(len(predicted), len(ground_truth))
    if max_len == 0:
        return 1.0
    return 1.0 - editdistance.eval(predicted, ground_truth) / max_len


def anls_score(predicted: str, ground_truth: str, threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity (ANLS).

    Used in document understanding benchmarks (DocVQA, etc.).
    Returns NLS if above threshold, else 0.0.

    Args:
        predicted: Predicted text.
        ground_truth: Ground truth text.
        threshold: Minimum NLS to count as a match.

    Returns:
        ANLS score (0.0 or NLS value).
    """
    nls = normalized_levenshtein_similarity(predicted, ground_truth)
    return nls if nls >= threshold else 0.0


def exact_match(predicted: str, ground_truth: str, normalize: bool = True) -> bool:
    """Check if predicted text exactly matches ground truth.

    Args:
        predicted: Predicted text.
        ground_truth: Ground truth text.
        normalize: If True, strip whitespace and lowercase before comparing.

    Returns:
        True if exact match.
    """
    if normalize:
        predicted = predicted.strip().lower()
        ground_truth = ground_truth.strip().lower()
    return predicted == ground_truth


def field_f1(
    predicted_fields: dict[str, str],
    ground_truth_fields: dict[str, str],
    threshold: float = 0.5,
) -> dict:
    """Compute per-field F1 using ANLS matching.

    Args:
        predicted_fields: Dict of field_name -> predicted_value.
        ground_truth_fields: Dict of field_name -> ground_truth_value.
        threshold: ANLS threshold for considering a match.

    Returns:
        Dict with per-field scores and macro averages.
    """
    all_fields = set(list(predicted_fields.keys()) + list(ground_truth_fields.keys()))

    tp = 0
    fp = 0
    fn = 0
    per_field = {}

    for field in all_fields:
        pred = (predicted_fields.get(field) or "").strip()
        gt = (ground_truth_fields.get(field) or "").strip()

        if not gt and not pred:
            continue

        if not gt:
            fp += 1
            per_field[field] = {"anls": 0.0, "match": False}
            continue

        if not pred:
            fn += 1
            per_field[field] = {"anls": 0.0, "match": False}
            continue

        score = anls_score(pred, gt, threshold)
        if score > 0:
            tp += 1
            per_field[field] = {"anls": score, "match": True}
        else:
            fp += 1
            fn += 1
            per_field[field] = {"anls": score, "match": False}

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_field": per_field,
    }


def receipt_to_fields(receipt_dict: dict) -> dict[str, str]:
    """Flatten a Receipt dict into a field_name -> value mapping.

    Converts nested Receipt structure into flat dict for comparison.

    Args:
        receipt_dict: Receipt as dict (from model_dump()).

    Returns:
        Flat dict like {"total_price": "96,800", "menu_0_nm": "Nasi Goreng", ...}
    """
    fields = {}

    # Total fields
    total = receipt_dict.get("total") or {}
    for key in ["total_price", "cashprice", "changeprice"]:
        val = total.get(key)
        if val:
            fields[f"total.{key}"] = str(val)

    # Subtotal fields
    sub = receipt_dict.get("sub_total") or {}
    for key in ["subtotal_price", "tax_price", "service_price", "discount_price"]:
        val = sub.get(key)
        if val:
            fields[f"sub_total.{key}"] = str(val)

    # Menu items
    menu = receipt_dict.get("menu") or []
    if isinstance(menu, dict):
        menu = [menu]
    for i, item in enumerate(menu):
        if isinstance(item, dict):
            for key in ["nm", "cnt", "price"]:
                val = item.get(key)
                if val:
                    fields[f"menu.{i}.{key}"] = str(val)

    return fields


def evaluate_receipt(
    predicted: dict,
    ground_truth: dict,
    anls_threshold: float = 0.5,
) -> dict:
    """Evaluate a predicted receipt against ground truth.

    Args:
        predicted: Predicted Receipt as dict.
        ground_truth: Ground truth Receipt as dict.
        anls_threshold: Threshold for ANLS matching.

    Returns:
        Dict with all metrics: CER, WER, F1, ANLS, exact matches.
    """
    pred_fields = receipt_to_fields(predicted)
    gt_fields = receipt_to_fields(ground_truth)

    # Field-level F1
    f1_result = field_f1(pred_fields, gt_fields, anls_threshold)

    # Total price exact match
    pred_total = (predicted.get("total") or {}).get("total_price", "")
    gt_total = (ground_truth.get("total") or {}).get("total_price", "")
    total_em = exact_match(str(pred_total or ""), str(gt_total or ""))

    # Text-level CER/WER (all fields concatenated)
    pred_text = " ".join(str(v) for v in pred_fields.values())
    gt_text = " ".join(str(v) for v in gt_fields.values())

    cer = character_error_rate(pred_text, gt_text)
    wer = word_error_rate(pred_text, gt_text)

    # ANLS per field
    anls_scores = []
    for field in gt_fields:
        pred_val = pred_fields.get(field, "")
        gt_val = gt_fields[field]
        anls_scores.append(anls_score(str(pred_val), str(gt_val), anls_threshold))
    mean_anls = float(np.mean(anls_scores)) if anls_scores else 0.0

    return {
        "cer": cer,
        "wer": wer,
        "f1": f1_result["f1"],
        "precision": f1_result["precision"],
        "recall": f1_result["recall"],
        "anls": mean_anls,
        "total_exact_match": total_em,
        "num_pred_fields": len(pred_fields),
        "num_gt_fields": len(gt_fields),
        "per_field": f1_result["per_field"],
    }
