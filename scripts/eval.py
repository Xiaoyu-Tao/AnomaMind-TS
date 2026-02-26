# -*- coding: utf-8 -*-
"""
Evaluation script: parse model predictions and groundtruth, compute Precision, Recall, F1, BestF1.

Rules:
1. Folders prefixed with "700_" etc. belong to the same time series slice
2. Concatenate predictions and GT by sequence
3. Compute F1 per sequence
4. Output average Precision, Recall, F1, BestF1
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Union
from collections import defaultdict, Counter

import pandas as pd


def extract_sequence_id(folder_name: str) -> str:
    """Extract sequence ID from folder name.

    E.g. "697_YAHOO_id_147_WebService_tr_500_1st_225_seg008" -> "697"
    """
    parts = folder_name.split('_')
    return parts[0] if parts else "unknown"


def extract_segment_number(folder_name: str) -> int:
    """Extract segment number from folder name.

    E.g. "697_YAHOO_id_147_WebService_tr_500_1st_225_seg008" -> 8
    """
    parts = folder_name.split('_')
    for part in parts:
        if part.startswith('seg'):
            try:
                return int(part[3:])
            except ValueError:
                return -1
    return -1


def get_segment_start_idx(segment_data_file: Path) -> Optional[int]:
    """Read segment start index from segment_data.csv."""
    if segment_data_file is None or not segment_data_file.exists():
        return None
    try:
        segment_df = pd.read_csv(segment_data_file)
        if 'Index' in segment_df.columns:
            return int(segment_df['Index'].iloc[0])
    except Exception as e:
        print(f"Warning: failed to read segment_data.csv ({segment_data_file}): {e}")
    return None


def load_groundtruth(gt_file: Path) -> Dict[int, int]:
    """Load groundtruth file.

    Supports:
    1. New format: {"ground_truth": [[start, end], ...]} - global interval indices
    2. Old format: {"Index": [...], "Label": [...]} - segment-relative point indices

    Returns:
        {index: label} dict, label=1 for anomaly, 0 for normal (segment-relative indices)
    """
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    if "ground_truth" in gt_data:
        gt_file_dir = gt_file.parent
        segment_data_file = gt_file_dir / "segment_data.csv"
        if not segment_data_file.exists():
            print(f"Warning: segment_data.csv not found in {gt_file_dir}")
            return {}

        segment_df = pd.read_csv(segment_data_file)
        segment_length = len(segment_df)
        segment_start_idx = int(segment_df['Index'].iloc[0]) if 'Index' in segment_df.columns else 0
        segment_end_idx = int(segment_df['Index'].iloc[-1]) if 'Index' in segment_df.columns else segment_length - 1

        indices = list(range(segment_length))
        labels = [0] * segment_length
        ground_truth_intervals = gt_data.get("ground_truth", [])
        if not isinstance(ground_truth_intervals, list):
            ground_truth_intervals = []

        for gt_interval in ground_truth_intervals:
            if not isinstance(gt_interval, list) or len(gt_interval) < 2:
                continue
            global_start, global_end = int(gt_interval[0]), int(gt_interval[1])
            if global_end < segment_start_idx or global_start > segment_end_idx:
                continue
            global_start = max(global_start, segment_start_idx)
            global_end = min(global_end, segment_end_idx)
            relative_start = global_start - segment_start_idx
            relative_end = global_end - segment_start_idx
            if relative_start < 0 or relative_end >= segment_length:
                continue
            for idx in range(relative_start, relative_end + 1):
                if 0 <= idx < segment_length:
                    labels[idx] = 1

        return {int(idx): int(label) for idx, label in zip(indices, labels)}
    else:
        indices = gt_data.get("Index", [])
        labels = gt_data.get("Label", [])
        return {int(idx): int(label) for idx, label in zip(indices, labels)}


def load_prediction(
    result_file: Path,
    segment_data_file: Optional[Path] = None,
    with_confidence: bool = False
) -> Union[List[Tuple[int, int]], Tuple[List[Tuple[int, int]], List[int]]]:
    """Load prediction result file.

    Returns:
        intervals, or (intervals, confidences) when with_confidence=True
    """
    with open(result_file, 'r', encoding='utf-8') as f:
        result_data = json.load(f)

    if result_data.get("error", False):
        return ([], []) if with_confidence else []

    anomaly_intervals = result_data.get("anomaly_intervals", [])
    confidences = result_data.get("confidences", [])
    segment_start_idx = get_segment_start_idx(segment_data_file) if segment_data_file else None

    intervals = []
    interval_confidences = []
    for i, interval in enumerate(anomaly_intervals):
        if not (isinstance(interval, list) and len(interval) >= 2):
            continue
        confidence = confidences[i] if i < len(confidences) else 1

        global_start, global_end = int(interval[0]), int(interval[1])
        appended = False
        if segment_start_idx is not None:
            rel_start = global_start - segment_start_idx
            rel_end = global_end - segment_start_idx
            if rel_start >= 0 and rel_end >= 0:
                intervals.append((rel_start, rel_end))
                appended = True
        else:
            intervals.append((global_start, global_end))
            appended = True

        if with_confidence and appended:
            try:
                interval_confidences.append(int(confidence))
            except (ValueError, TypeError):
                interval_confidences.append(1)

    return (intervals, interval_confidences) if with_confidence else intervals


def intervals_to_point_set(intervals: List[Tuple[int, int]]) -> Set[int]:
    """Convert interval list to point set (inclusive endpoints)."""
    point_set = set()
    for start, end in intervals:
        for point in range(start, end + 1):
            point_set.add(point)
    return point_set


def convert_points_to_intervals(points: Set[int]) -> List[Tuple[int, int]]:
    """Convert point set to interval list (merge consecutive points)."""
    if not points:
        return []
    sorted_points = sorted(points)
    intervals = []
    start = end = sorted_points[0]
    for point in sorted_points[1:]:
        if point == end + 1:
            end = point
        else:
            intervals.append((start, end))
            start = end = point
    intervals.append((start, end))
    return intervals


def calculate_f1_score(y_true: Set[int], y_pred: Set[int]) -> Tuple[float, float, float, Dict]:
    """Compute F1 score. Returns (precision, recall, f1, metrics_dict)."""
    tp = len(y_true & y_pred)
    fp = len(y_pred - y_true)
    fn = len(y_true - y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def calculate_best_f1_by_confidence(
    all_gt_points: Set[int],
    all_pred_intervals_with_confidence: List[Tuple[Tuple[int, int], int]]
) -> Dict:
    """Compute F1 at confidence thresholds 1,2,3 and return best."""
    if not all_gt_points:
        return {"best_confidence": None, "best_f1": 0.0, "best_metrics": None}

    best_f1 = 0.0
    best_confidence = None
    best_metrics = None

    for conf_threshold in [1, 2, 3]:
        filtered = [iv for iv, c in all_pred_intervals_with_confidence if c >= conf_threshold]
        pred_points = intervals_to_point_set(filtered)
        precision, recall, f1, metrics = calculate_f1_score(all_gt_points, pred_points)
        if f1 > best_f1:
            best_f1 = f1
            best_confidence = conf_threshold
            best_metrics = {"precision": precision, "recall": recall, "f1": f1, **metrics}

    if best_confidence is None:
        best_confidence = 1
        filtered = [iv for iv, c in all_pred_intervals_with_confidence if c >= 1]
        _, _, best_f1, best_metrics = calculate_f1_score(all_gt_points, intervals_to_point_set(filtered))

    return {"best_confidence": best_confidence, "best_f1": best_f1, "best_metrics": best_metrics}


def evaluate_sequence(sequence_samples: List[Dict]) -> Optional[Dict]:
    """Evaluate all samples in a sequence."""
    sequence_samples.sort(key=lambda x: x["segment_number"])

    all_gt_points = set()
    all_pred_points = set()
    all_pred_intervals_with_confidence = []
    segment_results = []

    for sample in sequence_samples:
        try:
            with open(sample["result_file"], 'r', encoding='utf-8') as f:
                if json.load(f).get("error", False):
                    print(f"Warning: skipping {sample['sample_name']} (detection error)")
                    continue
        except Exception as e:
            print(f"Warning: failed to read {sample['result_file']}: {e}")
            continue

        result_file_dir = sample["result_file"].parent
        segment_data_file = result_file_dir / "segment_data.csv"
        segment_start_idx = get_segment_start_idx(segment_data_file) or 0

        gt_dict = load_groundtruth(sample["gt_file"])
        gt_anomaly_points = {idx for idx, label in gt_dict.items() if label == 1}
        gt_anomaly_global = {idx + segment_start_idx for idx in gt_anomaly_points}

        pred_intervals, pred_confidences = load_prediction(
            sample["result_file"], segment_data_file=segment_data_file, with_confidence=True
        )

        pred_points = intervals_to_point_set(pred_intervals)
        pred_intervals_global = [(s + segment_start_idx, e + segment_start_idx) for s, e in pred_intervals]
        pred_points_global = intervals_to_point_set(pred_intervals_global)

        all_pred_intervals_with_confidence.extend(
            ((s + segment_start_idx, e + segment_start_idx), c)
            for (s, e), c in zip(pred_intervals, pred_confidences)
        )
        all_gt_points.update(gt_anomaly_global)
        all_pred_points.update(pred_points_global)

        seg_precision, seg_recall, seg_f1, seg_metrics = calculate_f1_score(gt_anomaly_points, pred_points)
        segment_results.append({
            "sample_name": sample["sample_name"],
            "segment_number": sample["segment_number"],
            "metrics": seg_metrics,
        })

    if not all_gt_points:
        seq_id = sequence_samples[0]["sample_name"].split('_')[0] if sequence_samples else "unknown"
        print(f"  Skipping sequence {seq_id}: no anomaly samples")
        return None

    seq_precision, seq_recall, seq_f1, seq_metrics = calculate_f1_score(all_gt_points, all_pred_points)
    best_f1_result = calculate_best_f1_by_confidence(all_gt_points, all_pred_intervals_with_confidence)

    return {
        "sequence_id": sequence_samples[0]["sample_name"].split('_')[0] if sequence_samples else "unknown",
        "num_segments": len(sequence_samples),
        "sequence_metrics": seq_metrics,
        "best_f1_result": best_f1_result,
        "segment_results": segment_results,
        "total_gt_anomalies": len(all_gt_points),
        "total_pred_anomalies": len(all_pred_points),
    }


def evaluate_batch_results(results_dir: str) -> Dict:
    """Evaluate batch processing results."""
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    sequences = defaultdict(list)
    for sample_dir in results_path.iterdir():
        if not sample_dir.is_dir():
            continue
        gt_file = sample_dir / "groundtruth.json"
        result_file = sample_dir / "detection_result.json"
        if not gt_file.exists() or not result_file.exists():
            print(f"Warning: skipping {sample_dir.name} (missing required files)")
            continue
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                if json.load(f).get("error", False):
                    continue
        except Exception:
            continue

        sequence_id = extract_sequence_id(sample_dir.name)
        segment_number = extract_segment_number(sample_dir.name)
        sequences[sequence_id].append({
            "sample_name": sample_dir.name,
            "gt_file": gt_file,
            "result_file": result_file,
            "segment_number": segment_number,
        })

    print(f"Found {len(sequences)} sequences")
    sequence_evaluations = []

    for sequence_id, samples in sequences.items():
        print(f"\nEvaluating sequence: {sequence_id} ({len(samples)} segments)")
        try:
            result = evaluate_sequence(samples)
            if result is not None:
                sequence_evaluations.append(result)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    precisions = [s["sequence_metrics"]["precision"] for s in sequence_evaluations]
    recalls = [s["sequence_metrics"]["recall"] for s in sequence_evaluations]
    f1s = [s["sequence_metrics"]["f1"] for s in sequence_evaluations]
    best_f1s = [s["best_f1_result"]["best_f1"] for s in sequence_evaluations]
    best_confs = [s["best_f1_result"].get("best_confidence") for s in sequence_evaluations]
    best_confs = [c for c in best_confs if c is not None]
    most_common_conf = Counter(best_confs).most_common(1)[0][0] if best_confs else None

    return {
        "total_sequences": len(sequence_evaluations),
        "average_metrics": {
            "precision": sum(precisions) / len(precisions) if precisions else 0.0,
            "recall": sum(recalls) / len(recalls) if recalls else 0.0,
            "f1": sum(f1s) / len(f1s) if f1s else 0.0,
        },
        "best_f1_metrics": {
            "average_best_f1": sum(best_f1s) / len(best_f1s) if best_f1s else 0.0,
            "most_common_best_confidence": most_common_conf,
        },
        "sequence_evaluations": sequence_evaluations,
    }


def print_evaluation_summary(summary: Dict):
    """Print evaluation summary."""
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"\nTotal sequences: {summary['total_sequences']}")

    am = summary["average_metrics"]
    print(f"\nAverage metrics (Precision, Recall, F1):")
    print(f"  Precision: {am['precision']:.4f}")
    print(f"  Recall:    {am['recall']:.4f}")
    print(f"  F1:        {am['f1']:.4f}")

    bf = summary.get("best_f1_metrics", {})
    print(f"\nBest F1:")
    print(f"  Average Best F1: {bf.get('average_best_f1', 0):.4f}")
    if bf.get("most_common_best_confidence") is not None:
        print(f"  Most common best confidence threshold: {bf['most_common_best_confidence']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate batch processing results")
    parser.add_argument("--results_dir", type=str, required=True, help="Results directory")
    parser.add_argument("--output", type=str, default="./evaluation_results.json", help="Output file")
    args = parser.parse_args()

    print("Starting evaluation...")
    summary = evaluate_batch_results(args.results_dir)
    print_evaluation_summary(summary)

    with open(Path(args.output), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {args.output}")
