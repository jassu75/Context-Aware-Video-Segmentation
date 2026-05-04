"""
Evaluate classifier timelines against manual self_promo/recap labels.

This reads Test Data/manual_non_content_labels/self_promo_recap.json instead
of the inserted-ad JSONs. Use this while tuning labels that do not have
assignment-provided ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.eval import (
    PRED_DIR_DEFAULT,
    find_false_positives,
    intervals_to_mask,
    match_gt_to_pred,
    per_second_metrics,
)
from backend.eval_category import load_prediction_for_labels


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANUAL_LABELS_DEFAULT = PROJECT_ROOT / "Test Data" / "manual_non_content_labels" / "self_promo_recap.json"


def load_manual_label_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manual_segments(data: dict, name: str, labels: set[str]) -> list[dict]:
    video = data.get("videos", {}).get(name, {})
    return [
        {
            "start": float(seg["start_seconds"]),
            "end": float(seg["end_seconds"]),
            "type": seg.get("type"),
            "notes": seg.get("notes", ""),
            "segment_id": seg.get("segment_id"),
        }
        for seg in video.get("segments", [])
        if seg.get("type") in labels
    ]


def video_duration_from_segments(gt_segments: list[dict], pred_segments: list[dict]) -> float:
    all_segments = gt_segments + pred_segments
    if not all_segments:
        return 0.0
    return max(float(seg["end"]) for seg in all_segments) + 1.0


def print_video_report(name: str, labels: set[str], manual_data: dict, pred_path: Path) -> dict:
    gt_segments = load_manual_segments(manual_data, name, labels)
    pred_segments = load_prediction_for_labels(pred_path, labels)
    duration = video_duration_from_segments(gt_segments, pred_segments)

    gt_mask = intervals_to_mask(gt_segments, duration)
    pred_mask = intervals_to_mask(pred_segments, duration)
    metrics = per_second_metrics(gt_mask, pred_mask)
    matches = match_gt_to_pred(gt_segments, pred_segments)
    false_pos = find_false_positives(gt_segments, pred_segments)

    total_gt_sec = sum(seg["end"] - seg["start"] for seg in gt_segments)
    total_pred_sec = sum(seg["end"] - seg["start"] for seg in pred_segments)

    print(f"Video: {name}")
    print(f"Labels evaluated: {', '.join(sorted(labels))}")
    print(f"GT segments:   {len(gt_segments)}  ({total_gt_sec:.1f}s)")
    print(f"Pred segments: {len(pred_segments)}  ({total_pred_sec:.1f}s)")
    print(f"Precision: {metrics['precision']:.3f}  Recall: {metrics['recall']:.3f}  F1: {metrics['f1']:.3f}")
    print(f"TP: {metrics['tp_sec']}s   FP: {metrics['fp_sec']}s   FN: {metrics['fn_sec']}s")

    print("GT -> best prediction:")
    for match in matches:
        gt = match["gt"]
        pred = match["best_pred"]
        note = f"  ({gt.get('notes')})" if gt.get("notes") else ""
        print(f"  {gt['type']} {gt['start']:7.1f}s - {gt['end']:7.1f}s{note}")
        if pred:
            print(f"    -> {pred['type']} {pred['start']:7.1f}s - {pred['end']:7.1f}s  IoU={match['iou']:.3f}")
        else:
            print("    -> no matching prediction")

    print(f"False positives (IoU < 0.1): {len(false_pos)}")
    print(
        f"SUMMARY video={name} labels={','.join(sorted(labels))} "
        f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
        f"F1={metrics['f1']:.3f} FPs={len(false_pos)}"
    )
    print()

    return {
        "gt_segments": gt_segments,
        "pred_segments": pred_segments,
        "metrics": metrics,
        "false_pos": false_pos,
    }


def print_aggregate(results: list[dict], labels: set[str]) -> None:
    gt_segments = []
    pred_segments = []
    offset = 0.0
    for result in results:
        gt_segments.extend(_offset_segments(result["gt_segments"], offset))
        pred_segments.extend(_offset_segments(result["pred_segments"], offset))
        max_end = max(
            [seg["end"] for seg in result["gt_segments"] + result["pred_segments"]],
            default=0.0,
        )
        offset += max_end + 10.0

    duration = max([seg["end"] for seg in gt_segments + pred_segments], default=0.0) + 1.0
    metrics = per_second_metrics(
        intervals_to_mask(gt_segments, duration),
        intervals_to_mask(pred_segments, duration),
    )
    false_pos = find_false_positives(gt_segments, pred_segments)
    print("==== Aggregate ====")
    print(f"Labels evaluated: {', '.join(sorted(labels))}")
    print(f"GT segments:   {len(gt_segments)}")
    print(f"Pred segments: {len(pred_segments)}")
    print(f"Precision: {metrics['precision']:.3f}  Recall: {metrics['recall']:.3f}  F1: {metrics['f1']:.3f}")
    print(f"TP: {metrics['tp_sec']}s   FP: {metrics['fp_sec']}s   FN: {metrics['fn_sec']}s")
    print(f"False positives (IoU < 0.1): {len(false_pos)}")


def _offset_segments(segments: list[dict], offset: float) -> list[dict]:
    out = []
    for seg in segments:
        row = dict(seg)
        row["start"] = float(row["start"]) + offset
        row["end"] = float(row["end"]) + offset
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions against manual self_promo/recap labels.")
    parser.add_argument("names", nargs="*", default=["test_001", "test_002", "test_003", "test_004", "test_005"])
    parser.add_argument("--labels", default="self_promo,recap", help="Comma-separated labels to evaluate")
    parser.add_argument("--manual", default=str(MANUAL_LABELS_DEFAULT), help="Manual label JSON path")
    parser.add_argument("--pred-dir", default=str(PRED_DIR_DEFAULT), help="Directory containing *_timeline.json files")
    args = parser.parse_args()

    labels = {label.strip() for label in args.labels.split(",") if label.strip()}
    manual_data = load_manual_label_file(Path(args.manual))
    pred_dir = Path(args.pred_dir)

    results = []
    for name in args.names:
        results.append(print_video_report(name, labels, manual_data, pred_dir / f"{name}_timeline.json"))
    print_aggregate(results, labels)


if __name__ == "__main__":
    main()
