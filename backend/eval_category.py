"""
Evaluate selected predicted labels against ground-truth ad intervals.

This is useful while developing one scorer at a time. The original eval.py
rolls every non-content label into the positive class; this script lets us
ask narrower questions like "how did only sponsor predictions do?"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.eval import (
    GT_DIR_DEFAULT,
    PRED_DIR_DEFAULT,
    find_false_positives,
    intervals_to_mask,
    load_ground_truth,
    match_gt_to_pred,
    per_second_metrics,
)


def load_prediction_for_labels(path: Path, labels: set[str]):
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = data if isinstance(data, list) else data.get("timeline_segments", [])
    return [
        {
            "start": float(s["start_seconds"]),
            "end": float(s["end_seconds"]),
            "type": s.get("type"),
        }
        for s in segments
        if s.get("type") in labels
    ]


def print_report(name: str, labels: set[str], gt_path: Path, pred_path: Path) -> None:
    duration, gt_ads = load_ground_truth(gt_path)
    pred_segments = load_prediction_for_labels(pred_path, labels)

    gt_mask = intervals_to_mask(gt_ads, duration)
    pred_mask = intervals_to_mask(pred_segments, duration)

    metrics = per_second_metrics(gt_mask, pred_mask)
    matches = match_gt_to_pred(gt_ads, pred_segments)
    false_pos = find_false_positives(gt_ads, pred_segments)

    total_gt_sec = sum(a["end"] - a["start"] for a in gt_ads)
    total_pred_sec = sum(a["end"] - a["start"] for a in pred_segments)

    print(f"Video: {name}")
    print(f"Labels evaluated: {', '.join(sorted(labels))}")
    print(f"GT ads:        {len(gt_ads)}  ({total_gt_sec:.1f}s)")
    print(f"Pred segments: {len(pred_segments)}  ({total_pred_sec:.1f}s)")
    print()

    print("---- per-second metrics ----")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1:        {metrics['f1']:.3f}")
    print(f"  TP: {metrics['tp_sec']}s   FP: {metrics['fp_sec']}s   FN: {metrics['fn_sec']}s")
    print()

    print("---- ground-truth ad -> best matching prediction ----")
    for i, m in enumerate(matches, start=1):
        g = m["gt"]
        p = m["best_pred"]
        print(f"  Ad {i}: {g['start']:7.1f}s - {g['end']:7.1f}s")
        if p:
            print(f"     -> {p['type']} {p['start']:7.1f}s - {p['end']:7.1f}s  IoU={m['iou']:.3f}")
        else:
            print("     -> no matching prediction")
    print()

    print(f"False positives (IoU < 0.1): {len(false_pos)}")
    print(
        f"SUMMARY labels={','.join(sorted(labels))} "
        f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
        f"F1={metrics['f1']:.3f} FPs={len(false_pos)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate selected prediction labels against ad ground truth.")
    parser.add_argument("name", help="Video base name, e.g. test_003")
    parser.add_argument(
        "--labels",
        default="sponsor",
        help="Comma-separated predicted labels to evaluate. Default: sponsor",
    )
    parser.add_argument("--gt", default=None, help="Override path to ground-truth JSON")
    parser.add_argument("--pred", default=None, help="Override path to prediction JSON")
    args = parser.parse_args()

    labels = {label.strip() for label in args.labels.split(",") if label.strip()}
    gt_path = Path(args.gt) if args.gt else GT_DIR_DEFAULT / f"{args.name}.json"
    pred_path = Path(args.pred) if args.pred else PRED_DIR_DEFAULT / f"{args.name}_timeline.json"

    print_report(args.name, labels, gt_path, pred_path)


if __name__ == "__main__":
    main()
