"""
eval.py — score a pipeline timeline against ground truth.

Compares the classifier's predicted ad intervals to the inserted_ads[]
from the ground-truth JSON. Prints three things:

  1. Per-second metrics (precision / recall / F1) treating "ad" as the
     positive class.
  2. For each ground-truth ad, the best-matching predicted segment and
     their IoU.
  3. False positives — predicted ad segments that don't overlap any
     real ad.

Usage:
    python -m backend.eval test_001
    python -m backend.eval test_001 --gt path/to/gt.json --pred path/to/pred.json

Author: Jesus Ramos
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ==== Default paths =====================================================
# These match the weirdly-named Drive export folders. Override via CLI
# flags if the layout ever changes.

PROJECT_ROOT     = Path(__file__).resolve().parent.parent
GT_DIR_DEFAULT   = PROJECT_ROOT / "Test Data" / "video_info-20260417T165731Z-3-001" / "video_info"
PRED_DIR_DEFAULT = (
    PROJECT_ROOT / "Test Data"
                 / "videos_with_ads-20260417T170700Z-3-001"
                 / "videos_with_ads"
                 / "analysis"
)


# ==== Loaders ===========================================================

def load_ground_truth(path: Path):
    """Pull the total duration and the inserted_ads intervals."""
    data = json.loads(path.read_text(encoding="utf-8"))
    duration = float(data.get("output_duration_seconds", 0.0))
    ads = [
        {
            "start": float(a["final_video_ad_start_seconds"]),
            "end":   float(a["final_video_ad_end_seconds"]),
        }
        for a in data.get("inserted_ads", [])
    ]
    return duration, ads


# Anything in this set is treated as "ad" for the purposes of comparing
# against the ground truth (which only knows about ads). When the
# classifier later distinguishes between sponsor / intro / dead_air etc.,
# the eval still rolls them all up to non-content here.
NON_CONTENT_TYPES = {
    "ad",                # legacy single-label output
    "ad_break",
    "sponsor",
    "self_promo",
    "recap",
    "intro",
    "outro",
    "transition",
    "dead_air",
    "holding_screen",
    "filler",
}


def load_prediction(path: Path):
    """Pull predicted non-content intervals from the pipeline's timeline JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "start": float(s["start_seconds"]),
            "end":   float(s["end_seconds"]),
        }
        for s in data.get("timeline_segments", [])
        if s.get("type") in NON_CONTENT_TYPES
    ]


# ==== Metrics ===========================================================

def intervals_to_mask(intervals, duration, resolution=1.0):
    """
    Rasterize a list of intervals into a boolean list sampled at
    `resolution` seconds. Default 1 Hz is plenty for this problem.
    """
    n = int(duration / resolution) + 1
    mask = [False] * n
    for iv in intervals:
        lo = max(0, int(iv["start"] / resolution))
        hi = min(n - 1, int(iv["end"] / resolution))
        for i in range(lo, hi + 1):
            mask[i] = True
    return mask


def per_second_metrics(gt_mask, pred_mask):
    tp = fp = fn = tn = 0
    for g, p in zip(gt_mask, pred_mask):
        if g and p:
            tp += 1
        elif p and not g:
            fp += 1
        elif g and not p:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )

    return {
        "tp_sec": tp, "fp_sec": fp, "fn_sec": fn, "tn_sec": tn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def iou(a, b):
    """Intersection over union of two [start, end] intervals."""
    lo = max(a["start"], b["start"])
    hi = min(a["end"],   b["end"])
    inter = max(0.0, hi - lo)
    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - inter
    return inter / union if union > 0 else 0.0


def match_gt_to_pred(gt_ads, pred_ads):
    """For each GT ad, find the predicted segment with highest IoU."""
    rows = []
    for g in gt_ads:
        best_iou = 0.0
        best_pred = None
        for p in pred_ads:
            i = iou(g, p)
            if i > best_iou:
                best_iou = i
                best_pred = p
        rows.append({"gt": g, "best_pred": best_pred, "iou": best_iou})
    return rows


def find_false_positives(gt_ads, pred_ads, min_iou=0.1):
    """Predicted ads that don't meaningfully overlap any real ad."""
    fps = []
    for p in pred_ads:
        best = max((iou(p, g) for g in gt_ads), default=0.0)
        if best < min_iou:
            fps.append({"pred": p, "best_iou": best})
    return fps


# ==== Reporting =========================================================

def print_report(gt_path, pred_path):
    duration, gt_ads = load_ground_truth(gt_path)
    pred_ads = load_prediction(pred_path)

    gt_mask   = intervals_to_mask(gt_ads,   duration)
    pred_mask = intervals_to_mask(pred_ads, duration)

    metrics     = per_second_metrics(gt_mask, pred_mask)
    matches     = match_gt_to_pred(gt_ads, pred_ads)
    false_pos   = find_false_positives(gt_ads, pred_ads)
    total_gt_ad_sec   = sum(a["end"] - a["start"] for a in gt_ads)
    total_pred_ad_sec = sum(a["end"] - a["start"] for a in pred_ads)

    print(f"GT:   {gt_path}")
    print(f"Pred: {pred_path}")
    print(f"Video duration: {duration:.1f}s")
    print()
    print(f"Ground-truth ads: {len(gt_ads)}  (total {total_gt_ad_sec:.1f}s)")
    print(f"Predicted ads:    {len(pred_ads)}  (total {total_pred_ad_sec:.1f}s)")
    print()

    print("---- per-second metrics (ads = positive class) ----")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1:        {metrics['f1']:.3f}")
    print(f"  TP: {metrics['tp_sec']}s   FP: {metrics['fp_sec']}s   FN: {metrics['fn_sec']}s")
    print()

    print("---- ground-truth ad -> best matching prediction ----")
    for i, m in enumerate(matches, start=1):
        g = m["gt"]
        p = m["best_pred"]
        dur = g["end"] - g["start"]
        print(f"  Ad {i}: {g['start']:7.1f}s – {g['end']:7.1f}s  ({dur:5.1f}s)")
        if p:
            print(f"     -> pred {p['start']:7.1f}s – {p['end']:7.1f}s   IoU = {m['iou']:.3f}")
        else:
            print("     -> no matching prediction")
    print()

    if false_pos:
        print(f"---- false-positive predictions (IoU < 0.1) ----")
        for fp in false_pos:
            p = fp["pred"]
            print(f"  pred {p['start']:7.1f}s – {p['end']:7.1f}s   (best IoU with any GT ad: {fp['best_iou']:.3f})")
        print()

    # Single-line summary for easy diffing between runs
    print(
        f"SUMMARY  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
        f"F1={metrics['f1']:.3f}  FPs={len(false_pos)}"
    )


# ==== CLI ===============================================================

def _resolve_paths(name: str, gt_override: str | None, pred_override: str | None):
    gt_path = Path(gt_override) if gt_override else GT_DIR_DEFAULT / f"{name}.json"
    pred_path = (
        Path(pred_override) if pred_override
        else PRED_DIR_DEFAULT / f"{name}_timeline.json"
    )
    return gt_path, pred_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline output vs ground truth.")
    parser.add_argument("name", help="Video base name, e.g. 'test_001'")
    parser.add_argument("--gt",   default=None, help="Override path to ground-truth JSON")
    parser.add_argument("--pred", default=None, help="Override path to prediction JSON")
    args = parser.parse_args()

    gt_path, pred_path = _resolve_paths(args.name, args.gt, args.pred)

    if not gt_path.exists():
        print(f"Ground-truth file not found: {gt_path}", file=sys.stderr)
        sys.exit(1)
    if not pred_path.exists():
        print(f"Prediction file not found: {pred_path}", file=sys.stderr)
        print("Run the pipeline first:", file=sys.stderr)
        print(f"  python -m backend.pipeline <video>", file=sys.stderr)
        sys.exit(1)

    print_report(gt_path, pred_path)


if __name__ == "__main__":
    main()
