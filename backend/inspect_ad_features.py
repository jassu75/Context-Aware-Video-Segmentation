"""
Print feature summaries around ground-truth ad intervals.

This is a development utility for tuning scorer_ad_break.py. It compares
each known inserted ad against the surrounding content using the JSON files
created by the audio/text/scene/video processors.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR_DEFAULT = (
    PROJECT_ROOT / "Test Data"
    / "videos_with_ads-20260417T170700Z-3-001"
    / "videos_with_ads"
    / "analysis"
)
GT_DIR_DEFAULT = (
    PROJECT_ROOT / "Test Data"
    / "video_info-20260417T165731Z-3-001"
    / "video_info"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect multimodal features around known ads.")
    parser.add_argument("names", nargs="*", default=["test_001", "test_002", "test_003", "test_004", "test_005"])
    parser.add_argument("--analysis-dir", default=str(ANALYSIS_DIR_DEFAULT))
    parser.add_argument("--gt-dir", default=str(GT_DIR_DEFAULT))
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    gt_dir = Path(args.gt_dir)

    for name in args.names:
        inspect_video(name, analysis_dir, gt_dir)


def inspect_video(name: str, analysis_dir: Path, gt_dir: Path) -> None:
    audio = _load(analysis_dir / f"{name}_audio.json")
    text = _load(analysis_dir / f"{name}_text.json")
    scene = _load(analysis_dir / f"{name}_scenes.json")
    video = _load(analysis_dir / f"{name}_video.json")
    gt = _load(gt_dir / f"{name}.json")

    ads = [
        (
            float(ad["final_video_ad_start_seconds"]),
            float(ad["final_video_ad_end_seconds"]),
        )
        for ad in gt.get("inserted_ads", [])
    ]

    print(f"\n===== {name} =====")
    print("idx start   end     dur   scene_iou scene_dur words rms   rmsR  flat  br    brR  sat   fdMax edge  static black histB")
    for idx, (start, end) in enumerate(ads, start=1):
        row = _interval_summary(audio, text, scene, video, start, end)
        print(
            f"{idx:>3} "
            f"{start:>7.1f} {end:>7.1f} {end-start:>6.1f} "
            f"{row['scene_iou']:>8.3f} {row['scene_dur']:>9.1f} "
            f"{row['words']:>5.2f} {row['rms']:>5.3f} {row['rms_ratio']:>5.2f} "
            f"{row['flat']:>5.3f} {row['brightness']:>5.1f} {row['brightness_ratio']:>4.2f} "
            f"{row['saturation']:>5.1f} {row['frame_diff_max']:>5.1f} "
            f"{row['edge']:>5.3f} {row['static']:>6.2f} {row['black']:>5.2f} "
            f"{row['hist_boundary']:>5.2f}"
        )


def _interval_summary(audio, text, scene, video, start: float, end: float) -> dict:
    audio_w = audio.get("windows", [])
    text_w = text.get("windows", [])
    video_w = video.get("windows", [])

    rms_base = _median([w.get("features", {}).get("rms", 0.0) for w in audio_w])
    brightness_base = _median([w.get("mean_brightness_mean", 0.0) for w in video_w])

    best_scene = _best_scene(scene.get("scenes", []), start, end)
    scene_iou = best_scene["iou"] if best_scene else 0.0
    scene_dur = best_scene["end"] - best_scene["start"] if best_scene else 0.0

    rms = _mean(audio_w, start, end, lambda w: w.get("features", {}).get("rms", 0.0))
    brightness = _mean(video_w, start, end, lambda w: w.get("mean_brightness_mean", 0.0))

    return {
        "scene_iou": scene_iou,
        "scene_dur": scene_dur,
        "words": _mean(text_w, start, end, lambda w: w.get("features", {}).get("word_count", 0)),
        "rms": rms,
        "rms_ratio": rms / rms_base if rms_base else 0.0,
        "flat": _mean(audio_w, start, end, lambda w: w.get("features", {}).get("spectral_flatness_mean", 0.0)),
        "brightness": brightness,
        "brightness_ratio": brightness / brightness_base if brightness_base else 0.0,
        "saturation": _mean(video_w, start, end, lambda w: w.get("mean_hsv_s_mean", 0.0)),
        "frame_diff_max": _mean(video_w, start, end, lambda w: w.get("frame_diff_max", 0.0)),
        "edge": _mean(video_w, start, end, lambda w: w.get("edge_density_mean", 0.0)),
        "static": _mean(video_w, start, end, lambda w: w.get("static_frame_ratio", 0.0)),
        "black": _mean(video_w, start, end, lambda w: w.get("black_frame_ratio", 0.0)),
        "hist_boundary": _hist_boundary(video_w, start, end),
    }


def _best_scene(scenes, start: float, end: float):
    best = None
    best_iou = 0.0
    for scene in scenes:
        s = float(scene.get("start_s", 0.0) or 0.0)
        e = float(scene.get("end_s", s) or s)
        iou = _iou(start, end, s, e)
        if iou > best_iou:
            best_iou = iou
            best = {"start": s, "end": e, "iou": iou}
    return best


def _hist_boundary(video_windows, start: float, end: float) -> float:
    if not video_windows:
        return 0.0
    starts = [float(w.get("start_s", 0.0) or 0.0) for w in video_windows]
    start_idx = min(range(len(starts)), key=lambda i: abs(starts[i] - start))
    end_idx = min(range(len(starts)), key=lambda i: abs(starts[i] - end))

    vals = []
    for idx in (start_idx, end_idx):
        if 0 < idx < len(video_windows):
            vals.append(_hist_dist(video_windows[idx - 1], video_windows[idx]))
        if idx + 1 < len(video_windows):
            vals.append(_hist_dist(video_windows[idx], video_windows[idx + 1]))
    return max(vals) if vals else 0.0


def _hist_dist(a, b) -> float:
    ah = a.get("color_hist_mean", [])
    bh = b.get("color_hist_mean", [])
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(ah, bh)))


def _mean(windows, start: float, end: float, getter) -> float:
    vals = [
        getter(w)
        for w in windows
        if _overlap(float(w.get("start_s", 0.0) or 0.0), float(w.get("end_s", 0.0) or 0.0), start, end) > 0
    ]
    return sum(vals) / len(vals) if vals else 0.0


def _median(vals) -> float:
    vals = sorted(v for v in vals if v > 0)
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def _iou(a_start, a_end, b_start, b_end) -> float:
    inter = _overlap(a_start, a_end, b_start, b_end)
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


def _overlap(a_start, a_end, b_start, b_end) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _load(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()
