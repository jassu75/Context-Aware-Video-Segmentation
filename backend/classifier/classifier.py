"""
classifier.py — orchestrate per-category non-content scorers.

Each known non-content segment type lives in its own scorer module
(scorer_sponsor.py, scorer_intro.py, etc). This file just wires them
together: it builds a shared per-window score array, hands it to every
registered scorer, then picks a winning label per window and merges
adjacent same-label windows into segments.

Score scale per category: scorers may emit raw rule points; the classifier
normalizes each category onto a shared 0-10 scale before comparing labels.
A window stays "video_content" unless some category's score >= MIN_LABEL_SCORE.

Categories and ownership (one scorer module per bullet):
  - sponsor          (Jesus)  scorer_sponsor.py
  - self_promo       (Jesus)  scorer_self_promo.py
  - recap            (Jesus)  scorer_recap.py
  - intro            (Tejas)  scorer_intro.py
  - outro            (Tejas)  scorer_outro.py
  - transition       (Tejas)  scorer_transition.py
  - dead_air         (Michael) scorer_dead_air.py
  - holding_screen   (Michael) scorer_holding_screen.py
  - filler           (Michael) scorer_filler.py

Scorer module contract — every scorer_*.py must export:
  LABEL: str
      The category name (matches the file's suffix).
  score(audio_data, text_data, scene_data, video_data, scores) -> list[dict]
      Mutates the shared `scores` array in place by writing scores[i][LABEL].
      Returns a list of {"window_index", "label", "score"} entries.

Author: Jesus Ramos (orchestrator + sponsor / self_promo / recap scorers)
"""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path


# ==== Tunable knobs =====================================================

# A window is labeled non-content only if some category clears this bar.
MIN_LABEL_SCORE = 4.0

# Fallback denominator for scorer normalization. Scorers can override this by
# exporting MAX_POINTS = their total possible raw rule points.
DEFAULT_MAX_POINTS = 10.0

# After labeling, fill short content gaps surrounded by the same non-content
# label. Per-window scoring tends to fragment a real ad block into many
# tiny segments because some windows land just below threshold.
SMOOTHING_MAX_GAP_WINDOWS = 2

# Default minimum duration (fallback)
DEFAULT_MIN_DURATION_SEC = 3.0

# Per-label minimum durations
MIN_DURATION_BY_LABEL = {
    "intro": 6.0,
    "outro": 3.0,
    "self_promo": 2.0,
    "recap": 8.0,
}

CATEGORY_PRIORITY = [
    "ad_break",
    "sponsor",
    "intro",
    "outro",
    "self_promo",
    "recap",
    "dead_air",
    "holding_screen",
    "transition",
    "filler",
]

# Positional constraints
POSITION_CONSTRAINTS = {
    "intro": "start_only",
    "outro": "end_only",
}


# ==== Scorer registration ===============================================
# Try to import each expected scorer module. Missing files don't crash
# the pipeline — they just print a heads-up and get skipped. That way
# Jesus / Michael / Tejas can land their files independently.

_EXPECTED_SCORERS = [
    "scorer_ad_break",
    "scorer_sponsor",
    "scorer_self_promo",
    "scorer_recap",
    "scorer_intro",
    "scorer_outro",
    "scorer_transition",
    "scorer_dead_air",
    "scorer_holding_screen",
    "scorer_filler",
]

ENABLED_SCORERS = []
for _name in _EXPECTED_SCORERS:
    try:
        ENABLED_SCORERS.append(import_module(f"backend.classifier.{_name}"))
    except ImportError as _e:
        print(f"[classifier] {_name} not available - skipping ({_e})")


# ==== Public API ========================================================

def classify(audio_data, text_data, scene_data, video_data=None, debug: bool = False) -> dict:
    """
    Run every registered scorer, label each window, and merge into segments.

    Returns a dict with:
      - "windows":            per-window scores + winning label
      - "timeline_segments":  merged segments for the player / eval
      - "per_category_scores": each scorer's own returned list (debugging)
      - "enabled_categories": which scorer modules were actually loaded
    """
    scores = _init_scores(audio_data, text_data, video_data)

    # Each scorer mutates `scores` in place AND returns its own list.
    per_category_results = {}
    window_debug = {i: {} for i in range(len(scores))}

    for mod in ENABLED_SCORERS:
        results = mod.score(audio_data, text_data, scene_data, video_data, scores, debug=debug)

        per_category_results[mod.LABEL] = results

        if debug:
            for r in results:
                i = r["window_index"]
                window_debug[i][mod.LABEL] = r.get("debug")

    scores = _normalize_scores(scores)
    labeled_windows = _label_windows(scores, window_debug if debug else None)
    labeled_windows = _smooth_labels(labeled_windows, SMOOTHING_MAX_GAP_WINDOWS)
    segments        = _merge_to_segments(labeled_windows)
    segments        = _enforce_position_constraints(segments)
    segments        = _drop_short_non_content(segments, DEFAULT_MIN_DURATION_SEC)

    return {
        "windows":             labeled_windows,
        "timeline_segments":   segments,
        "per_category_scores": per_category_results,
        "enabled_categories":  [m.LABEL for m in ENABLED_SCORERS],
    }


def classify_from_files(audio_path, text_path, scene_path, video_path=None, debug: bool = False) -> dict:
    """Same as classify() but loads JSONs from disk."""
    audio = _load_json(audio_path)
    text  = _load_json(text_path)
    scene = _load_json(scene_path)
    video = _load_json(video_path) if video_path else None
    return classify(audio, text, scene, video, debug=debug)


# ==== Score array setup =================================================

def _init_scores(audio_data, text_data, video_data) -> list[dict]:
    """
    Build the shared per-window score array. Audio is the canonical
    window list; text/video should match. Each row gets a 0.0 slot
    for every enabled category.
    """
    canonical = []
    for src in (audio_data, text_data, video_data):
        if src and src.get("windows"):
            canonical = src["windows"]
            break

    scores = []
    for w in canonical:
        row = {
            "window_index": w.get("window_index"),
            "start_s":      w.get("start_s"),
            "end_s":        w.get("end_s"),
        }
        for mod in ENABLED_SCORERS:
            row[mod.LABEL] = 0.0
        scores.append(row)

    return scores


def _normalization_scales() -> dict[str, float]:
    scales = {}
    for mod in ENABLED_SCORERS:
        max_points = float(getattr(mod, "MAX_POINTS", DEFAULT_MAX_POINTS) or DEFAULT_MAX_POINTS)
        scales[mod.LABEL] = max_points
    return scales


def _normalize_scores(scores) -> list[dict]:
    """
    Convert each scorer's raw rule points to the common 0-10 comparison scale.

    This lets scorer owners choose natural rule weights internally while the
    classifier compares labels consistently. We intentionally do not cap here:
    scorer MAX_POINTS should equal the total points available if all rules pass.
    """
    scales = _normalization_scales()
    normalized = []
    for row in scores:
        out = dict(row)
        raw_scores = {}
        for label, max_points in scales.items():
            raw = float(row.get(label, 0.0) or 0.0)
            raw_scores[label] = raw
            out[label] = (raw / max_points) * 10.0 if max_points > 0 else raw
        out["raw_scores"] = raw_scores
        normalized.append(out)
    return normalized


# ==== Labeling ==========================================================

def _label_windows(scores, window_debug=None) -> list[dict]:
    """
    Pick the winning category per window using normalized scores. Highest
    score wins; ties go to whichever category is earlier in CATEGORY_PRIORITY.
    If the winner doesn't clear MIN_LABEL_SCORE, the window is content.
    """
    priority_index = {label: i for i, label in enumerate(CATEGORY_PRIORITY)}
    category_labels = [m.LABEL for m in ENABLED_SCORERS]

    labeled = []
    for row in scores:
        # Build (label, score) pairs and sort by score desc, then priority.
        pairs = [(label, row.get(label, 0.0)) for label in category_labels]
        pairs.sort(key=lambda p: (-p[1], priority_index.get(p[0], 999)))

        if pairs and pairs[0][1] >= MIN_LABEL_SCORE:
            label, score = pairs[0]
        else:
            label, score = "video_content", 0.0

        labeled.append({
            **row,
            "label":     label,
            "max_score": round(score, 2),
            "debug":     window_debug.get(row["window_index"]) if window_debug else None,
        })

    return labeled


# ==== Smoothing =========================================================

def _smooth_labels(labeled_windows, max_gap_windows):
    """
    Fill short content gaps between same-label non-content windows.

    Per-window scoring fragments a real ad/sponsor block when one or two
    windows in the middle land just below threshold. This walks the list,
    finds short content gaps that have the same non-content label on both
    sides, and flips the gap to that label.
    """
    if not labeled_windows or max_gap_windows < 1:
        return labeled_windows

    out = [dict(w) for w in labeled_windows]
    n = len(out)

    i = 0
    while i < n:
        if out[i]["label"] != "video_content":
            i += 1
            continue

        # Walk through this run of content
        j = i
        while j < n and out[j]["label"] == "video_content":
            j += 1
        gap_len = j - i

        # Fill if short, bounded on both sides, and bounds match
        if 0 < gap_len <= max_gap_windows and i > 0 and j < n:
            left  = out[i - 1]["label"]
            right = out[j]["label"]
            if left == right and left != "video_content":
                for k in range(i, j):
                    out[k]["label"] = left
        i = j

    return out


def _drop_short_non_content(segments, default_min_duration_sec):
    cleaned = []

    for seg in segments:
        if seg["type"] == "video_content":
            cleaned.append(seg)
            continue

        min_required = MIN_DURATION_BY_LABEL.get(
            seg["type"],
            default_min_duration_sec
        )

        if seg["duration_seconds"] < min_required:
            new_seg = dict(seg)
            new_seg["type"] = "video_content"
            new_seg["max_score"] = 0.0
            cleaned.append(new_seg)
        else:
            cleaned.append(seg)

    return _remerge_content(cleaned)


# ==== Position constraints ==============================================

def _enforce_position_constraints(segments) -> list[dict]:
    if not segments:
        return segments

    out = [dict(s) for s in segments]
    last_idx = len(out) - 1

    for i, seg in enumerate(out):
        label = seg["type"]
        rule = POSITION_CONSTRAINTS.get(label)

        if not rule:
            continue

        if rule == "start_only" and i != 0:
            seg["type"] = "video_content"
            seg["max_score"] = 0.0

        elif rule == "end_only" and i != last_idx:
            seg["type"] = "video_content"
            seg["max_score"] = 0.0

    return _remerge_content(out)


def _remerge_content(segments) -> list[dict]:
    if not segments:
        return segments

    merged = [dict(segments[0])]
    for seg in segments[1:]:
        last = merged[-1]
        if seg["type"] == last["type"] == "video_content":
            last["end_seconds"]      = seg["end_seconds"]
            last["duration_seconds"] = round(last["end_seconds"] - last["start_seconds"], 3)
        else:
            merged.append(dict(seg))

    return merged


# ==== Segment merging ===================================================

def _merge_to_segments(labeled_windows) -> list[dict]:
    if not labeled_windows:
        return []

    segments = []
    current = _start_segment(labeled_windows[0], _window_end(labeled_windows, 0))

    for idx, w in enumerate(labeled_windows[1:], start=1):
        window_end = _window_end(labeled_windows, idx)
        if w["label"] == current["type"]:
            current["end_seconds"] = window_end
            current["max_score"]   = max(current["max_score"], w["max_score"])
        else:
            segments.append(_finalize_segment(current))
            current = _start_segment(w, window_end)

    segments.append(_finalize_segment(current))
    return segments


def _window_end(windows, idx):
    """
    Convert overlapping analysis windows into a non-overlapping timeline.

    Audio/text/video use 2s windows with 1s hops. For playback metadata, each
    label should own the interval from this window's start to the next window's
    start, with the final window carrying the true tail end.
    """
    if idx + 1 < len(windows):
        next_start = windows[idx + 1].get("start_s")
        if next_start is not None and next_start > windows[idx].get("start_s", 0.0):
            return next_start
    return windows[idx].get("end_s")


def _start_segment(window, window_end) -> dict:
    return {
        "type":          window["label"],
        "start_seconds": window["start_s"],
        "end_seconds":   window_end,
        "max_score":     window["max_score"],
    }


def _finalize_segment(seg) -> dict:
    seg["duration_seconds"] = round(seg["end_seconds"] - seg["start_seconds"], 3)
    return seg


# ==== IO helpers ========================================================

def _load_json(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==== Quick manual test =================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = "test_001"

    base = Path(__file__).resolve().parent.parent.parent / "Test Data" \
           / "videos_with_ads-20260417T170700Z-3-001" / "videos_with_ads" / "analysis"

    audio_path = base / f"{name}_audio.json"
    text_path  = base / f"{name}_text.json"
    scene_path = base / f"{name}_scenes.json"
    video_path = base / f"{name}_video.json"   # may not exist yet

    result = classify_from_files(
        audio_path, text_path, scene_path,
        video_path if video_path.exists() else None,
    )

    print(f"Enabled categories: {result['enabled_categories']}")
    print(f"Total windows:      {len(result['windows'])}")
    print(f"Total segments:     {len(result['timeline_segments'])}\n")

    for seg in result["timeline_segments"]:
        marker = "" if seg["type"] == "video_content" else f"  score={seg['max_score']}"
        print(f"  {seg['type']:18s}  "
              f"{seg['start_seconds']:7.1f}s - {seg['end_seconds']:7.1f}s  "
              f"({seg['duration_seconds']:.1f}s){marker}")

    # Write timeline JSON so the player / eval can consume it.
    out_path = base / f"{name}_timeline.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result["timeline_segments"], f, indent=2)
    print(f"\nTimeline written to: {out_path}")
