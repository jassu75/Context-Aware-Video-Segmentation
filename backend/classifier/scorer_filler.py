from __future__ import annotations
from statistics import median
from backend.classifier._shared import at, scenes_from, windows_from

LABEL = "filler"
MAX_SCORE = 10.0

# ==== Per-rule point allocations ========================================

PTS_HAS_FILLER_WORDS = 3.0
PTS_HAS_STILL_SCENES = 3.0

# ==== Thresholds ========================================
MIN_SILENCE_DURATION = 5
FRAME_DIFF_THRESHOLD = 1.0

# ==== Scoring ===========================================================

def score(audio_data, text_data, scene_data, video_data, scores, debug=False):
    audio_windows = windows_from(audio_data)
    text_windows = windows_from(text_data)
    video_windows = windows_from(video_data)
    scenes = scenes_from(scene_data)

    results = []

    for i, row in enumerate(scores):
        s = 0.0

        # likely to be filler if 
        # 1. there is filler text in this window
        text_win = at(text_windows, i)
        has_filler_words = _filler_words_detected(text_win)
        long_still_scene = _still_scenes(video_windows, i)

        # add points if this window is part of a longer sequence of silent windows
        if has_filler_words:
            s += PTS_HAS_FILLER_WORDS
        if long_still_scene:
            s += PTS_HAS_STILL_SCENES

        s = min(s, MAX_SCORE)

        row[LABEL] = round(s, 2)

        results.append({
            "window_index": row["window_index"],
            "label": LABEL,
            "score": round(s, 2),
        })

    return results

# ==== Helper Functions ===========================================================

def _filler_words_detected(text_window) -> bool:
    hasFillerWords = True
    if text_window and text_window.get("features") and text_window.get("features").get("word_count") and text_window.get("features").get("has_recap_filler") == True:
        hasWords = True
    return hasWords

def _still_scenes(video_windows, cur_win_idx) -> bool:
    cur_win = at(video_windows, cur_win_idx)
    diff = FRAME_DIFF_THRESHOLD + 1.0

    if cur_win and cur_win.get("frame_diff_mean"):
        diff = cur_win.get("frame_diff_mean")
    return diff <= FRAME_DIFF_THRESHOLD

