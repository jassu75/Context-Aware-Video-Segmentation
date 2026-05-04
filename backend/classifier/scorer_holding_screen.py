from __future__ import annotations
from statistics import median
from backend.classifier._shared import at, scenes_from, windows_from
import re

LABEL = "holding_screen"
MAX_SCORE = 10.0

# ==== Per-rule point allocations ========================================

PTS_DEAD_SPACE_FLAG = 1.0
PTS_STILL_FRAMES = 1.0
PTS_SILENCE_DUR = 5.0
PTS_NO_TEXT = 1.0
PTS_HAS_COUNTDOWN = 3.5

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

        # likely to be holding screen if 
        # 1. the audio output sets dead air flag to True
        # 2. there is little to no change in frame sequence
        # 3. silence duration lasts longer than 5 seconds if there is any
        # 4. if there is a countdown, the countdown only consists of numbers
        audio_win = at(audio_windows, i)
        text_win = at(text_windows, i)
        dead_air_flag_val = _dead_air_flag_state(audio_win)
        win_silence_dur = _silence_duration(audio_win, audio_windows, dead_air_flag_val, i)
        has_words = _words_detected(text_win)
        still_scene_stat = _still_scenes(video_windows, i)
        has_countdown = _has_countdown(text_windows, i)

        # add points if this window is part of a longer sequence of silent windows
        if win_silence_dur >= MIN_SILENCE_DURATION:
            s += PTS_SILENCE_DUR
        if dead_air_flag_val:
            s += PTS_DEAD_SPACE_FLAG
        if not has_words:
            s += PTS_NO_TEXT
        if still_scene_stat:
            s += PTS_STILL_FRAMES
        if has_countdown:
            s += PTS_HAS_COUNTDOWN

        s = min(s, MAX_SCORE)

        row[LABEL] = round(s, 2)

        results.append({
            "window_index": row["window_index"],
            "label": LABEL,
            "score": round(s, 2),
        })

    return results

# ==== Helper Functions ===========================================================

def _dead_air_flag_state(audio_window) -> bool:
    if audio_window:
        return audio_window.get("dead_air_flag", False)
    return False

def _silence_duration(audio_window, audio_windows_list, cur_audio_win_has_dead_air, current_audio_win_idx) -> float:
    dead_air_duration = 0.0
    if cur_audio_win_has_dead_air:
        # silence run start field is provided, allowing us to calculate the past/present contribution 
        # to the silence duration
        # with that, we just need to check future windows to see if silence duration continues
        if audio_window and audio_window.get("end_s") and audio_window.get("silence_run_start"):
            dead_air_duration = abs(audio_window.get("end_s") - audio_window.get("silence_run_start"))

            right_idx = current_audio_win_idx + 1

            while right_idx < len(audio_windows_list):
                next_window = at(audio_windows_list, right_idx)
                if (next_window) and (_dead_air_flag_state(next_window) is True) and (next_window.get("end_s") and (next_window.get("silence_run_start"))):
                    dead_air_duration = abs(next_window.get("end_s") - next_window.get("silence_run_start"))
                    right_idx += 1
                else:
                    break

    return dead_air_duration

def _words_detected(text_window) -> bool:
    if not text_window:
        return False
    features = text_window.get("features", {})
    if "word_count" in features:
        return int(features.get("word_count") or 0) > 0
    if "words" in text_window:
        return len(text_window.get("words") or []) > 0
    return bool((text_window.get("transcript") or "").strip())

def _still_scenes(video_windows, cur_win_idx) -> bool:
    cur_win = at(video_windows, cur_win_idx)
    diff = FRAME_DIFF_THRESHOLD + 1.0

    if cur_win and cur_win.get("frame_diff_mean"):
        diff = cur_win.get("frame_diff_mean")
    return diff <= FRAME_DIFF_THRESHOLD

def _has_countdown(text_windows, cur_win_idx) -> bool:
    stat = False
    text_window = at(text_windows, cur_win_idx)
    if text_window and text_window.get("transcript"):
        filtered_transcript = "".join(char for char in text_window.get("transcript") if char.isalnum())
        if re.fullmatch(r"^\d+$", filtered_transcript):
            stat = True
    return stat
