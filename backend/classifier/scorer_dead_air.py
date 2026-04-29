from __future__ import annotations
from statistics import median
from backend.classifier._shared import at, scenes_from, windows_from

LABEL = "dead_air"
MAX_SCORE = 10.0

# ==== Per-rule point allocations ========================================

PTS_DEAD_SPACE_FLAG = 2.5
PTS_STILL_FRAMES = 2.5
PTS_SILENCE_DUR = 5.0
PTS_NO_TEXT = 1.5

# ==== Thresholds ========================================
MIN_SILENCE_DURATION = 5

# ==== Scoring ===========================================================

def score(audio_data, text_data, scene_data, video_data, scores):
    audio_windows = windows_from(audio_data)
    text_windows = windows_from(text_data)
    video_windows = windows_from(video_data)
    scenes = scenes_from(scene_data)

    results = []

    for i, row in enumerate(scores):
        s = 0.0

        # likely to be dead air/silence/inactivity if 
        # 1. the audio output sets dead air flag to True
        # 2. there is little to no change in frame sequence
        # 3. silence duration lasts longer than 5 seconds
        # 4. no text is detected for a particular window
        audio_win = at(audio_windows, i)
        text_win = at(text_windows, i)
        dead_air_flag_val = _dead_air_flag_state(audio_win)
        win_silence_dur = _silence_duration(audio_win, audio_windows, dead_air_flag_val, i)
        has_words = _words_detected(text_win)
        still_scene_stat = _still_scenes()

        # add points if this window is part of a longer sequence of silent windows
        if win_silence_dur >= MIN_SILENCE_DURATION:
            s += PTS_SILENCE_DUR
        if dead_air_flag_val:
            s += PTS_DEAD_SPACE_FLAG
        if has_words:
            s += PTS_NO_TEXT
        if still_scene_stat:
            s += PTS_STILL_FRAMES

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
    hasWords = True
    if text_window and text_window.get("transcript") and text_window.get("transcript") == "":
        hasWords = False
    if text_window and text_window.get("words") and len(text_window.get("words")) == 0:
        hasWords = False
    if text_window and text_window.get("features") and text_window.get("features").get("word_count") and text_window.get("features").get("word_count") == 0:
        hasWords = False
    return hasWords

def _still_scenes() -> bool:
    return False

