from __future__ import annotations

import re

from backend.classifier._shared import (
    at,
    rms_baseline,
    windows_from,
    DebugAccumulator,
)

LABEL = "outro"
MAX_POINTS = 10.0
MAX_SCORE = MAX_POINTS


# ---- OUTRO PHRASES -----------------------------------------------------

OUTRO_PATTERNS = [
    r"\bthanks\s+for\s+watching\b",
    r"\bthank\s+you\s+for\s+watching\b",
    r"\bthat'?s\s+(?:it|all)\s+for\s+(?:today|this\s+video)\b",
    r"\bsee\s+you\s+(?:in\s+the\s+next|next\s+time)\b",
    r"\bcatch\s+you\s+in\s+the\s+next\b",
    r"\bi'?ll\s+see\s+you\s+(?:guys|all)?\s*(?:next\s+time|later)\b",
    r"\bhope\s+you\s+(?:enjoyed|liked)\b",
    r"\bdon'?t\s+forget\s+to\s+(?:like|subscribe|comment)\b",
    r"\bmake\s+sure\s+to\s+(?:like|subscribe|comment)\b",
    r"\bsubscribe\s+for\s+more\b",
    r"\bfollow\s+for\s+more\b",
    r"\bstay\s+tuned\s+for\s+more\b",
    r"\bcheck\s+out\s+(?:my|our)\b",
    r"\blink\s+in\s+the\s+description\b",
]


# ---- CONTEXT -----------------------------------------------------------

SELF_REF_PATTERN = r"\b(i|i'm|i'll|i've|we|we're|we'll|my|our)\b"

WRAPUP_PATTERNS = [
    r"\bthat'?s\s+(?:it|all)\b",
    r"\bin\s+summary\b",
    r"\bto\s+wrap\s+up\b",
    r"\bfinal\s+thoughts?\b",
    r"\boverall\b",
    r"\bhope\s+you\s+(?:enjoyed|liked)\b",
]


# ---- POINTS ------------------------------------------------------------

PTS_OUTRO_KEYWORD     = 4.0
PTS_POSITION_LATE     = 2.0
PTS_CONTEXT_WRAPUP    = 1.25
PTS_LOW_WORD_DENSITY  = 1.5
PTS_VISUAL_END_CARD   = 1.0
PTS_VISUAL_FADE_OUT   = 1.0
PTS_QUIET_AUDIO       = 0.25
PTS_BW_SCREEN         = 0.5


# ---- THRESHOLDS --------------------------------------------------------

OUTRO_WINDOW_SEC = 90.0   # last N seconds of video

LOW_WPS_THRESHOLD      = 0.8
QUIET_RMS_RATIO_MAX    = 0.6

ENDCARD_STATIC_MIN     = 0.6
ENDCARD_EDGE_MAX       = 0.08
FADE_OUT_BLACK_MIN     = 0.6

BW_SATURATION_MAX = 20.0
BW_BLACK_MAX      = 0.95


def score(audio_data, text_data, scene_data, video_data, scores, debug: bool = False):
    text_windows  = windows_from(text_data)
    audio_windows = windows_from(audio_data)
    video_windows = windows_from(video_data)
    rms_base = rms_baseline(audio_windows)

    # ---- total duration (important for outro positioning) ----
    total_duration = 0.0
    if scores:
        total_duration = scores[-1].get("end_s", 0.0) or 0.0

    results = []

    for i, row in enumerate(scores):
        dbg = DebugAccumulator(debug=debug)
        s = 0.0

        text_w  = at(text_windows,  i)
        audio_w = at(audio_windows, i)
        video_w = at(video_windows, i)

        win_start = row.get("start_s", 0.0) or 0.0
        time_from_end = total_duration - win_start

        # ---- OUTSIDE OUTRO WINDOW ----------------------------------------
        if time_from_end > OUTRO_WINDOW_SEC:
            row[LABEL] = 0.0
            results.append({
                "window_index": i,
                "label": LABEL,
                "score": 0.0,
                "debug": dbg.get()
            })
            continue

        # ---- TEXT CONTEXT ------------------------------------------------
        context_text = _context_text(text_windows, i)

        if _has_outro_phrase(context_text):
            s = dbg.add(s, PTS_OUTRO_KEYWORD, "outro_keyword")

        # ---- CONTEXT WRAP-UP --------------------------------------------
        if text_w:
            words = context_text.split()
            wc = max(len(words), 1)

            self_refs = len(re.findall(SELF_REF_PATTERN, context_text, re.IGNORECASE))
            self_ref_ratio = self_refs / wc

            has_wrapup = any(
                re.search(p, context_text, re.IGNORECASE)
                for p in WRAPUP_PATTERNS
            )

            if wc >= 8 and (self_ref_ratio > 0.1 or has_wrapup):
                s = dbg.add(s, PTS_CONTEXT_WRAPUP, "context_wrapup", {
                    "self_ref_ratio": round(self_ref_ratio, 3),
                    "has_wrapup": has_wrapup
                })

        # ---- POSITION (near end) ----------------------------------------
        s = dbg.add(s, PTS_POSITION_LATE, "position_late", {
            "time_from_end": round(time_from_end, 2)
        })

        # ---- WORD DENSITY -----------------------------------------------
        if text_w:
            feats = text_w.get("features", {})
            wc    = feats.get("word_count", 0)
            dur   = max(text_w.get("end_s", 0) - text_w.get("start_s", 0), 0.001)
            wps   = wc / dur

            if wps < LOW_WPS_THRESHOLD:
                s = dbg.add(s, PTS_LOW_WORD_DENSITY, "low_word_density", {"wps": wps})

        # ---- VISUAL -----------------------------------------------------
        if video_w:
            black_ratio  = video_w.get("black_frame_ratio",  0.0)
            static_ratio = video_w.get("static_frame_ratio", 0.0)
            edge_density = video_w.get("edge_density_mean",  0.0)
            mean_sat     = video_w.get("mean_hsv_s_mean",    0.0)

            is_end_card = (
                static_ratio >= ENDCARD_STATIC_MIN and
                edge_density <= ENDCARD_EDGE_MAX
            )

            is_fade_out = black_ratio >= FADE_OUT_BLACK_MIN

            is_bw_screen = (
                mean_sat <= BW_SATURATION_MAX and
                black_ratio < BW_BLACK_MAX and
                static_ratio > 0.4   # reduces false positives
            )

            if is_end_card:
                s = dbg.add(s, PTS_VISUAL_END_CARD, "visual_end_card", {
                    "static_ratio": static_ratio,
                    "edge_density": edge_density,
                })

            elif is_fade_out:
                s = dbg.add(s, PTS_VISUAL_FADE_OUT, "visual_fade_out", {
                    "black_ratio": black_ratio,
                })

            if is_bw_screen:
                s = dbg.add(s, PTS_BW_SCREEN, "visual_bw_screen", {
                    "mean_sat": mean_sat,
                    "black_ratio": black_ratio,
                })

        # ---- AUDIO ------------------------------------------------------
        if audio_w and rms_base > 0:
            rms = audio_w.get("features", {}).get("rms", 0.0)
            ratio = rms / rms_base

            if ratio <= QUIET_RMS_RATIO_MAX:
                s = dbg.add(s, PTS_QUIET_AUDIO, "quiet_audio", {"ratio": ratio})

        # ---- FINALIZE ---------------------------------------------------
        score_val = round(min(s, MAX_SCORE), 2)
        row[LABEL] = score_val

        results.append({
            "window_index": i,
            "label": LABEL,
            "score": score_val,
            "debug": dbg.get()
        })

    return results


def _context_text(text_windows, center_idx, radius=3):
    parts = []
    for i in range(center_idx - radius, center_idx + radius + 1):
        w = at(text_windows, i)
        if w and w.get("transcript"):
            parts.append(w["transcript"])
    return " ".join(parts).lower()


def _has_outro_phrase(text):
    return any(re.search(p, text, re.IGNORECASE) for p in OUTRO_PATTERNS)
