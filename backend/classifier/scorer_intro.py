from __future__ import annotations

import re

from backend.classifier._shared import (
    at,
    rms_baseline,
    windows_from,
    DebugAccumulator,
)

LABEL = "intro"
MAX_SCORE = 10.0

# ---- INTRO -----------------------------------------------

INTRO_PATTERNS = [
    r"\bwelcome\s+(?:back\s+)?to\b",
    r"\bhey\s+(?:guys|everyone|folks|what'?s\s+up)\b",
    r"\bwhat'?s\s+(?:up\s+)?(?:guys|everyone|folks)\b",
    r"\bin\s+(?:today'?s?|this)\s+video\b",
    r"\btoday\s+(?:i'?m|we'?re|we\s+are)\s+(?:going\s+to|gonna|covering|looking\s+at|talking\s+about)\b",
    r"\btoday\s+(?:on|in)\s+(?:the\s+)?(?:channel|video|episode)\b",
    r"\bthis\s+(?:video|episode)\s+(?:is\s+(?:all\s+)?about|covers|will\s+cover)\b",
    r"\bi'?m\s+(?:your\s+host|back\s+with\s+another)\b",
    r"\blet'?s\s+(?:get\s+(?:into\s+it|started)|dive\s+(?:right\s+)?in|jump\s+(?:right\s+)?in)\b",
    r"\bdon'?t\s+forget\s+to\s+(?:subscribe|like)\b",
    r"\bif\s+you'?re\s+new\s+(?:here|to\s+the\s+channel)\b",
    r"\bi'?m\s+\w+\s+and\b",
    r"\bmy\s+name\s+is\b",
    r"\bwelcome\s+to\s+(?:the\s+)?\w+\s+(?:channel|podcast|show)\b",
    r"\bin\s+this\s+video\s+(?:i'?ll|we'?ll|you'?ll)\b",
    r"\bby\s+the\s+end\s+of\s+this\b",
]

# ---- CONTEXT -----------------------------------------------

SELF_REF_PATTERN = r"\b(i|i'm|i'll|i've|we|we're|we'll|my|our)\b"

PREVIEW_PATTERNS = [
    r"\b(?:i'?ll|we'?ll|you'?ll)\s+(?:show|cover|talk|explain|look|go)\b",
    r"\bgoing\s+to\s+(?:show|cover|talk|explain|look|go)\b",
    r"\bby\s+the\s+end\s+of\s+this\b",
    r"\bstay\s+(?:tuned|with\s+us)\b",
    r"\blet'?s\s+(?:get\s+(?:into\s+it|started)|dive\s+(?:right\s+)?in|jump\s+(?:right\s+)?in)\b",
]

# ---- POINTS -------------------------------------------------

PTS_INTRO_KEYWORD     = 4.0
PTS_POSITION_EARLY    = 2.0
PTS_CONTEXT           = 1.25
PTS_LOW_WORD_DENSITY  = 1.5
PTS_VISUAL_LOGO_CARD  = 1.0
PTS_VISUAL_FADE_IN    = 1.0
PTS_VISUAL_MONTAGE    = 0.5
PTS_QUIET_AUDIO       = 0.25


# ---- THRESHOLDS --------------------------------------------

INTRO_WINDOW_SEC         = 90.0
LOGO_MAX_TIME_SEC        = 25.0

LOW_WPS_THRESHOLD        = 0.8
QUIET_RMS_RATIO_MAX   = 0.55

LOGO_STATIC_MIN          = 0.5
LOGO_EDGE_MAX            = 0.08
FADE_IN_BLACK_MIN        = 0.6
MONTAGE_DIFF_MIN         = 12.0


def score(audio_data, text_data, scene_data, video_data, scores, debug: bool = False):
    text_windows  = windows_from(text_data)
    audio_windows = windows_from(audio_data)
    video_windows = windows_from(video_data)
    rms_base = rms_baseline(audio_windows)


    results = []

    for i, row in enumerate(scores):
        dbg = DebugAccumulator(debug=debug)
        s = 0.0

        text_w  = at(text_windows,  i)
        audio_w = at(audio_windows, i)
        video_w = at(video_windows, i)

        win_start = row.get("start_s", 0.0) or 0.0

        # ---- OUTSIDE INTRO WINDOW ------------------------------------------
        if win_start >= INTRO_WINDOW_SEC:
            row[LABEL] = 0.0
            results.append({
                "window_index": i,
                "label": LABEL,
                "score": 0.0,
                "debug": dbg.get()
            })
            continue

        # ---- TEXT CONTEXT --------------------------------------------------
        context_text = _context_text(text_windows, i)

        if _has_intro_phrase(context_text):
            s = dbg.add(s, PTS_INTRO_KEYWORD, "intro_keyword")

        # ---- CONTEXT SETUP -------------------------------------------------
        if text_w:
            words = context_text.split()
            wc = max(len(words), 1)

            self_refs = len(re.findall(SELF_REF_PATTERN, context_text, re.IGNORECASE))
            self_ref_ratio = self_refs / wc

            has_preview = any(
                re.search(p, context_text, re.IGNORECASE)
                for p in PREVIEW_PATTERNS
            )

            if wc >= 8 and (self_ref_ratio > 0.12 or has_preview):
                s = dbg.add(s, PTS_CONTEXT, "context_setup", {
                    "self_ref_ratio": round(self_ref_ratio, 3),
                    "has_preview": has_preview
                })

        # ---- POSITION ------------------------------------------------------
        s = dbg.add(s, PTS_POSITION_EARLY, "position", {"win_start": win_start})

        # ---- WORD DENSITY --------------------------------------------------
        if text_w:
            feats = text_w.get("features", {})
            wc    = feats.get("word_count", 0)
            dur   = max(text_w.get("end_s", 0) - text_w.get("start_s", 0), 0.001)
            wps   = wc / dur

            if wps < LOW_WPS_THRESHOLD:
                s = dbg.add(s, PTS_LOW_WORD_DENSITY, "low_word_density", {"wps": wps})

        # ---- VISUAL --------------------------------------------------------
        if video_w:
            black_ratio  = video_w.get("black_frame_ratio",  0.0)
            static_ratio = video_w.get("static_frame_ratio", 0.0)
            edge_density = video_w.get("edge_density_mean",  0.0)
            frame_diff   = video_w.get("frame_diff_mean",    0.0)

            is_logo_card = (
                static_ratio >= LOGO_STATIC_MIN and
                edge_density <= LOGO_EDGE_MAX
            )

            if is_logo_card and win_start < LOGO_MAX_TIME_SEC:
                s = dbg.add(s, PTS_VISUAL_LOGO_CARD, "visual_logo_card", {
                    "static_ratio": static_ratio,
                    "edge_density": edge_density,
                    "black_ratio": black_ratio,
                })

            elif black_ratio >= FADE_IN_BLACK_MIN and win_start < 10.0:
                s = dbg.add(s, PTS_VISUAL_FADE_IN, "visual_fade_in_black", {
                    "black_ratio": black_ratio,
                })

            elif frame_diff >= MONTAGE_DIFF_MIN:
                s = dbg.add(s, PTS_VISUAL_MONTAGE, "visual_montage", {
                    "frame_diff_mean": frame_diff,
                })

         # ---- AUDIO ----
        if audio_w and rms_base > 0:
            rms = audio_w.get("features", {}).get("rms", 0.0)
            ratio = rms / rms_base

            if ratio <= QUIET_RMS_RATIO_MAX:
                s = dbg.add(s, PTS_QUIET_AUDIO, "quiet_audio", {"ratio": ratio})

        # ---- FINALIZE ------------------------------------------------------
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


def _has_intro_phrase(text):
    return any(re.search(p, text, re.IGNORECASE) for p in INTRO_PATTERNS)