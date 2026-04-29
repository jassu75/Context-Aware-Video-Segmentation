"""
scorer_self_promo.py — score windows for self-promotion / channel promotion.

Catches "subscribe", "like and share", "follow me on...", "smash that bell",
"check the description for my Patreon", merch plugs, and anything else where
the host is selling their own brand/audience-action rather than a sponsor's.

Almost always at the very start or end of a video, almost always
speech-heavy (host directly addressing camera), and almost always shot
in a sustained single take rather than a rapid-cut sequence.

Anchored on the has_self_promo keyword from text_processor — without that,
the score caps below the label threshold so we don't false-positive on
every intro window.

Author: Jesus Ramos
"""

from __future__ import annotations

import re

from backend.classifier._shared import (
    windows_from, at, rms_baseline, shot_containing_window,
)


LABEL = "self_promo"
MAX_POINTS = 10.0


# ==== Phrase patterns ====================================================

SELF_PROMO_PATTERNS = [
    r"\b(?:please\s+)?subscribe(?:\s+to\s+(?:my|the)\s+channel)?\b",
    r"\bhit\s+(?:the\s+)?(?:subscribe|bell|notification)s?\b",
    r"\bturn\s+on\s+notifications?\b",
    r"\b(?:smash|click|tap)\s+(?:the\s+)?(?:like|subscribe|bell)\b",
    r"\bleave\s+a\s+comment\b",
    r"\bcomment\s+(?:down\s+)?below\b",
    r"\blet\s+me\s+know\s+in\s+the\s+comments?\b",
    r"\bfollow\s+(?:me|us)\s+on\s+(?:instagram|tiktok|twitter|x|facebook|discord)\b",
    r"\b(?:my|our)\s+(?:instagram|tiktok|twitter|x|facebook|discord|patreon)\b",
    r"\bjoin\s+(?:my|our|the)\s+(?:channel|community|discord|patreon)\b",
    r"\bsupport\s+(?:my|our|the)\s+channel\b",
    r"\bcheck\s+out\s+(?:my|our)\s+(?:channel|podcast|patreon|merch|merchandise|website)\b",
    r"\bmerch(?:andise)?\s+(?:is\s+)?(?:available|linked|in\s+the\s+description)\b",
    r"\blink\s+in\s+(?:the\s+)?description\b",
]


# ==== Per-rule point allocations ========================================

PTS_SELF_PROMO_KEYWORD   = 5.0   # the anchor: audience action / channel CTA
PTS_POSITION_INTRO_OUTRO = 2.0   # window sits in first or last X seconds
PTS_INTRO_OUTRO_KEYWORD  = 1.0   # often co-occurs ("welcome back", "see you")
PTS_SPEECH_HEAVY         = 1.0   # host directly talking to camera
PTS_SUSTAINED_SHOT       = 0.5   # window is inside a long single shot
PTS_AUDIO_BASELINE       = 0.5   # audio profile matches the host's regular speech


# ==== Thresholds ========================================================

# What counts as "intro" or "outro" position.
INTRO_DURATION_SEC = 60.0
OUTRO_DURATION_SEC = 60.0

# Speech-heavy = host narrating directly. Above ~1.5 wps is normal speech.
SPEECH_HEAVY_WPS = 1.5

# Audio similarity — RMS within ±30% of baseline = "host's normal level".
RMS_BASELINE_TOLERANCE = 0.3

# Sustained shot — the shot containing this window must be at least this
# long. Self-promo end cards / host monologues are usually 10-30s shots.
SUSTAINED_SHOT_MIN_SEC = 10.0


# ==== Scoring ===========================================================

def score(audio_data, text_data, scene_data, video_data, scores):
    text_windows  = windows_from(text_data)
    audio_windows = windows_from(audio_data)

    duration = _video_duration(audio_data, text_data, video_data, scores)
    rms_base = rms_baseline(audio_windows)

    results = []
    for i, row in enumerate(scores):
        s = 0.0

        text_w  = at(text_windows,  i)
        audio_w = at(audio_windows, i)
        context_text = _context_text(text_windows, i)
        has_self_promo = _has_self_promo_phrase(context_text)

        # ---- Text signals ----
        # Anchor on our own stricter phrase list. The text processor's broad
        # "like this" hit is useful for search, but too noisy for labels.
        if has_self_promo:
            s += PTS_SELF_PROMO_KEYWORD

        if text_w and has_self_promo:
            feats = text_w.get("features", {})
            if feats.get("has_intro_outro"):
                s += PTS_INTRO_OUTRO_KEYWORD

            wc = feats.get("word_count", 0)
            window_dur = max(text_w.get("end_s", 1) - text_w.get("start_s", 0), 0.001)
            if wc / window_dur > SPEECH_HEAVY_WPS:
                s += PTS_SPEECH_HEAVY

        # ---- Position signal ----
        win_start = row.get("start_s", 0.0) or 0.0
        is_intro = win_start < INTRO_DURATION_SEC
        is_outro = duration > 0 and win_start > duration - OUTRO_DURATION_SEC
        if has_self_promo and (is_intro or is_outro):
            s += PTS_POSITION_INTRO_OUTRO

        # ---- Scene signal: sustained shot (host on camera) ----
        shot = shot_containing_window(row, scene_data)
        if shot and has_self_promo:
            shot_duration = shot.get("end_s", 0) - shot.get("start_s", 0)
            if shot_duration >= SUSTAINED_SHOT_MIN_SEC:
                s += PTS_SUSTAINED_SHOT

        # ---- Audio similarity to baseline ----
        if audio_w and rms_base > 0 and has_self_promo:
            rms = audio_w.get("features", {}).get("rms", 0.0)
            ratio = rms / rms_base
            if abs(ratio - 1.0) < RMS_BASELINE_TOLERANCE:
                s += PTS_AUDIO_BASELINE

        row[LABEL] = round(s, 2)
        results.append({
            "window_index": row["window_index"],
            "label":        LABEL,
            "score":        round(s, 2),
        })

    return results


# ==== Small helpers =====================================================

def _video_duration(audio_data, text_data, video_data, scores) -> float:
    """Best guess at the full video length, however we can find it."""
    for src in (audio_data, text_data, video_data):
        if src and src.get("duration_s"):
            return float(src["duration_s"])
    if scores:
        return float(scores[-1].get("end_s", 0) or 0)
    return 0.0


def _context_text(text_windows, center_idx, radius_windows=4) -> str:
    parts = []
    for idx in range(center_idx - radius_windows, center_idx + radius_windows + 1):
        w = at(text_windows, idx)
        if w and w.get("transcript"):
            parts.append(w["transcript"])
    return " ".join(parts).lower()


def _has_self_promo_phrase(text: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in SELF_PROMO_PATTERNS)
