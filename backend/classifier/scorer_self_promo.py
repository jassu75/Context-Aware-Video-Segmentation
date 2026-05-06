"""
scorer_self_promo.py - score windows for self-promotion / channel promotion.

Self-promo is owned promotion: the content temporarily shifts away from the
primary topic to promote the creator, channel, publisher/platform, future
content, products, subscriptions, or viewer engagement. It is not a third-party
ad and it is not a persistent watermark.

Strong evidence is explicit CTA language ("subscribe", "follow us", "join our
Discord", "watch part 2", "new episode drops Friday") or a short publisher
title/end card at the very beginning/end of the video.

Author: Jesus Ramos
"""

from __future__ import annotations

import re

from backend.classifier._shared import (
    windows_from,
    at,
    rms_baseline,
    shot_containing_window,
)


LABEL = "self_promo"
MAX_SCORE = 10.0


# ==== Phrase patterns ====================================================

SELF_PROMO_PATTERNS = [
    # Engagement / subscription CTAs
    r"\b(?:please\s+)?subscribe(?:\s+to\s+(?:my|the)\s+channel)?\b",
    r"\bhit\s+(?:the\s+)?(?:subscribe|bell|notification)s?\b",
    r"\bturn\s+on\s+notifications?\b",
    r"\bbell\s+icon\b",
    r"\b(?:smash|click|tap)\s+(?:the\s+)?(?:like|subscribe|bell)\b",
    r"\bdon'?t\s+forget\s+to\s+(?:subscribe|like|comment|share)\b",
    r"\blike\s+and\s+comment\b",
    r"\blike\s+(?:this\s+)?video\b",
    r"\bleave\s+a\s+comment\b",
    r"\bcomment\s+(?:down\s+)?below\b",
    r"\blet\s+me\s+know\s+in\s+the\s+comments?\b",

    # Owned social/community support
    r"\bfollow\s+(?:me|us)\b",
    r"\bfollow\s+(?:me|us)\s+on\s+(?:instagram|tiktok|twitter|x|facebook|discord)\b",
    r"\b(?:my|our)\s+(?:instagram|tiktok|twitter|x|facebook|discord|patreon)\b",
    r"\bjoin\s+(?:my|our|the)\s+(?:channel|community|discord|patreon)\b",
    r"\bjoin\s+(?:my|our)\s+(?:discord|patreon)\b",
    r"\bsupport\s+(?:my|our|the)\s+channel\b",
    r"\bsupport\s+(?:me|us)\s+on\s+patreon\b",

    # Owned sites/products/apps/merch
    r"\bcheck\s+out\s+(?:my|our)\s+(?:channel|podcast|patreon|merch|merchandise|website)\b",
    r"\b(?:visit|go\s+to)\s+(?:my|our)\s+(?:website|site)\b",
    r"\bdownload\s+(?:my|our)\s+app\b",
    r"\buse\s+(?:my|our)\s+(?:code|promo\s*code)\b",
    r"\b(?:my|our)\s+(?:merch|merchandise|podcast|patreon|sponsors?)\b",
    r"\bmerch(?:andise)?\s+(?:is\s+)?(?:available|linked|in\s+the\s+description)\b",
    r"\blink\s+in\s+(?:the\s+)?description\b",

    # Future / related owned content
    r"\bwatch\s+(?:part\s+2|the\s+next\s+(?:part|episode|video))\b",
    r"\bnew\s+episode\s+drops\b",
    r"\bnext\s+episode\b",
    r"\bin\s+part\s+2\b",
    r"\bon\s+this\s+channel\b",
    r"\bhere\s+at\s+[\w\s]+(?:tv|channel)\b",
    r"\bour\s+(?:next\s+video|series|podcast|patreon\s+community)\b",
    r"\bcoming\s+up\b",
    r"\blater\s+in\s+this\s+video\b",
    r"\bstay\s+tuned\b",
]


# ==== Per-rule point allocations ========================================

PTS_PRIMARY_EVIDENCE     = 6.0   # owned CTA phrase or short publisher card
PTS_POSITION_INTRO_OUTRO = 2.0   # window sits in first or last X seconds
PTS_INTRO_OUTRO_KEYWORD  = 1.0   # often co-occurs ("welcome back", "see you")
PTS_SPEECH_HEAVY         = 1.0   # host directly talking to camera
PTS_SUSTAINED_SHOT       = 0.5   # window is inside a long single shot
PTS_AUDIO_BASELINE       = 0.5   # audio profile matches regular speech
PTS_VISUAL_STABILITY     = 1.0   # visual self-promo is usually stable/held


# ==== Thresholds ========================================================

# Spoken CTAs can happen during intro/outro regions.
INTRO_DURATION_SEC = 60.0
OUTRO_DURATION_SEC = 60.0

# Visual-only evidence is much noisier. Keep it restricted to brief publisher
# cards at the very beginning or very end. Persistent corner logos/watermarks
# are not self-promo because the content has not shifted away from the topic.
VISUAL_INTRO_DURATION_SEC = 18.0
VISUAL_OUTRO_DURATION_SEC = 4.0

# Speech-heavy = host narrating directly. Above ~1.5 wps is normal speech.
SPEECH_HEAVY_WPS = 1.5

# Audio similarity: RMS within +/-30% of baseline = "host's normal level".
RMS_BASELINE_TOLERANCE = 0.3

# Sustained shot: the shot containing this window must be at least this long.
SUSTAINED_SHOT_MIN_SEC = 10.0

STABLE_VISUAL_DIFF_MAX = 8.0


# ==== Scoring ===========================================================

def score(audio_data, text_data, scene_data, video_data, scores, debug=False):
    text_windows = windows_from(text_data)
    audio_windows = windows_from(audio_data)
    video_windows = windows_from(video_data)

    duration = _video_duration(audio_data, text_data, video_data, scores)
    rms_base = rms_baseline(audio_windows)

    results = []
    for i, row in enumerate(scores):
        s = 0.0

        text_w = at(text_windows, i)
        audio_w = at(audio_windows, i)
        video_w = at(video_windows, i)
        context_text = _context_text(text_windows, i)
        has_self_promo = _has_self_promo_phrase(context_text)
        has_visual_self_promo = _looks_like_publisher_card(video_w, text_w, row, duration)

        win_start = row.get("start_s", 0.0) or 0.0
        is_intro = win_start < INTRO_DURATION_SEC
        is_outro = duration > 0 and win_start > duration - OUTRO_DURATION_SEC

        # ---- Primary evidence ----
        if has_self_promo or has_visual_self_promo:
            s += PTS_PRIMARY_EVIDENCE

        # ---- Text signals ----
        if text_w and has_self_promo:
            feats = text_w.get("features", {})
            if feats.get("has_intro_outro"):
                s += PTS_INTRO_OUTRO_KEYWORD

            wc = feats.get("word_count", 0)
            window_dur = max(text_w.get("end_s", 1) - text_w.get("start_s", 0), 0.001)
            if wc / window_dur > SPEECH_HEAVY_WPS:
                s += PTS_SPEECH_HEAVY

        # ---- Position signal ----
        if has_self_promo and (is_intro or is_outro):
            s += PTS_POSITION_INTRO_OUTRO
        elif has_visual_self_promo:
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

        # ---- Visual stability ----
        if video_w and has_visual_self_promo and _is_visually_stable(video_w):
            s += PTS_VISUAL_STABILITY

        row[LABEL] = round(min(s, MAX_SCORE), 2)
        results.append({
            "window_index": row["window_index"],
            "label": LABEL,
            "score": round(min(s, MAX_SCORE), 2),
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


def _looks_like_publisher_card(video_w, text_w, row, duration: float) -> bool:
    """
    Detect short title cards / publisher cards used as self-promo by publishers.

    These are usually held near the start or final seconds, have very low
    saturation, and are either black title cards or bright simple cards with
    sparse text/logo. Longer credits are intentionally excluded by the narrow
    end window above.
    """
    if not video_w:
        return False

    start = row.get("start_s", 0.0) or 0.0
    near_start_or_end = (
        start < VISUAL_INTRO_DURATION_SEC
        or (duration > 0 and start > duration - VISUAL_OUTRO_DURATION_SEC)
    )
    if not near_start_or_end:
        return False

    word_count = 0
    if text_w:
        word_count = text_w.get("features", {}).get("word_count", 0) or 0
    low_speech = word_count <= 2

    brightness = video_w.get("mean_brightness_mean", 0.0)
    saturation = video_w.get("mean_hsv_s_mean", 0.0)
    edge = video_w.get("edge_density_mean", 0.0)
    static = video_w.get("static_frame_ratio", 0.0)
    black = video_w.get("black_frame_ratio", 0.0)
    frame_diff = video_w.get("frame_diff_mean", 999.0)

    black_card = black >= 0.75 and 0.004 <= edge <= 0.080 and static >= 0.50
    bright_card = (
        brightness >= 175.0
        and saturation <= 45.0
        and edge <= 0.060
        and frame_diff <= 12.0
    )
    return low_speech and (black_card or bright_card)


def _is_visually_stable(video_w) -> bool:
    if not video_w:
        return False
    if video_w.get("static_frame_ratio", 0.0) >= 0.50:
        return True
    return video_w.get("frame_diff_mean", 999.0) <= STABLE_VISUAL_DIFF_MAX
