"""
scorer_recap.py - score windows for recap / repeated boilerplate.

This is transcript-first. It catches explicit recap language such as
"previously on", "last time", and "quick recap", plus repeated transcript
blocks that appear far apart from each other. Generic filler like "anyway" is
not enough by itself.
"""

from __future__ import annotations

import re

from backend.classifier._shared import (
    windows_from, at, shot_density_near,
)


LABEL = "recap"


# ==== Phrase patterns ====================================================

RECAP_PATTERNS = [
    r"\bpreviously\s+on\b",
    r"\b(?:last|previous)\s+(?:episode|week)\b",
    r"\blast\s+time\s+(?:we|i|you)\s+(?:talked|covered|discussed|looked|learned|saw|left|were)\b",
    r"\bin\s+(?:the\s+)?(?:last|previous)\s+(?:video|episode|part)\b",
    r"\bquick\s+recap\b",
    r"\brecap(?:ping)?\b",
    r"\bto\s+(?:recap|summarize)\b",
    r"\bin\s+summary\b",
    r"\bas\s+(?:i|we)\s+(?:said|mentioned|covered|discussed)\s+(?:earlier|before|last\s+time)?\b",
    r"\b(?:earlier|before)\s+(?:we|i)\s+(?:said|covered|discussed|looked\s+at)\b",
]


# ==== Per-rule point allocations ========================================

PTS_RECAP_KEYWORD = 5.0
PTS_REPEATED_BOILERPLATE = 5.0
PTS_POSITION_INTRO = 2.5
PTS_NARRATION_SPEECH = 1.0
PTS_NO_PROMOTION = 1.0
PTS_MODERATE_SHOT_DENSITY = 0.5


# ==== Thresholds ========================================================

RECAP_INTRO_DURATION_SEC = 90.0
NARRATION_WPS_MIN = 1.0
NARRATION_WPS_MAX = 3.5
SHOT_DENSITY_RADIUS_SEC = 10.0
MODERATE_DENSITY_MIN = 0.10
MODERATE_DENSITY_MAX = 0.30


# ==== Scoring ===========================================================

def score(audio_data, text_data, scene_data, video_data, scores):
    text_windows = windows_from(text_data)
    repeated_indices = _repeated_boilerplate_indices(text_windows)
    duration = _video_duration(text_data, scores)

    results = []
    for i, row in enumerate(scores):
        s = 0.0
        text_w = at(text_windows, i)
        context_text = _context_text(text_windows, i)
        has_recap_keyword = _has_recap_phrase(context_text)
        win_start = row.get("start_s", 0.0) or 0.0
        is_intro_or_outro = (
            win_start < RECAP_INTRO_DURATION_SEC
            or (duration > 0 and win_start > duration - RECAP_INTRO_DURATION_SEC)
        )
        has_repeated_boilerplate = i in repeated_indices and is_intro_or_outro

        # Anchor on explicit recap language or a repeated transcript block.
        # Generic filler words like "anyway" are intentionally ignored.
        if text_w and (has_recap_keyword or has_repeated_boilerplate):
            feats = text_w.get("features", {})
            primary_evidence = 0.0
            if has_recap_keyword:
                primary_evidence = max(primary_evidence, PTS_RECAP_KEYWORD)
            if has_repeated_boilerplate:
                primary_evidence = max(primary_evidence, PTS_REPEATED_BOILERPLATE)
            s += primary_evidence

            if win_start < RECAP_INTRO_DURATION_SEC:
                s += PTS_POSITION_INTRO

            wc = feats.get("word_count", 0)
            window_dur = max(text_w.get("end_s", 1) - text_w.get("start_s", 0), 0.001)
            wps = wc / window_dur
            if NARRATION_WPS_MIN < wps < NARRATION_WPS_MAX:
                s += PTS_NARRATION_SPEECH

            if not feats.get("has_brand_mention") and not feats.get("has_self_promo"):
                s += PTS_NO_PROMOTION

            density = shot_density_near(row, scene_data, SHOT_DENSITY_RADIUS_SEC)
            if MODERATE_DENSITY_MIN <= density <= MODERATE_DENSITY_MAX:
                s += PTS_MODERATE_SHOT_DENSITY

        row[LABEL] = round(s, 2)
        results.append({
            "window_index": row["window_index"],
            "label": LABEL,
            "score": round(s, 2),
        })

    return results


# ==== Text helpers ======================================================

def _context_text(text_windows, center_idx, radius_windows=4) -> str:
    parts = []
    for idx in range(center_idx - radius_windows, center_idx + radius_windows + 1):
        w = at(text_windows, idx)
        if w and w.get("transcript"):
            parts.append(w["transcript"])
    return " ".join(parts).lower()


def _has_recap_phrase(text: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in RECAP_PATTERNS)


def _repeated_boilerplate_indices(text_windows) -> set[int]:
    """
    Mark windows whose surrounding transcript is repeated elsewhere far away.

    Overlapping 2s windows are naturally similar, so comparisons skip nearby
    windows and require enough content words to avoid matching tiny phrases.
    """
    normalized = []
    for idx, _window in enumerate(text_windows):
        tokens = _content_tokens(_context_text(text_windows, idx, radius_windows=5))
        normalized.append((idx, tokens))

    repeated = set()
    for pos, (idx, tokens) in enumerate(normalized):
        if len(tokens) < 8 or len(set(tokens)) < 5:
            continue
        token_set = set(tokens)
        for other_idx, other_tokens in normalized[pos + 1:]:
            if abs(other_idx - idx) < 45 or len(other_tokens) < 8 or len(set(other_tokens)) < 5:
                continue
            other_set = set(other_tokens)
            union = token_set | other_set
            if not union:
                continue
            similarity = len(token_set & other_set) / len(union)
            if similarity >= 0.82:
                repeated.add(idx)
                repeated.add(other_idx)
    return repeated


def _content_tokens(text: str) -> list[str]:
    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
        "from", "i", "in", "is", "it", "of", "on", "or", "so", "that",
        "the", "this", "to", "we", "you", "your",
    }
    words = re.findall(r"[a-z0-9']+", text.lower())
    return [w for w in words if len(w) > 2 and w not in stop_words]


def _video_duration(text_data, scores) -> float:
    if text_data and text_data.get("duration_s"):
        return float(text_data["duration_s"])
    if scores:
        return float(scores[-1].get("end_s", 0) or 0)
    return 0.0
