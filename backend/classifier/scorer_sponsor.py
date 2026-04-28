"""
scorer_sponsor.py - score windows for spoken sponsor reads.

Inserted commercial breaks behave differently enough that they live in
scorer_ad_break.py. This scorer stays transcript-centered: it looks for
phrases like "sponsored by", "use code", "link in the description", and
brand mentions that appear near those sponsor phrases.

Author: Jesus Ramos
"""

from __future__ import annotations

from backend.classifier._shared import at, windows_from


LABEL = "sponsor"


# ==== Per-rule point allocations ========================================

PTS_AD_SPONSOR_KEYWORD = 6.0
PTS_BRAND_MENTION = 2.0
PTS_PROMO_LANGUAGE = 1.0
PTS_SPEECH_CONTEXT = 1.0


# ==== Thresholds ========================================================

SPEECH_CONTEXT_WPS = 0.75


def score(audio_data, text_data, scene_data, video_data, scores):
    """
    Score every window 0-10 for spoken sponsor-read likelihood.

    A brand mention alone is not enough; movies, reviews, and news can name
    brands as part of the core content. Brand points only count when the
    same local transcript already contains explicit sponsor/promo wording.
    """
    text_windows = windows_from(text_data)

    results = []
    for i, row in enumerate(scores):
        s = 0.0
        text_w = at(text_windows, i)

        if text_w:
            feats = text_w.get("features", {})
            has_sponsor_phrase = feats.get("has_ad_sponsor", False)

            if has_sponsor_phrase:
                s += PTS_AD_SPONSOR_KEYWORD

                if feats.get("has_brand_mention"):
                    s += PTS_BRAND_MENTION

                if feats.get("ad_sponsor_count", 0) > 1:
                    s += PTS_PROMO_LANGUAGE

                wc = feats.get("word_count", 0)
                window_dur = max(text_w.get("end_s", 1) - text_w.get("start_s", 0), 0.001)
                if wc / window_dur >= SPEECH_CONTEXT_WPS:
                    s += PTS_SPEECH_CONTEXT

        row[LABEL] = round(s, 2)
        results.append({
            "window_index": row["window_index"],
            "label": LABEL,
            "score": round(s, 2),
        })

    return results
