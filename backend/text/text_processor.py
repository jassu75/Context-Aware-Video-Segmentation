"""
text_processor.py

CSCI 576 — Multimodal Segmentation of Long-Form Online Video

Description:
Transcribes video audio using OpenAI Whisper ASR and extracts NLP features
(keyword detection, word counts). Outputs per-window results to JSON
aligned with audio_processor.py windows for multimodal classification.

Author: Jesus Ramos
Python Version: 3.12

Usage:
    python text_processor.py --video test_003.mp4 --output test_003_text.json
    python text_processor.py --video test_003.mp4 --model base --window 2.0

Dependencies:
    - openai-whisper
    - numpy
    - imageio-ffmpeg
"""

import argparse
import json
import re
import subprocess
import tempfile
import os
from pathlib import Path
from dataclasses import dataclass, field

import whisper
import numpy as np
import imageio_ffmpeg


# ==== Configuration defaults =================================================

# Must match audio_processor.py. The classifier expects window indices to
# line up across modalities.
WINDOW_SEC = 2.0
HOP_SEC    = 1.0

# Options: tiny, base, small, medium, large.
WHISPER_MODEL = "base"

# Whisper is hardcoded to expect 16 kHz input.
WHISPER_SR = 16000


# ==== Keyword patterns =======================================================
#
# Try to find non-content segments using regex.

AD_SPONSOR_PATTERNS = [
    # Sponsor callouts
    r"\bsponsored?\s+by\b",
    r"\bbrought\s+to\s+you\s+by\b",
    r"\bpartner(?:ed|ing|ship)?\s+with\b",
    r"\bin\s+partnership\s+with\b",
    r"\bpowered\s+by\b",

    # Promo / CTAs
    r"\buse\s+(?:my\s+)?(?:code|link)\b",
    r"\bpromo\s*code\b",
    r"\bdiscount\s*code\b",
    r"\bcoupon\s*code\b",
    r"\bspecial\s+offer\b",
    r"\blimited\s+time\s+offer\b",
    r"\bexclusive\s+(?:deal|offer|discount)\b",
    r"\bget\s+\d+%?\s*(?:off|discount)\b",
    r"\bfree\s+shipping\b",
    r"\bfree\s+trial\b",
    r"\bmoney[\s-]?back\s+guarantee\b",

    # URLs / link mentions
    r"\bvisit\s+\w+\.(?:com|co|io|org|net)\b",
    r"\bgo\s+to\s+\w+\.(?:com|co|io|org|net)\b",
    r"\bcheck\s+(?:out\s+)?the\s+link\b",
    r"\blink\s+(?:in\s+(?:the\s+)?)?description\b",
    r"\blink\s+(?:down\s+)?below\b",
    r"\bclick\s+(?:the\s+)?link\b",

    # Product pitches
    r"\bsign\s+up\s+(?:now|today|for)\b",
    r"\bdownload\s+(?:the\s+)?(?:app|now|today|for\s+free)\b",
    r"\btry\s+(?:it\s+)?(?:now|today|for\s+free)\b",
    r"\bstart\s+your\s+free\b",
]

SELF_PROMO_PATTERNS = [
    # Subscribe prompts
    r"\bsubscribe\b",
    r"\bhit\s+(?:the\s+)?(?:subscribe|bell|notification)\b",
    r"\bturn\s+on\s+notifications?\b",
    r"\bbell\s+icon\b",
    r"\bjoin\s+(?:the\s+)?(?:channel|community)\b",

    # Engagement bait
    r"\blike\s+(?:the\s+)?(?:video|this)\b",
    r"\bleave\s+a\s+(?:like|comment)\b",
    r"\bcomment\s+(?:down\s+)?below\b",
    r"\blet\s+me\s+know\s+(?:in\s+the\s+comments?|what\s+you\s+think)\b",
    r"\bshare\s+(?:this\s+)?(?:video|with)\b",

    # Cross-promo to other platforms
    r"\bfollow\s+(?:me\s+)?(?:on|@)\b",
    r"\b(?:my|the)\s+(?:twitter|instagram|tiktok|facebook|discord|patreon)\b",
    r"\b(?:twitter|instagram|tiktok|x)\s+@\b",

    # Merch / memberships
    r"\bmerch(?:andise)?\s+(?:store|link|available)\b",
    r"\bpatreon\b",
    r"\bmembership\b",
    r"\bsupport\s+(?:the\s+)?channel\b",
]

INTRO_OUTRO_PATTERNS = [
    # Intros
    r"\bwelcome\s+(?:back\s+)?to\b",
    r"\bhey\s+(?:guys|everyone|what'?s\s+up)\b",
    r"\bwhat'?s\s+(?:up|going\s+on)\b",
    r"\bhello\s+(?:and\s+)?welcome\b",
    r"\btoday\s+we(?:'re|\s+are)\s+(?:going\s+to|gonna)\b",
    r"\bin\s+this\s+(?:video|episode)\b",

    # Outros
    r"\bthanks?\s+for\s+watching\b",
    r"\bsee\s+you\s+(?:in\s+the\s+)?next\b",
    r"\buntil\s+next\s+time\b",
    r"\bthat'?s\s+(?:it\s+)?for\s+(?:today|this\s+video)\b",
    r"\bpeace\s+out\b",
    r"\bcatch\s+you\s+(?:later|next\s+time)\b",
    r"\bbye\s+(?:bye|for\s+now)\b",
]

RECAP_FILLER_PATTERNS = [
    # Recaps
    r"\bpreviously\s+on\b",
    r"\blast\s+(?:time|episode|week)\b",
    r"\bquick\s+recap\b",
    r"\bto\s+summarize\b",
    r"\bas\s+(?:i|we)\s+(?:said|mentioned)\b",

    # Filler / transitions
    r"\banyway(?:s)?\b",
    r"\bso\s+yeah\b",
    r"\bmoving\s+on\b",
    r"\bwithout\s+further\s+ado\b",
    r"\blet'?s\s+(?:get\s+)?(?:into|started|begin)\b",
]

# ==== Known brand / sponsor mentions =========================================
#
# Fires when a brand name is spoken out loud in the transcript. Weaker signal
# than AD_SPONSOR_PATTERNS — brands appear in real content too (reviews, news).
# The classifier should weight this lower than explicit sponsor phrasing.
#
# Sourced from manual review of the five test videos. Avg ad duration: 50.4s,
# min: 28.4s, max: 118.2s.
#
# TO ADD MORE: append lowercase strings, one per line. Word-boundary matching
# is case-insensitive, so "apple" won't match "applesauce".

BRAND_MENTIONS = [
    # Found in test videos
    "apple",
    "lays",
    "doritos",
    "starbucks",
    "pepsi",
    "dove",
    "sony",
    "ikea",
    "mcdonalds", "mcdonald's",
    "nike run club", "nrc",
    "google pixel",
    "coca cola", "coke",
    "bulgari", "bvlgari",
    # Add more brands here
]


@dataclass
class TranscriptSegment:
    """One Whisper segment — text plus word-level timing."""
    start_s: float
    end_s: float
    text: str
    words: list = field(default_factory=list)


# ==== Get audio for Whisper =========================================

def extract_audio_for_whisper(video_path: str, out_wav: str) -> str:
    """
    Same idea as audio_processor.extract_audio, but pinned at 16 kHz
    because that's what Whisper expects.
    """
    print(f"[text] Extracting audio from {video_path} ...")
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_path, "-y",
        "-i", video_path,
        "-ac", "1",                # mono
        "-ar", str(WHISPER_SR),    # 16 kHz
        "-vn",                     # drop video
        out_wav
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
    print(f"[text] WAV written to {out_wav}")
    return out_wav


# ==== Transcribe =====================================================

def transcribe_audio(wav_path: str, model_name: str = WHISPER_MODEL) -> list[TranscriptSegment]:
    """
    Run Whisper with word-level timestamps on.

    We load the WAV ourselves and pass the numpy array in, which skips
    Whisper's internal ffmpeg call (it would otherwise shell out to a
    system ffmpeg that might not be on PATH).
    """
    print(f"[text] Loading Whisper model '{model_name}' ...")
    model = whisper.load_model(model_name)

    print(f"[text] Loading audio from WAV ...")
    import wave
    with wave.open(wav_path, 'rb') as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    print(f"[text] Transcribing (this may take a while) ...")
    result = model.transcribe(
        audio,
        word_timestamps=True,
        verbose=False,
    )

    segments = []
    for seg in result.get("segments", []):
        words = []
        for w in seg.get("words", []):
            words.append({
                "word": w.get("word", "").strip(),
                "start": w.get("start", 0.0),
                "end": w.get("end", 0.0),
            })

        segments.append(TranscriptSegment(
            start_s=seg.get("start", 0.0),
            end_s=seg.get("end", 0.0),
            text=seg.get("text", "").strip(),
            words=words,
        ))

    total_words = sum(len(s.words) for s in segments)
    print(f"[text] Transcribed {len(segments)} segments, {total_words} words")
    return segments


# ==== Bucket words into analysis windows =============================

def get_text_for_window(segments: list[TranscriptSegment],
                        win_start: float, win_end: float) -> dict:
    """
    Collect every word whose midpoint falls inside [win_start, win_end).

    Using the midpoint (not the start) keeps a word from being claimed by
    two windows when it straddles a boundary.
    """
    window_words = []
    window_text_parts = []

    for seg in segments:
        # Skip segments that sit entirely outside the window
        if seg.end_s <= win_start or seg.start_s >= win_end:
            continue

        for w in seg.words:
            word_mid = (w["start"] + w["end"]) / 2
            if win_start <= word_mid < win_end:
                window_words.append(w)
                window_text_parts.append(w["word"])

    return {
        "text": " ".join(window_text_parts).strip(),
        "words": window_words,
        "word_count": len(window_words),
    }


# ====  Match patterns and score =======================================

def detect_patterns(text: str, patterns: list[str]) -> list[str]:
    """Case-insensitive search across the pattern list; returns every hit."""
    text_lower = text.lower()
    matches = []
    for pattern in patterns:
        for m in re.finditer(pattern, text_lower, re.IGNORECASE):
            matches.append(m.group())
    return matches


def detect_brand_mentions(text: str) -> list[str]:
    """Check whether any known brand name was spoken in this window."""
    text_lower = text.lower()
    matches = []
    for brand in BRAND_MENTIONS:
        pattern = r"\b" + re.escape(brand) + r"\b"
        for m in re.finditer(pattern, text_lower):
            matches.append(m.group())
    return matches


def compute_text_features(text: str, word_count: int) -> dict:
    """
    Build the feature dict for one window.

    Most of these are booleans the classifier can use directly. A sponsor
    read often sounds like normal speech in the audio features, so the
    keyword hits here are what give it away.
    """
    features = {
        "word_count": word_count,
        "char_count": len(text),
        "is_empty": word_count == 0,
    }

    ad_matches = detect_patterns(text, AD_SPONSOR_PATTERNS)
    promo_matches = detect_patterns(text, SELF_PROMO_PATTERNS)
    intro_outro_matches = detect_patterns(text, INTRO_OUTRO_PATTERNS)
    recap_matches = detect_patterns(text, RECAP_FILLER_PATTERNS)

    features["ad_sponsor_keywords"] = ad_matches
    features["ad_sponsor_count"] = len(ad_matches)
    features["has_ad_sponsor"] = len(ad_matches) > 0

    features["self_promo_keywords"] = promo_matches
    features["self_promo_count"] = len(promo_matches)
    features["has_self_promo"] = len(promo_matches) > 0

    features["intro_outro_keywords"] = intro_outro_matches
    features["intro_outro_count"] = len(intro_outro_matches)
    features["has_intro_outro"] = len(intro_outro_matches) > 0

    features["recap_filler_keywords"] = recap_matches
    features["recap_filler_count"] = len(recap_matches)
    features["has_recap_filler"] = len(recap_matches) > 0

    features["non_content_keyword_count"] = (
        features["ad_sponsor_count"] +
        features["self_promo_count"] +
        features["intro_outro_count"] +
        features["recap_filler_count"]
    )
    features["has_non_content_keywords"] = features["non_content_keyword_count"] > 0

    # Brand mentions — weaker signal, weight lower than explicit sponsor phrases
    brand_matches = detect_brand_mentions(text)
    features["brand_mentions"] = brand_matches
    features["brand_mention_count"] = len(brand_matches)
    features["has_brand_mention"] = len(brand_matches) > 0

    return features


# ==== Align windows with audio_processor =============================

def make_text_windows(segments: list[TranscriptSegment],
                      duration_s: float,
                      window_sec: float = WINDOW_SEC,
                      hop_sec: float = HOP_SEC) -> list[dict]:
    """
    Build text windows that line up 1:1 with the audio windows. Window
    index i here corresponds to window index i in the audio JSON.
    """
    windows = []
    start = 0.0
    idx = 0

    while start + window_sec <= duration_s:
        end = start + window_sec

        text_data = get_text_for_window(segments, start, end)
        features = compute_text_features(text_data["text"], text_data["word_count"])

        windows.append({
            "window_index": idx,
            "start_s": round(start, 3),
            "end_s": round(end, 3),
            "transcript": text_data["text"],
            "words": text_data["words"],
            "features": features,
        })

        start += hop_sec
        idx += 1

    # Partial tail window
    if start < duration_s:
        text_data = get_text_for_window(segments, start, duration_s)
        features = compute_text_features(text_data["text"], text_data["word_count"])

        windows.append({
            "window_index": idx,
            "start_s": round(start, 3),
            "end_s": round(duration_s, 3),
            "transcript": text_data["text"],
            "words": text_data["words"],
            "features": features,
        })

    return windows


# ==== Preserve full transcript =======================================

def build_full_transcript(segments: list[TranscriptSegment]) -> list[dict]:
    """
    Whisper's own segmentation, kept alongside the windowed version. Useful
    if something downstream wants natural sentence boundaries instead of
    our 2-second grid.
    """
    return [
        {
            "start_s": round(seg.start_s, 3),
            "end_s": round(seg.end_s, 3),
            "text": seg.text,
        }
        for seg in segments
    ]


# ==== Put it together ===============================================

def process_video(video_path: str,
                  output_path: str,
                  model_name: str = WHISPER_MODEL,
                  window_sec: float = WINDOW_SEC,
                  hop_sec: float = HOP_SEC) -> dict:
    """
    End-to-end: video → 16 kHz WAV → Whisper → windowed features → JSON.

    Output schema (aligned with audio_processor.py):
    {
      "video":        str,
      "source_file":  str,
      "duration_s":   float,
      "whisper_model": str,
      "window_sec":   float,
      "hop_sec":      float,
      "num_windows":  int,
      "full_transcript": [
        {"start_s": float, "end_s": float, "text": str}
      ],
      "windows": [
        {
          "window_index":  int,
          "start_s":       float,
          "end_s":         float,
          "transcript":    str,
          "words":         list[{word, start, end}],
          "features": {
            "word_count":               int,
            "char_count":               int,
            "is_empty":                 bool,
            "ad_sponsor_keywords":      list[str],
            "ad_sponsor_count":         int,
            "has_ad_sponsor":           bool,
            "self_promo_keywords":      list[str],
            "self_promo_count":         int,
            "has_self_promo":           bool,
            "intro_outro_keywords":     list[str],
            "intro_outro_count":        int,
            "has_intro_outro":          bool,
            "recap_filler_keywords":    list[str],
            "recap_filler_count":       int,
            "has_recap_filler":         bool,
            "non_content_keyword_count": int,
            "has_non_content_keywords": bool
          }
        }
      ]
    }
    """
    video_name = Path(video_path).stem

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, f"{video_name}_16k.wav")

        extract_audio_for_whisper(video_path, wav_path)

        print("[text] Getting audio duration ...")
        import wave
        with wave.open(wav_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration_s = frames / float(rate)
        print(f"[text] Duration: {duration_s:.2f}s")

        segments = transcribe_audio(wav_path, model_name)

        print(f"[text] Generating {window_sec}s windows with {hop_sec}s hop ...")
        windows = make_text_windows(segments, duration_s, window_sec, hop_sec)
        print(f"[text] {len(windows)} windows generated")

        full_transcript = build_full_transcript(segments)

    total_ad_kw = sum(w["features"]["ad_sponsor_count"] for w in windows)
    total_promo_kw = sum(w["features"]["self_promo_count"] for w in windows)
    windows_with_noncontent = sum(
        1 for w in windows if w["features"]["has_non_content_keywords"]
    )

    output = {
        "video": video_name,
        "source_file": str(video_path),
        "duration_s": round(duration_s, 3),
        "whisper_model": model_name,
        "window_sec": window_sec,
        "hop_sec": hop_sec,
        "num_windows": len(windows),
        "summary": {
            "total_words": sum(w["features"]["word_count"] for w in windows),
            "total_ad_sponsor_keywords": total_ad_kw,
            "total_self_promo_keywords": total_promo_kw,
            "windows_with_non_content_keywords": windows_with_noncontent,
        },
        "full_transcript": full_transcript,
        "windows": windows,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[text] Done — features written to {output_path}")
    return output


# ==== CLI ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run Whisper on a video and output per-window text "
                    "features aligned with audio_processor.py."
    )
    parser.add_argument("--video", required=True,
                        help="Path to input .mp4 file")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: <video>_text.json)")
    parser.add_argument("--model", default=WHISPER_MODEL,
                        choices=["tiny", "base", "small", "medium", "large"],
                        help=f"Whisper model size (default: {WHISPER_MODEL})")
    parser.add_argument("--window", type=float, default=WINDOW_SEC,
                        help=f"Window size in seconds (default: {WINDOW_SEC})")
    parser.add_argument("--hop", type=float, default=HOP_SEC,
                        help=f"Hop size in seconds (default: {HOP_SEC})")
    args = parser.parse_args()

    output_path = args.output or Path(args.video).stem + "_text.json"
    process_video(
        video_path=args.video,
        output_path=output_path,
        model_name=args.model,
        window_sec=args.window,
        hop_sec=args.hop,
    )


if __name__ == "__main__":
    main()
