"""
audio_processor.py

CSCI 576 — Multimodal Segmentation of Long-Form Online Video

Description:
Extracts audio from MP4 video files, segments it into overlapping windows,
and computes per-window feature vectors (RMS, spectral, MFCC). Outputs
results to JSON for downstream classification modules.

Author: Jesus Ramos
Python Version: 3.12

Usage:
    python audio_processor.py --video test_003.mp4 --output test_003_audio.json
    python audio_processor.py --video test_003.mp4 --window 2.0 --hop 1.0

Dependencies:
    - librosa
    - numpy
    - imageio-ffmpeg
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import imageio_ffmpeg


# ==== Configuration defaults =================================================

# 22050 Hz covers frequencies up to ~11 kHz (Nyquist), which is plenty for
# speech and music. CD-quality 44.1 kHz just doubles the data volume without
# adding anything useful for content-type classification. Mono because stereo
# positioning doesn't tell us whether something is an ad.
SAMPLE_RATE = 22050

# 2.0s window with 1.0s hop gives 50% overlap. The overlap matters because
# a transition landing near a window edge still falls fully inside its
# neighbor — no boundary gets missed.
WINDOW_SEC  = 2.0
HOP_SEC     = 1.0

# 13 MFCCs is the standard choice for speech-adjacent tasks. Lower
# coefficients describe the broad spectral shape (timbre), higher ones pick
# up finer detail. Beyond ~13 returns diminish.
N_MFCC = 13


# ==== Get audio from video ====================================

def extract_audio(video_path: str, out_wav: str, sr: int = SAMPLE_RATE) -> str:
    """
    Strip the video stream and write a mono PCM WAV.

    ffmpeg does the heavy lifting. Output is 16-bit PCM (ffmpeg's default
    for .wav), which librosa can load directly. Downmix to mono here so we
    don't have to deal with channel handling anywhere downstream.
    """
    print(f"[audio] Extracting audio from {video_path} ...")
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_path, "-y",
        "-i", video_path,
        "-ac", "1",       # mono
        "-ar", str(sr),   # resample
        "-vn",            # drop video stream
        out_wav
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
    print(f"[audio] WAV written to {out_wav}")
    return out_wav


# ==== Chop audio into windows ========================================

def make_windows(y: np.ndarray, sr: int,
                 window_sec: float, hop_sec: float) -> list[dict]:
    """
    Slide a fixed-length window across the signal.

    A single DFT over the whole clip would average everything together and
    lose all temporal info — we wouldn't know *when* an ad started. The
    STFT approach (Havaldar Ch. 5) solves this by running the DFT on short
    overlapping windows. The tail frame gets zero-padded so every window
    has the same length and the classifier sees a consistent shape.
    """
    window_samples = int(window_sec * sr)
    hop_samples    = int(hop_sec    * sr)
    total_samples  = len(y)

    windows = []
    start = 0
    while start + window_samples <= total_samples:
        end = start + window_samples
        windows.append({
            "start_s": round(start / sr, 3),
            "end_s":   round(end   / sr, 3),
            "samples": y[start:end],
        })
        start += hop_samples

    # Zero-pad whatever's left so the tail frame has the same shape.
    if start < total_samples:
        tail   = y[start:]
        padded = np.pad(tail, (0, window_samples - len(tail)))
        windows.append({
            "start_s": round(start / sr, 3),
            "end_s":   round(total_samples / sr, 3),
            "samples": padded,
        })

    return windows


# ==== Extract features for one window ================================

def extract_features(samples: np.ndarray, sr: int, n_mfcc: int = N_MFCC) -> dict:
    """
    Compute the feature vector for a single window.

    Time-domain:
      rms               broadcast ads are mastered louder than regular
                        content, so elevated energy is a useful cue
      zero_crossing     speech has high ZCR (fricatives), music is lower;
                        mixed ad audio lands in between

    Spectral:
      spectral_centroid "center of mass" of the spectrum — bright,
                        treble-heavy ad mixes read high here
      spectral_rolloff  frequency below which 85% of energy lives;
                        separates speech from music-bed content
      spectral_flatness ~1.0 for noise/music, ~0.0 for tonal speech
      tempo_bpm         music-backed ads have a beat; talking heads don't

    Perceptual:
      mfcc_mean/std     13 cepstral coefficients summarized across the
                        window. Biggest single-feature win for content-type
                        classification — captures the timbral fingerprint.
    """
    features = {}

    # Time-domain
    rms = float(np.sqrt(np.mean(samples ** 2)))
    features["rms"]       = round(rms, 6)
    features["is_silent"] = rms < 0.001  # empirical threshold; tune if needed

    zcr = librosa.feature.zero_crossing_rate(samples)
    features["zcr_mean"] = round(float(np.mean(zcr)), 6)
    features["zcr_std"]  = round(float(np.std(zcr)),  6)

    # Spectral (STFT)
    centroid = librosa.feature.spectral_centroid(y=samples, sr=sr)
    features["spectral_centroid_mean"] = round(float(np.mean(centroid)), 3)

    rolloff = librosa.feature.spectral_rolloff(y=samples, sr=sr, roll_percent=0.85)
    features["spectral_rolloff_mean"] = round(float(np.mean(rolloff)), 3)

    flatness = librosa.feature.spectral_flatness(y=samples)
    features["spectral_flatness_mean"] = round(float(np.mean(flatness)), 6)

    try:
        tempo, _ = librosa.beat.beat_track(y=samples, sr=sr)
        features["tempo_bpm"] = round(float(tempo), 2)
    except Exception:
        features["tempo_bpm"] = 0.0

    # MFCC — summarize across time so the vector length is fixed at 2*n_mfcc
    mfcc = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=n_mfcc)
    features["mfcc_mean"] = [round(float(v), 4) for v in np.mean(mfcc, axis=1)]
    features["mfcc_std"]  = [round(float(v), 4) for v in np.std( mfcc, axis=1)]

    return features


# ==== Flag dead-air runs =============================================

def flag_silence_runs(windows: list[dict],
                      min_run_sec: float = 1.5) -> list[dict]:
    """
    A single quiet window is probably just a pause mid-sentence. A run of
    them is something structural — holding screen, countdown, dead air
    between segments — which the taxonomy calls non-content.

    Two passes: first mark which silence run each silent window belongs to,
    then check whether the run is long enough to flag as dead air.
    """
    run_start = None
    for w in windows:
        if w["features"]["is_silent"]:
            if run_start is None:
                run_start = w["start_s"]
            w["silence_run_start"] = run_start
        else:
            run_start = None
            w["silence_run_start"] = None

    for w in windows:
        run_s = w.get("silence_run_start")
        if run_s is not None:
            run_len = w["end_s"] - run_s
            w["dead_air_flag"] = run_len >= min_run_sec
        else:
            w["dead_air_flag"] = False

    return windows


# ==== Putting everything together ===============================================

def process_video(video_path: str,
                  output_path: str,
                  window_sec: float = WINDOW_SEC,
                  hop_sec:    float = HOP_SEC,
                  sr:         int   = SAMPLE_RATE,
                  n_mfcc:     int   = N_MFCC) -> dict:
    """
    End-to-end: video → WAV → windows → features → JSON.

    Output schema (sync later with classifier):
    {
      "video":       str,
      "duration_s":  float,
      "sample_rate": int,
      "window_sec":  float,
      "hop_sec":     float,
      "n_mfcc":      int,
      "num_windows": int,
      "windows": [
        {
          "window_index":  int,
          "start_s":       float,
          "end_s":         float,
          "dead_air_flag": bool,
          "features": {
            "rms":                    float,
            "is_silent":              bool,
            "zcr_mean":               float,
            "zcr_std":                float,
            "spectral_centroid_mean": float,
            "spectral_rolloff_mean":  float,
            "spectral_flatness_mean": float,
            "tempo_bpm":              float,
            "mfcc_mean":              list[float],  // length n_mfcc
            "mfcc_std":               list[float]   // length n_mfcc
          }
        }
      ]
    }
    """
    video_name = Path(video_path).stem

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, f"{video_name}.wav")

        extract_audio(video_path, wav_path, sr=sr)

        print("[audio] Loading WAV into librosa ...")
        y, sr_loaded = librosa.load(wav_path, sr=sr, mono=True)
        duration_s   = float(len(y) / sr_loaded)
        print(f"[audio] Duration: {duration_s:.2f}s  |  Sample rate: {sr_loaded} Hz")

        raw_windows = make_windows(y, sr_loaded, window_sec, hop_sec)
        print(f"[audio] {len(raw_windows)} windows "
              f"({window_sec}s window / {hop_sec}s hop)")

        print("[audio] Extracting features ...")
        result_windows = []
        for i, w in enumerate(raw_windows):
            feats = extract_features(w["samples"], sr_loaded, n_mfcc)
            result_windows.append({
                "window_index": i,
                "start_s":      w["start_s"],
                "end_s":        w["end_s"],
                "features":     feats,
            })
            if (i + 1) % 50 == 0:
                print(f"  ... {i + 1}/{len(raw_windows)} windows done")

        result_windows = flag_silence_runs(result_windows)

    output = {
        "video":        video_name,
        "source_file":  str(video_path),
        "duration_s":   round(duration_s, 3),
        "sample_rate":  sr_loaded,
        "window_sec":   window_sec,
        "hop_sec":      hop_sec,
        "n_mfcc":       n_mfcc,
        "num_windows":  len(result_windows),
        "windows":      result_windows,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[audio] Done — features written to {output_path}")
    return output


# ==== CLI ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract per-window audio features from a video for "
                    "multimodal ad/content segmentation."
    )
    parser.add_argument("--video",  required=True,
                        help="Path to input .mp4 file")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: <video>_audio.json)")
    parser.add_argument("--window", type=float, default=WINDOW_SEC,
                        help=f"STFT window size in seconds (default: {WINDOW_SEC})")
    parser.add_argument("--hop",    type=float, default=HOP_SEC,
                        help=f"Hop size in seconds (default: {HOP_SEC})")
    parser.add_argument("--sr",     type=int,   default=SAMPLE_RATE,
                        help=f"Sample rate Hz (default: {SAMPLE_RATE})")
    parser.add_argument("--n_mfcc", type=int,   default=N_MFCC,
                        help=f"Number of MFCC coefficients (default: {N_MFCC})")
    args = parser.parse_args()

    output_path = args.output or Path(args.video).stem + "_audio.json"
    process_video(
        video_path  = args.video,
        output_path = output_path,
        window_sec  = args.window,
        hop_sec     = args.hop,
        sr          = args.sr,
        n_mfcc      = args.n_mfcc,
    )


if __name__ == "__main__":
    main()
