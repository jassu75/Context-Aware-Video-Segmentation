"""
_shared.py — small helpers used across the per-category scorers.

Anything more than a one-liner that two or more scorer modules need
goes here so we're not duplicating it. Keep it dependency-free.

Author: Jesus Ramos
"""

from __future__ import annotations

from statistics import median


# ==== Modality-data accessors ===========================================

def windows_from(modality_data) -> list:
    """Pull the windows list off a modality JSON. None-safe."""
    return modality_data.get("windows", []) if modality_data else []


def at(arr, i):
    """Safe indexing: returns None if i is out of range."""
    return arr[i] if 0 <= i < len(arr) else None


# ==== Audio baselines ===================================================

def rms_baseline(audio_windows) -> float:
    """
    Median RMS across the whole video. Median beats mean here because a
    loud ad block can pull the mean up and hide its own anomaly.
    """
    vals = [w.get("features", {}).get("rms", 0.0) for w in audio_windows]
    vals = [v for v in vals if v > 0]
    return median(vals) if vals else 0.0


# ==== Scene-data lookups ================================================

def scenes_from(scene_data) -> list:
    """Pull the scenes list off the scene-detector JSON. None-safe."""
    return scene_data.get("scenes", []) if scene_data else []


def shot_containing_window(window, scene_data):
    """
    Return the shot dict whose [start_s, end_s] interval contains the
    midpoint of this window. None if no scene_data or no match.
    """
    scenes = scenes_from(scene_data)
    if not scenes:
        return None
    win_mid = ((window.get("start_s", 0.0) or 0.0) + (window.get("end_s", 0.0) or 0.0)) / 2
    for s in scenes:
        if s.get("start_s", 0.0) <= win_mid <= s.get("end_s", 0.0):
            return s
    return None


def shot_density_near(window, scene_data, radius_sec: float = 10.0) -> float:
    """
    Shots per second in a window of ±radius_sec around this window's
    midpoint. Counts scene START times that fall inside the radius.

    High value = rapid cuts (typical of ad breaks, montages).
    Low value  = sustained single shot (typical of host monologue).
    """
    scenes = scenes_from(scene_data)
    if not scenes:
        return 0.0
    win_mid = ((window.get("start_s", 0.0) or 0.0) + (window.get("end_s", 0.0) or 0.0)) / 2
    lo = win_mid - radius_sec
    hi = win_mid + radius_sec
    count = sum(1 for s in scenes if lo <= s.get("start_s", 0.0) <= hi)
    return count / (2 * radius_sec)
