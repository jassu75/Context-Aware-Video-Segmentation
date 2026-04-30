"""
scorer_ad_break.py - score inserted commercial/ad-break intervals.

Unlike creator sponsor reads, inserted ads are often spliced-in clips with
their own visual style, audio mix, and temporal structure. This scorer builds
candidate intervals from multimodal evidence, then assigns a high score to
windows inside those candidates.
"""

from __future__ import annotations

from statistics import median

from backend.classifier._shared import at, scenes_from, windows_from


LABEL = "ad_break"
MAX_SCORE = 10.0


# Ground-truth ads in the supplied set are about 28s-118s. Keep the range a
# little wider so the rules are not hardcoded to the five files.
MIN_AD_DURATION_SEC = 20.0
MAX_AD_DURATION_SEC = 130.0


def score(audio_data, text_data, scene_data, video_data, scores, debug=False):
    audio_windows = windows_from(audio_data)
    text_windows = windows_from(text_data)
    video_windows = windows_from(video_data)
    scenes = scenes_from(scene_data)

    candidates = []
    candidates.extend(_scene_block_candidates(scenes, audio_windows, text_windows, video_windows))
    candidates.extend(_short_scene_cluster_candidates(scenes, audio_windows, text_windows, video_windows))
    candidates.extend(_quiet_desaturated_subcluster_candidates(scenes, audio_windows, text_windows, video_windows))
    candidates.extend(_low_speech_visual_candidates(audio_windows, text_windows, video_windows))
    candidates.extend(_high_motion_montage_candidates(audio_windows, video_windows))
    candidates = _merge_intervals(candidates, max_gap_sec=3.0)
    candidates = _expand_ad_opening_boundaries(candidates, audio_windows, text_windows, video_windows)
    candidates = _merge_intervals(candidates, max_gap_sec=3.0)

    results = []
    for i, row in enumerate(scores):
        s = 0.0
        win_start = row.get("start_s", 0.0) or 0.0
        win_end = row.get("end_s", win_start) or win_start

        for cand in candidates:
            if _overlap(win_start, win_end, cand["start"], cand["end"]) > 0:
                s = max(s, cand["score"])

        row[LABEL] = round(s, 2)
        results.append({
            "window_index": row["window_index"],
            "label": LABEL,
            "score": round(s, 2),
        })

    return results


def _scene_block_candidates(scenes, audio_windows, text_windows, video_windows):
    """
    Detect ad-like single scene blocks.

    This catches inserted ads that SceneDetect sees as one coherent external
    clip: commercial-sized duration, sparse transcript, louder-than-baseline
    audio, and a darker visual profile than the surrounding movie/content.
    """
    if not scenes or not audio_windows or not text_windows or not video_windows:
        return []

    baselines = _baselines(audio_windows, video_windows)

    out = []
    for scene in scenes:
        start = float(scene.get("start_s", 0.0) or 0.0)
        end = float(scene.get("end_s", start) or start)
        dur = end - start
        if not (25.0 <= dur <= MAX_AD_DURATION_SEC):
            continue

        features = _interval_features(audio_windows, text_windows, video_windows, start, end, baselines)

        sparse_speech = features["words"] <= 1.0
        active_enough = features["static"] <= 0.55
        strong_visual_shift = (
            features["brightness_ratio"] <= 0.70
            or features["brightness_ratio"] >= 1.80
            or features["black"] >= 0.04
        )

        if sparse_speech and active_enough and strong_visual_shift:
            out.append({
                "start": start,
                "end": end,
                "score": 8.5,
                "reason": "scene_block",
            })

    return out


def _short_scene_cluster_candidates(scenes, audio_windows, text_windows, video_windows):
    """
    Detect commercial blocks made of many short scenes.

    Most missed supplied ads are not one neat SceneDetect block. They appear
    as compact clusters of quick cuts, low-to-moderate transcript density,
    active frames, and at least one visual/audio shift from the surrounding
    content.
    """
    if not scenes or not audio_windows or not text_windows or not video_windows:
        return []

    baselines = _baselines(audio_windows, video_windows)
    out = []
    i = 0

    while i < len(scenes):
        scene_start = float(scenes[i].get("start_s", 0.0) or 0.0)
        scene_end = float(scenes[i].get("end_s", scene_start) or scene_start)
        scene_dur = scene_end - scene_start

        if scene_dur > 22.0:
            i += 1
            continue

        start = scene_start
        end = scene_end
        count = 0
        j = i
        while j < len(scenes):
            s = float(scenes[j].get("start_s", 0.0) or 0.0)
            e = float(scenes[j].get("end_s", s) or s)
            if e - s > 22.0:
                break
            end = e
            count += 1
            j += 1

        dur = end - start
        if count >= 2 and MIN_AD_DURATION_SEC <= dur <= MAX_AD_DURATION_SEC:
            features = _interval_features(audio_windows, text_windows, video_windows, start, end, baselines)
            motion = features["frame_diff_max"] >= 38.0 or features["edge"] >= 0.10
            active = features["static"] <= 0.35
            speech_ok = features["words"] <= 4.2
            distinct = (
                features["rms_ratio"] >= 1.20
                or features["flatness"] >= 0.04
                or features["brightness_ratio"] <= 0.75
                or features["brightness_ratio"] >= 1.25
                or features["saturation"] >= 90.0
                or features["black"] >= 0.04
            )

            if motion and active and speech_ok and distinct:
                out.append({
                    "start": start,
                    "end": end,
                    "score": 8.0,
                    "reason": "short_scene_cluster",
                })

        i = j

    return out


def _low_speech_visual_candidates(audio_windows, text_windows, video_windows):
    """
    Detect dark/desaturated low-speech stretches.

    This catches commercials that do not trigger many scene cuts but have a
    distinct dark/desaturated look and little transcript content.
    """
    n = min(len(text_windows), len(video_windows))
    if n == 0:
        return []

    flags = []
    for i in range(n):
        text_w = text_windows[i]
        video_w = video_windows[i]
        word_count = text_w.get("features", {}).get("word_count", 0)
        flags.append(
            word_count <= 1
            and video_w.get("mean_hsv_s_mean", 999.0) < 90.0
            and video_w.get("mean_brightness_mean", 999.0) < 45.0
            and video_w.get("static_frame_ratio", 1.0) < 0.30
        )

    intervals = _runs_from_flags(video_windows[:n], flags, min_run_sec=70.0, max_gap_sec=20.0)
    return [
        {"start": start, "end": end, "score": 8.0, "reason": "low_speech_visual"}
        for start, end in intervals
        if MIN_AD_DURATION_SEC <= end - start <= MAX_AD_DURATION_SEC
    ]


def _quiet_desaturated_subcluster_candidates(scenes, audio_windows, text_windows, video_windows):
    """
    Detect quieter ad inserts embedded inside a longer scene run.

    Some ads do not have elevated audio or many hard scene cuts. This scans
    inside maximal short-scene clusters for a compact, low-speech,
    desaturated sub-run with enough frame change to distinguish it from static
    creator/content screens.
    """
    if not scenes or not audio_windows or not text_windows or not video_windows:
        return []

    baselines = _baselines(audio_windows, video_windows)
    out = []
    i = 0

    while i < len(scenes):
        s = float(scenes[i].get("start_s", 0.0) or 0.0)
        e = float(scenes[i].get("end_s", s) or s)
        if e - s > 22.0:
            i += 1
            continue

        start_i = i
        while i < len(scenes):
            s = float(scenes[i].get("start_s", 0.0) or 0.0)
            e = float(scenes[i].get("end_s", s) or s)
            if e - s > 22.0:
                break
            i += 1
        end_i = i

        for first in range(start_i, end_i):
            for last in range(first + 1, end_i):
                start = float(scenes[first].get("start_s", 0.0) or 0.0)
                end = float(scenes[last].get("end_s", start) or start)
                dur = end - start
                count = last - first + 1
                if count < 2 or not (35.0 <= dur <= 70.0):
                    continue

                features = _interval_features(audio_windows, text_windows, video_windows, start, end, baselines)
                quiet_speech = features["words"] <= 0.5 and features["rms_ratio"] <= 1.25
                desaturated = features["saturation"] <= 75.0
                not_static = features["static"] <= 0.60
                enough_motion = features["frame_diff_max"] >= 30.0 or features["edge"] >= 0.075

                if quiet_speech and desaturated and not_static and enough_motion:
                    out.append({
                        "start": start,
                        "end": end,
                        "score": 7.5,
                        "reason": "quiet_desaturated_subcluster",
                    })

    return out


def _high_motion_montage_candidates(audio_windows, video_windows):
    """
    Detect short energetic commercial montages.

    This is intentionally strict. Action scenes can also have high motion, so
    the interval needs sustained frame changes plus elevated audio energy.
    """
    n = min(len(audio_windows), len(video_windows))
    if n == 0:
        return []

    rms_base = _median_feature(audio_windows, lambda w: w.get("features", {}).get("rms", 0.0))
    flags = []
    for i in range(n):
        audio_w = audio_windows[i]
        video_w = video_windows[i]
        rms = audio_w.get("features", {}).get("rms", 0.0)
        flags.append(
            rms_base > 0
            and rms >= rms_base * 1.65
            and video_w.get("frame_diff_max", 0.0) > 50.0
            and video_w.get("edge_density_mean", 0.0) > 0.08
            and video_w.get("static_frame_ratio", 1.0) < 0.10
        )

    intervals = _runs_from_flags(video_windows[:n], flags, min_run_sec=20.0, max_gap_sec=5.0)
    return [
        {"start": start, "end": end, "score": 7.0, "reason": "high_motion_montage"}
        for start, end in intervals
        if MIN_AD_DURATION_SEC <= end - start <= 60.0
    ]


def _expand_ad_opening_boundaries(candidates, audio_windows, text_windows, video_windows):
    """
    Pull a detected ad start backward when the immediately preceding windows
    look like a quiet on-screen-text commercial opening.

    This handles ads whose first shot is swallowed by the previous long scene,
    then only becomes obvious to SceneDetect once rapid cuts begin.
    """
    if not candidates or not text_windows or not video_windows:
        return candidates

    baselines = _baselines(audio_windows, video_windows)
    expanded = []

    for cand in candidates:
        start = float(cand["start"])
        best_start = start

        for window in reversed(video_windows):
            window_start = float(window.get("start_s", 0.0) or 0.0)
            preroll_dur = start - window_start
            if preroll_dur < 8.0:
                continue
            if preroll_dur > 21.0:
                break

            features = _interval_features(audio_windows, text_windows, video_windows, window_start, start, baselines)
            if _looks_like_text_heavy_ad_opening(features):
                best_start = window_start

        new_cand = dict(cand)
        if best_start < start:
            new_cand["start"] = best_start
            new_cand["reason"] = f"{new_cand['reason']}+opening_preroll"
        expanded.append(new_cand)

    return expanded


def _looks_like_text_heavy_ad_opening(features):
    return (
        features["words"] <= 0.5
        and features["static"] <= 0.20
        and (features["frame_diff_max"] >= 25.0 or features["edge"] >= 0.08)
        and features["saturation"] >= 65.0
    )


def _runs_from_flags(windows, flags, min_run_sec, max_gap_sec):
    out = []
    i = 0
    max_gap_windows = max(0, int(round(max_gap_sec)))
    while i < len(flags):
        if not flags[i]:
            i += 1
            continue

        start_idx = i
        last_true_idx = i
        gap = 0
        j = i + 1
        while j < len(flags):
            if flags[j]:
                last_true_idx = j
                gap = 0
            else:
                gap += 1
                if gap > max_gap_windows:
                    break
            j += 1

        start = float(windows[start_idx].get("start_s", 0.0) or 0.0)
        end = float(windows[last_true_idx].get("end_s", start) or start)
        if end - start >= min_run_sec:
            out.append((start, end))
        i = j

    return out


def _merge_intervals(candidates, max_gap_sec=0.0):
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda c: (c["start"], c["end"]))
    merged = [dict(candidates[0])]

    for cand in candidates[1:]:
        prev = merged[-1]
        if cand["start"] <= prev["end"] + max_gap_sec:
            prev["end"] = max(prev["end"], cand["end"])
            prev["score"] = max(prev["score"], cand["score"])
            prev["reason"] = f"{prev['reason']}+{cand['reason']}"
        else:
            merged.append(dict(cand))

    return merged


def _mean_over_interval(windows, start, end, getter):
    vals = [
        getter(w)
        for w in windows
        if _overlap(float(w.get("start_s", 0.0) or 0.0), float(w.get("end_s", 0.0) or 0.0), start, end) > 0
    ]
    return sum(vals) / len(vals) if vals else 0.0


def _median_feature(windows, getter):
    vals = [getter(w) for w in windows]
    vals = [v for v in vals if v > 0]
    return median(vals) if vals else 0.0


def _baselines(audio_windows, video_windows):
    return {
        "rms": _median_feature(audio_windows, lambda w: w.get("features", {}).get("rms", 0.0)),
        "brightness": _median_feature(video_windows, lambda w: w.get("mean_brightness_mean", 0.0)),
    }


def _interval_features(audio_windows, text_windows, video_windows, start, end, baselines):
    rms = _mean_over_interval(audio_windows, start, end, lambda w: w.get("features", {}).get("rms", 0.0))
    brightness = _mean_over_interval(video_windows, start, end, lambda w: w.get("mean_brightness_mean", 0.0))
    return {
        "words": _mean_over_interval(text_windows, start, end, lambda w: w.get("features", {}).get("word_count", 0)),
        "rms_ratio": rms / baselines["rms"] if baselines["rms"] else 0.0,
        "flatness": _mean_over_interval(
            audio_windows,
            start,
            end,
            lambda w: w.get("features", {}).get("spectral_flatness_mean", 0.0),
        ),
        "brightness_ratio": brightness / baselines["brightness"] if baselines["brightness"] else 0.0,
        "saturation": _mean_over_interval(video_windows, start, end, lambda w: w.get("mean_hsv_s_mean", 0.0)),
        "frame_diff_max": _mean_over_interval(video_windows, start, end, lambda w: w.get("frame_diff_max", 0.0)),
        "edge": _mean_over_interval(video_windows, start, end, lambda w: w.get("edge_density_mean", 0.0)),
        "static": _mean_over_interval(video_windows, start, end, lambda w: w.get("static_frame_ratio", 0.0)),
        "black": _mean_over_interval(video_windows, start, end, lambda w: w.get("black_frame_ratio", 0.0)),
    }


def _overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))
