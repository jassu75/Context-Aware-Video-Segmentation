"""
pipeline.py — top-level driver for the analysis pipeline.

Given a video file, this runs the three analyzers (audio, text, scene
detection) in parallel, then feeds their outputs to the classifier and
returns a labeled timeline.

Threads (not processes) because all three workers spend their time in
C/C++ extensions that release the GIL — numpy/scipy for librosa,
PyTorch for Whisper, OpenCV for PySceneDetect. Threads keep memory and
startup costs low; processes would re-load Whisper per child.

Typical use from the UI thread:

    from backend.pipeline import analyze_video
    non_content_list, path_to_non_content_json = analyze_video("path/to/video.mp4", output_dir="analysis/")

Author: Jesus Ramos
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse

from backend.audio.audio_processor       import process_video as run_audio_processor
from backend.text.text_processor         import process_video as run_text_processor
from backend.sceneDetector.sceneDetector import ImageSceneDetector
from backend.classifier.classifier       import classify

# video_processor lives on Tejas's feature branch and may not be in main yet.
# Importing optionally lets the pipeline still run on machines that don't
# have it. If it's missing, we just pass video_data=None to the classifier.
try:
    from backend.video.video_processor import VideoProcessor
    _HAS_VIDEO_PROCESSOR = True
except ImportError as _e:
    print(f"[pipeline] video_processor not available - running without it ({_e})")
    _HAS_VIDEO_PROCESSOR = False


# ==== Public API ========================================================

def analyze_video(video_path, output_dir=None, write_intermediates: bool = True, debug: bool = False) -> tuple[list, Path]:
    """
    Run the full analysis pipeline on one video.

    Parameters
    ----------
    video_path : str | Path
        Path to the source video file.
    output_dir : str | Path | None
        Where to write JSON outputs. Defaults to "<video_dir>/analysis/".
    write_intermediates : bool
        If True, also dumps the raw audio/text/scene JSONs alongside the
        final timeline. Handy for debugging and UI visualization.

    Returns
    -------
    tuple containing list with non-content dicts (see classifier.classify docstring) and Path to non-content json.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_dir = Path(output_dir) if output_dir else video_path.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pipeline] video: {video_path.name}")
    print(f"[pipeline] output dir: {out_dir}")

    t_start = time.time()

    # Fire off every analyzer at once. Video joins if its module is available.
    n_workers = 4 if _HAS_VIDEO_PROCESSOR else 3
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        fut_audio = pool.submit(_run_audio, video_path, out_dir)
        fut_text  = pool.submit(_run_text,  video_path, out_dir)
        fut_scene = pool.submit(_run_scene, video_path, out_dir)
        fut_video = pool.submit(_run_video, video_path, out_dir) if _HAS_VIDEO_PROCESSOR else None

        # .result() re-raises any exception from the worker, so one
        # failure stops the whole pipeline. Good default — swap to a
        # try/except per future if you want partial-result behavior.
        audio_data = fut_audio.result()
        text_data  = fut_text.result()
        scene_data = fut_scene.result()
        video_data = fut_video.result() if fut_video else None

    print(f"[pipeline] all analyzers finished in {time.time() - t_start:.1f}s")

    print("[pipeline] fusing modalities via classifier")
    timeline = classify(audio_data, text_data, scene_data, video_data, debug=debug)

    # extract just the segments that are not labeled as video_content
    non_content_segments = _extract_non_content_segment_info(timeline)
    non_content_json_path = out_dir / f"{video_path.stem}_non_content_segments.json"

    if write_intermediates:
        timeline_path = out_dir / f"{video_path.stem}_timeline.json"
        _write_json(timeline_path, timeline)
        print(f"[pipeline] timeline written to {timeline_path}")

        _write_json(non_content_json_path, non_content_segments)
        print(f"[pipeline] non-content data written to {non_content_json_path}")

    total = time.time() - t_start
    print(f"[pipeline] done in {total:.1f}s  ({len(non_content_segments)} non-content segments, {len(timeline['timeline_segments'])} total segments)")
    return non_content_segments, non_content_json_path


# ==== Per-modality wrappers =============================================
# Thin adapters around each module. If a module's entry point ever
# changes, update it here — the rest of the pipeline doesn't need to know.

def _run_audio(video_path: Path, out_dir: Path) -> dict:
    print("[audio] starting")
    t = time.time()
    audio_json_path = out_dir / f"{video_path.stem}_audio.json"
    data = run_audio_processor(str(video_path), str(audio_json_path))
    print(f"[audio] done in {time.time() - t:.1f}s")
    return data


def _run_text(video_path: Path, out_dir: Path) -> dict:
    print("[text] starting")
    t = time.time()
    text_json_path = out_dir / f"{video_path.stem}_text.json"
    data = run_text_processor(str(video_path), str(text_json_path))
    print(f"[text] done in {time.time() - t:.1f}s")
    return data


def _run_video(video_path: Path, out_dir: Path) -> dict:
    """Run Tejas's per-window visual feature extractor."""
    print("[video] starting")
    t = time.time()

    processor = VideoProcessor(str(video_path), window_sec=2.0, hop_sec=1.0, frame_count=4)
    windows, video_info = processor.extract()
    data = processor.to_json_dict(windows, video_info)

    video_json_path = out_dir / f"{video_path.stem}_video.json"
    _write_json(video_json_path, data)

    print(f"[video] done in {time.time() - t:.1f}s")
    return data


def _run_scene(video_path: Path, out_dir: Path) -> dict:
    print("[scene] starting")
    t = time.time()

    detector = ImageSceneDetector(str(video_path))
    raw_scenes = detector.get_scene_data()

    # sceneDetector returns tuples: (start_frame, start_tc, end_frame, end_tc).
    # Normalize into the shape the classifier expects.
    scenes = []
    for scene in raw_scenes:
        scenes.append({
            "start_frame": int(scene["start_frame"]),
            "end_frame":   int(scene["end_frame"]),
            "start_s":     float(scene["start_time_seconds"]),
            "end_s":       float(scene["end_time_seconds"]),
        })

    data = {
        "video":       video_path.stem,
        "source_file": str(video_path),
        "num_scenes":  len(scenes),
        "scenes":      scenes,
    }

    scene_json_path = out_dir / f"{video_path.stem}_scenes.json"
    _write_json(scene_json_path, data)

    print(f"[scene] done in {time.time() - t:.1f}s  ({len(scenes)} scenes)")
    return data


# ==== Utility helpers ===================================================

def _timecode_to_seconds(tc) -> float:
    """Handle scenedetect FrameTimecode objects and plain numbers."""
    if hasattr(tc, "get_seconds"):
        return float(tc.get_seconds())
    return float(tc)


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

def _extract_non_content_segment_info(timeline_dict) -> list:
    """
    Helper function to extract specifically non-content segment info from the video analyzer dict
    Fields to extract:
    (start_seconds, end_seconds, type) for types that are not labeld as 'video_content'

    Returns
    -------
    A list of dicts where the list has the following format...
    [{"start_seconds": 123.45, "end_seconds": 678.90, "content_type": "intro"}, {"start_seconds": 789.90, "end_seconds": 12345.90, "content_type": "self_promo"}, ...]
    """
    non_content_list = []
    segments_list = timeline_dict.get("timeline_segments", []) if timeline_dict else []
    
    for segment in segments_list:
        start_s = segment.get("start_seconds", None) if segment else None
        end_s = segment.get("end_seconds", None) if segment else None
        content_type = segment.get("type", None) if segment else None

        if start_s is not None and end_s is not None and content_type is not None and content_type != "video_content":
            non_content_list.append({"start_seconds": float(start_s), "end_seconds": float(end_s), "content_type": str(content_type)})

    return non_content_list


# ==== CLI entry point ===================================================

def _print_usage():
    print("Usage: python -m backend.pipeline <video_path> [output_dir]")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run full video analysis pipeline"
    )

    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input video file"
    )

    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        default=None,
        help="Optional output directory"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output for scorers"
    )

    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write intermediate JSON files"
    )

    args = parser.parse_args()

    cli_start = time.time()

    result, result_path = analyze_video(
        args.video_path,
        output_dir=args.output_dir,
        debug=args.debug,
        write_intermediates=not args.no_write
    )

    print("\n---- non-content segment summary ----")
    for seg in result:
        print(
            f"  {seg['content_type']:14s}  "
            f"{seg['start_seconds']:7.2f}s – {seg['end_seconds']:7.2f}s  "
        )

    total_elapsed = time.time() - cli_start
    mins, secs = divmod(total_elapsed, 60)
    print(f"\n[pipeline] TOTAL elapsed: {int(mins)}m {secs:.1f}s  ({total_elapsed:.1f}s)")
