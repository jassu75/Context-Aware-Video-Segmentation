import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class PerFrameFeatures:
    timestamp: float
    mean_brightness: float
    mean_hsv_s: float
    mean_hsv_v: float
    color_hist: np.ndarray
    edge_density: float
    corner_edge_density: Dict[str, float]
    corner_gray_std: Dict[str, float]
    is_black: bool


@dataclass
class WindowVideoFeatures:
    window_index: int
    start_s: float
    end_s: float
    sampled_frame_times: List[float]

    mean_brightness_mean: float
    mean_brightness_std: float

    mean_hsv_s_mean: float
    mean_hsv_s_std: float

    mean_hsv_v_mean: float
    mean_hsv_v_std: float

    edge_density_mean: float
    edge_density_std: float

    top_left_edge_density_mean: float
    top_right_edge_density_mean: float
    bottom_left_edge_density_mean: float
    bottom_right_edge_density_mean: float

    top_left_gray_std_mean: float
    top_right_gray_std_mean: float
    bottom_left_gray_std_mean: float
    bottom_right_gray_std_mean: float

    top_left_frame_diff_mean: float
    top_right_frame_diff_mean: float
    bottom_left_frame_diff_mean: float
    bottom_right_frame_diff_mean: float

    black_frame_ratio: float
    static_frame_ratio: float

    frame_diff_mean: float
    frame_diff_max: float

    color_hist_mean: List[float]


class VideoProcessor:
    def __init__(
        self,
        video_path: str,
        window_sec: float = 2.0,
        hop_sec: float = 1.0,
        frame_count: int = 4,
    ):
        self.video_path = video_path
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.frame_count = frame_count

    def extract(self) -> Tuple[List[WindowVideoFeatures], dict]:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / video_fps if video_fps > 0 else 0.0

        video_info = {
            "video": Path(self.video_path).stem,
            "source_file": self.video_path,
            "fps": video_fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration_s": round(duration, 3),
            "window_sec": self.window_sec,
            "hop_sec": self.hop_sec,
            "frame_count": self.frame_count,
        }

        windows: List[WindowVideoFeatures] = []

        start = 0.0
        idx = 0

        while start < duration:
            end = min(start + self.window_sec, duration)
            frame_times = self._get_frame_times_for_window(start, end)

            per_frame_features = []
            gray_frames = []

            for t in frame_times:
                frame = self._read_frame_at_time(cap, t, video_fps, total_frames)
                feats, gray = self._extract_frame_features(frame, t)
                per_frame_features.append(feats)
                gray_frames.append(gray)

            window_features = self._aggregate_window_features(
                window_index=idx,
                start_s=start,
                end_s=end,
                frame_times=frame_times,
                per_frame_features=per_frame_features,
                gray_frames=gray_frames,
            )
            windows.append(window_features)

            start += self.hop_sec
            idx += 1

        cap.release()
        return windows, video_info

    def _get_frame_times_for_window(self, start_s: float, end_s: float) -> List[float]:
        duration = max(end_s - start_s, 1e-6)

        if self.frame_count < 1:
            raise ValueError("frame_count must be >= 1")

        fractions = np.linspace(0.0, 1.0, self.frame_count + 2)[1:-1]

        return [round(start_s + f * duration, 6) for f in fractions]

    def _read_frame_at_time(
        self,
        cap,
        time_s: float,
        fps: float,
        total_frames: int,
    ) -> np.ndarray:
        frame_number = min(int(round(time_s * fps)), max(total_frames - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Could not read frame at {time_s:.3f}s (frame {frame_number})")
        return frame

    def _extract_frame_features(
        self,
        frame: np.ndarray,
        timestamp: float,
    ) -> Tuple[PerFrameFeatures, np.ndarray]:
        small = cv2.resize(frame, (320, 180))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        mean_brightness = float(np.mean(gray))
        mean_hsv_s = float(np.mean(hsv[:, :, 1]))
        mean_hsv_v = float(np.mean(hsv[:, :, 2]))

        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / edges.size
        corner_edge_density, corner_gray_std = self._extract_corner_features(gray)

        is_black = mean_hsv_v < 12.0
        color_hist = self._compute_color_histogram(small)

        return (
            PerFrameFeatures(
                timestamp=timestamp,
                mean_brightness=mean_brightness,
                mean_hsv_s=mean_hsv_s,
                mean_hsv_v=mean_hsv_v,
                color_hist=color_hist,
                edge_density=edge_density,
                corner_edge_density=corner_edge_density,
                corner_gray_std=corner_gray_std,
                is_black=is_black,
            ),
            gray,
        )

    def _extract_corner_features(self, gray: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Measure small corner regions where publisher watermarks/logos often sit.

        This is intentionally generic: it does not know what NASA, Stanford, or
        TED look like. It only exposes corner detail/contrast so scorers can
        decide whether a stable visual mark is meaningful.
        """
        rois = self._corner_rois(gray)
        edge_density = {}
        gray_std = {}
        for name, roi in rois.items():
            edges = cv2.Canny(roi, 50, 150)
            edge_density[name] = float(np.count_nonzero(edges)) / edges.size
            gray_std[name] = float(np.std(roi))
        return edge_density, gray_std

    def _corner_rois(self, gray: np.ndarray) -> Dict[str, np.ndarray]:
        h, w = gray.shape[:2]
        roi_w = max(int(w * 0.28), 1)
        roi_h = max(int(h * 0.22), 1)
        return {
            "top_left": gray[:roi_h, :roi_w],
            "top_right": gray[:roi_h, w - roi_w:],
            "bottom_left": gray[h - roi_h:, :roi_w],
            "bottom_right": gray[h - roi_h:, w - roi_w:],
        }

    def _compute_color_histogram(self, frame: np.ndarray) -> np.ndarray:
        hist = []
        for ch in range(3):
            h = cv2.calcHist([frame], [ch], None, [16], [0, 256])
            cv2.normalize(h, h)
            hist.append(h.flatten())
        return np.concatenate(hist)

    def _aggregate_window_features(
        self,
        window_index: int,
        start_s: float,
        end_s: float,
        frame_times: List[float],
        per_frame_features: List[PerFrameFeatures],
        gray_frames: List[np.ndarray],
    ) -> WindowVideoFeatures:
        brightness_vals = np.array([f.mean_brightness for f in per_frame_features], dtype=np.float32)
        hsv_s_vals = np.array([f.mean_hsv_s for f in per_frame_features], dtype=np.float32)
        hsv_v_vals = np.array([f.mean_hsv_v for f in per_frame_features], dtype=np.float32)
        edge_vals = np.array([f.edge_density for f in per_frame_features], dtype=np.float32)
        corner_names = ("top_left", "top_right", "bottom_left", "bottom_right")
        corner_edge_vals = {
            name: np.array([f.corner_edge_density[name] for f in per_frame_features], dtype=np.float32)
            for name in corner_names
        }
        corner_std_vals = {
            name: np.array([f.corner_gray_std[name] for f in per_frame_features], dtype=np.float32)
            for name in corner_names
        }
        black_vals = np.array([1.0 if f.is_black else 0.0 for f in per_frame_features], dtype=np.float32)
        hist_vals = np.stack([f.color_hist for f in per_frame_features], axis=0)

        frame_diffs = []
        corner_frame_diffs = {name: [] for name in corner_names}
        for i in range(1, len(gray_frames)):
            diff = cv2.absdiff(gray_frames[i], gray_frames[i - 1])
            frame_diffs.append(float(np.mean(diff)))
            current_rois = self._corner_rois(gray_frames[i])
            previous_rois = self._corner_rois(gray_frames[i - 1])
            for name in corner_names:
                roi_diff = cv2.absdiff(current_rois[name], previous_rois[name])
                corner_frame_diffs[name].append(float(np.mean(roi_diff)))

        if frame_diffs:
            frame_diffs_arr = np.array(frame_diffs, dtype=np.float32)
        else:
            frame_diffs_arr = np.array([0.0], dtype=np.float32)
        corner_diff_vals = {
            name: (
                np.array(values, dtype=np.float32)
                if values else np.array([0.0], dtype=np.float32)
            )
            for name, values in corner_frame_diffs.items()
        }

        static_flags = (frame_diffs_arr < 2.5).astype(np.float32)

        return WindowVideoFeatures(
            window_index=window_index,
            start_s=round(start_s, 3),
            end_s=round(end_s, 3),
            sampled_frame_times=[round(t, 3) for t in frame_times],
            mean_brightness_mean=round(float(np.mean(brightness_vals)), 6),
            mean_brightness_std=round(float(np.std(brightness_vals)), 6),
            mean_hsv_s_mean=round(float(np.mean(hsv_s_vals)), 6),
            mean_hsv_s_std=round(float(np.std(hsv_s_vals)), 6),
            mean_hsv_v_mean=round(float(np.mean(hsv_v_vals)), 6),
            mean_hsv_v_std=round(float(np.std(hsv_v_vals)), 6),
            edge_density_mean=round(float(np.mean(edge_vals)), 6),
            edge_density_std=round(float(np.std(edge_vals)), 6),
            top_left_edge_density_mean=round(float(np.mean(corner_edge_vals["top_left"])), 6),
            top_right_edge_density_mean=round(float(np.mean(corner_edge_vals["top_right"])), 6),
            bottom_left_edge_density_mean=round(float(np.mean(corner_edge_vals["bottom_left"])), 6),
            bottom_right_edge_density_mean=round(float(np.mean(corner_edge_vals["bottom_right"])), 6),
            top_left_gray_std_mean=round(float(np.mean(corner_std_vals["top_left"])), 6),
            top_right_gray_std_mean=round(float(np.mean(corner_std_vals["top_right"])), 6),
            bottom_left_gray_std_mean=round(float(np.mean(corner_std_vals["bottom_left"])), 6),
            bottom_right_gray_std_mean=round(float(np.mean(corner_std_vals["bottom_right"])), 6),
            top_left_frame_diff_mean=round(float(np.mean(corner_diff_vals["top_left"])), 6),
            top_right_frame_diff_mean=round(float(np.mean(corner_diff_vals["top_right"])), 6),
            bottom_left_frame_diff_mean=round(float(np.mean(corner_diff_vals["bottom_left"])), 6),
            bottom_right_frame_diff_mean=round(float(np.mean(corner_diff_vals["bottom_right"])), 6),
            black_frame_ratio=round(float(np.mean(black_vals)), 6),
            static_frame_ratio=round(float(np.mean(static_flags)), 6),
            frame_diff_mean=round(float(np.mean(frame_diffs_arr)), 6),
            frame_diff_max=round(float(np.max(frame_diffs_arr)), 6),
            color_hist_mean=[round(float(v), 6) for v in np.mean(hist_vals, axis=0)],
        )

    def to_json_dict(self, windows: List[WindowVideoFeatures], video_info: dict) -> dict:
        return {
            **video_info,
            "num_windows": len(windows),
            "windows": [asdict(w) for w in windows],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract video features")
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/video_processor_output.json",
        help="output JSON path",
    )
    parser.add_argument(
        "--frame_count",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    processor = VideoProcessor(
        video_path=args.video_path,
        window_sec=2.0,
        hop_sec=1.0,
        frame_count=args.frame_count,
    )

    windows, video_info = processor.extract()
    output = processor.to_json_dict(windows, video_info)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved video features to {out_path}")
