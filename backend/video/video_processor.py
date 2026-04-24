import cv2
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple
import json
from pathlib import Path
import argparse



@dataclass
class PerFrameFeatures:
    timestamp: float
    mean_brightness: float
    mean_hsv_s: float
    mean_hsv_v: float
    color_hist: np.ndarray
    edge_density: float
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
            "frame_count": self.frame_count
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

        fractions = np.linspace(
            0.0,
            1.0,
            self.frame_count + 2
        )[1:-1]

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
                is_black=is_black,
            ),
            gray,
        )

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
        black_vals = np.array([1.0 if f.is_black else 0.0 for f in per_frame_features], dtype=np.float32)
        hist_vals = np.stack([f.color_hist for f in per_frame_features], axis=0)

        frame_diffs = []
        for i in range(1, len(gray_frames)):
            diff = cv2.absdiff(gray_frames[i], gray_frames[i - 1])
            frame_diffs.append(float(np.mean(diff)))

        frame_diffs_arr = np.array(frame_diffs, dtype=np.float32) if frame_diffs else np.array([0.0], dtype=np.float32)

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
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved video features to {out_path}")