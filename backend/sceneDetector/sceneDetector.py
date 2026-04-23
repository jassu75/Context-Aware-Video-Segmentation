import os
import sys
from scenedetect import ContentDetector, AdaptiveDetector, HistogramDetector, FrameTimecode, HashDetector, open_video, SceneManager, StatsManager, stats_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import json

# Test Params
file_name = "test_003.mp4"
detector_override = ContentDetector(threshold=67, weights=ContentDetector.Components(delta_hue=1.0, delta_sat=1.0, delta_lum=1.0, delta_edges=0.0), luma_only=False)

class ImageSceneDetector:
    def __init__(self, video_file_path, scene_detector = ContentDetector(), verbose = False):
        # file path variables
        self.video_file_path = video_file_path
        self.stats_csv_filepath = Path(video_file_path).with_suffix(".csv")
        self.scene_data_json_filepath = Path(video_file_path).with_suffix(".json")

        # params for scene processing
        self.scene_manager = SceneManager(stats_manager=StatsManager())
        self.scene_manager.add_detector(scene_detector)
        self.scene_data = []

        # debugging params
        self.verbose = verbose

        # start processing video for scenes
        self.process_for_scenes()
        self.save_scene_data_as_json()
        self.print_scene_data()

    # Callback to invoke on the first frame of every new scene detection.
    def on_new_scene(self, frame_img: np.ndarray, frame_num: int):
        if self.verbose:
            print("New scene found at frame %d." % frame_num)
    
    # Getters
    # Get scene data as an array of dictionaries
    def get_scene_data(self):
        return self.scene_data
    
    def save_scene_data_as_json(self):
        with open(self.scene_data_json_filepath, "w") as outputFile:
            json.dump(self.scene_data, outputFile, indent=4)
    
    def get_stats_csv_filepath(self):
        return self.stats_csv_filepath
    
    def get_video_filepath(self):
        return self.video_file_path

    # Display Functions
    def print_scene_data(self):
        # Print the list of scenes (start and end frames and timecodes)
        for i, scene in enumerate(self.scene_data):
            print(f'Scene {i}: Start Frame { scene["start_frame"] }, Start Time {scene["start_timecode"]}, Start Time (seconds) { scene["start_time_seconds"] }, End Frame {scene["end_frame"]}, End Time {scene["end_timecode"]}, { scene["end_time_seconds"] }')

    def plot_scene_stats(self):
        # Display data for how scenes were determined, which is by hue/sat/lum differences with adjacent frames
        df = pd.read_csv(self.stats_csv_filepath)
        fig, axes = plt.subplots(nrows=1, ncols=2)
        df.plot(x=stats_manager.COLUMN_NAME_FRAME_NUMBER, y='content_val', ax=axes[0])
        df.plot(x=stats_manager.COLUMN_NAME_TIMECODE, y='content_val', ax=axes[1])
        plt.show()

    # Video Processing Functions
    def process_for_scenes(self):
        video_path = open_video(self.video_file_path)

        self.scene_manager.detect_scenes(video=video_path, show_progress=True, callback=self.on_new_scene)
        self.scene_manager.stats_manager.save_to_csv(csv_file=self.stats_csv_filepath)

        scene_list = self.scene_manager.get_scene_list()

        self.scene_data.clear()

        for i, scene in enumerate(scene_list):
            self.scene_data.append({"start_frame": scene[0].get_frames(), "start_timecode": scene[0].get_timecode(), "start_time_seconds": scene[0].get_seconds(), "end_frame": scene[1].get_frames(), "end_timecode": scene[1].get_timecode(), "end_time_seconds": scene[1].get_seconds()})  
        
if __name__ == "__main__":
    video_path = os.path.join(".", "dataset", "videos_with_ads", file_name)
    
    # test code for ImageSceneDetetctor class
    image_scene_detector = ImageSceneDetector(video_path, detector_override)
    image_scene_detector.plot_scene_stats()

    sys.exit(0)
    