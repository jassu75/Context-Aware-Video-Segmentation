import os
import sys
import cv2
from scenedetect import ContentDetector, AdaptiveDetector, HistogramDetector, FrameTimecode, HashDetector, open_video, SceneManager, StatsManager, stats_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import threading
import json

# Test Params
file_name = "test_003.mp4"
detector_override = ContentDetector(threshold=67, luma_only=True)

class ImageSceneDetector:
    def __init__(self, video_file_path, scene_detector = ContentDetector(threshold=67, luma_only=True)):
        # file path variables
        self.video_file_path = video_file_path
        self.stats_csv_filepath = Path(video_file_path).with_suffix(".csv")
        
        # params for tracking frame numbers
        self.start_frame_numbers_list = []
        self.start_frame_numbers_list_mutex = threading.Lock()
        self.end_frame_numbers_list = []
        self.end_frame_numbers_list_mutex = threading.Lock()

        # params for scene processing
        self.scene_manager = SceneManager(stats_manager=StatsManager())
        self.scene_manager.add_detector(scene_detector)
        self.scene_data = []

        # start processing video for scenes
        self.process_for_scenes()
        self.print_scene_data()

    # Callback to invoke on the first frame of every new scene detection.
    def on_new_scene(self, frame_img: np.ndarray, frame_num: int):
        #print("New scene found at frame %d." % frame_num)
        with self.start_frame_numbers_list_mutex:
            self.start_frame_numbers_list.append(frame_num)
        # the previous scene ends with the frame immediately preceeding the current frame_num
        with self.end_frame_numbers_list_mutex:
            self.end_frame_numbers_list.append(frame_num - 1)
    
    # Getters
    def get_scene_data_as_list(self):
        return self.scene_data
    
    def get_scene_data_as_json(self):
        pass
    
    def get_stats_csv_filepath(self):
        return self.stats_csv_filepath
    
    def get_video_filepath(self):
        return self.video_file_path

    # Display Functions
    def print_scene_data(self):
        # Print the list of scenes (start and end frames and timecodes)
        for i, scene in enumerate(self.scene_data):
            print(f'Scene {i}: Start Frame { scene[0] }, Start Time {scene[1]}, End Frame {scene[2]}, End Time {scene[3]}')

    def plot_scene_stats(self):
        # Display data for how scenes were determined
        df = pd.read_csv(self.stats_csv_filepath)
        fig, axes = plt.subplots(nrows=1, ncols=2)
        df.plot(x=stats_manager.COLUMN_NAME_FRAME_NUMBER, y='content_val', ax=axes[0])
        df.plot(x=stats_manager.COLUMN_NAME_TIMECODE, y='content_val', ax=axes[1])
        plt.show()

    # Video Processing Functions
    def process_for_scenes(self):
        self.start_frame_numbers_list_mutex.acquire()
        try:
            # make sure the list is cleared before trying to analyze for scenes
            # then append 0 as the first element since the callback function doesn't trigger with the very first scene
            self.start_frame_numbers_list.clear()
            self.start_frame_numbers_list.append(0)
        finally:
            self.start_frame_numbers_list_mutex.release()

        self.end_frame_numbers_list_mutex.acquire()
        try:
            # make sure the list is cleared before trying to analyze for scenes
            # don't need to pre-append anything because the ending scene can be determined by the start frame of the next scene
            self.end_frame_numbers_list.clear()
        finally:
            self.end_frame_numbers_list_mutex.release()

        video_path = open_video(self.video_file_path)

        self.scene_manager.detect_scenes(video=video_path, show_progress=True, callback=self.on_new_scene)
        self.scene_manager.stats_manager.save_to_csv(csv_file=self.stats_csv_filepath)

        scene_list = self.scene_manager.get_scene_list()

        df = pd.read_csv(self.stats_csv_filepath)

        # sort each list just in case threads added frame numbers to lists out of order
        self.start_frame_numbers_list_mutex.acquire()
        try:
            self.start_frame_numbers_list.sort()
        finally:
            self.start_frame_numbers_list_mutex.release()

        last_frame = df[stats_manager.COLUMN_NAME_FRAME_NUMBER].iloc[-1]

        self.end_frame_numbers_list_mutex.acquire()
        try:
            self.end_frame_numbers_list.append(last_frame)
            self.end_frame_numbers_list.sort()
        finally:
            self.end_frame_numbers_list_mutex.release()

        self.scene_data.clear()

        for i, scene in enumerate(scene_list):
            self.scene_data.append((self.start_frame_numbers_list[i], scene[0], self.end_frame_numbers_list[i], scene[1]))
        
        
if __name__ == "__main__":
    video_path = os.path.join(".", "dataset", "videos_with_ads", file_name)
    
    # test code for ImageSceneDetetctor class
    image_scene_detector = ImageSceneDetector(video_path, detector_override)
    image_scene_detector.plot_scene_stats()

    sys.exit(0)
    