import os
import cv2
from scenedetect import detect, ContentDetector, split_video_ffmpeg, AdaptiveDetector, HistogramDetector, FrameTimecode, HashDetector
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

video_path = os.path.join(".", "dataset", "videos_with_ads", "test_003.mp4")

# 1. Detect scenes using the high-level detect function
# Threshold 27.0 is a common starting point for content-aware detection
scene_list = detect(video_path, ContentDetector(threshold=67, luma_only=True), stats_file_path="stats.csv", show_progress=True)

# 2. Print the list of scenes (start and end timecodes)
for i, scene in enumerate(scene_list):
    print(f'Scene {i}: Start {scene[0]}, End {scene[1]}')

# 3. Optional: Split the video into individual clips (requires FFmpeg)
split_video_ffmpeg(video_path, scene_list)

df = pd.read_csv('stats.csv')
df.plot(x='Frame Number', y='content_val')
plt.show()

#with open('stats.txt') as f:
#    lines = f.readlines()
#    x = [line.split(',')[0] for line in lines]
#    y = [line.split(',')[2] for line in lines]
#    print(x)
#    print(y)

#    plt.xlabel('Frame Number')
#    plt.ylabel('content_val')
#    plt.title('Scene Change Plot')
#    plt.plot(x[1:20], y[1:20])
    
#    plt.show()

#data = np.loadtxt(fname='stats.txt', delimiter=',')
#plt.plot(data[:,0], data[:, 1])
#plt.show()


#video = cv2.VideoCapture(video_path)

#ret = True
#while ret:
#    ret, frame = video.read()

#    if ret:
#        cv2.imshow('thing', frame)
#        cv2.waitKey(40)

#video.release()
#cv2.destroyAllWindows()