# Multimodal Video Segmentation

An intelligent multimodal video analysis system that automatically segments long-form videos into meaningful categories such as core content, advertisements, sponsorships, intros, outros, and silent sections using computer vision, audio analysis, motion detection, and speech understanding.

## Preview

<img width="1448" height="1086" alt="Application Preview" src="https://github.com/user-attachments/assets/02b67b42-0d90-4905-9b29-39669a54ba66" />

## Description

- Architected an end-to-end multimodal system that classifies long-form video into core versus non-core segments by combining audio frequency analysis, visual frame structure, motion detection, and speech transcripts.
- Built a custom synchronized video player with an interactive timeline, color-coded segment mapping, and segment-level navigation controls including play, skip, and fast-forward functionality.
- Enabled users to selectively consume important content while bypassing advertisements, sponsorships, intros, and dead-air sections.

## Features

- 🎬 Automatic video segmentation
- 🧠 Multimodal analysis pipeline
- 🗣️ Speech-to-text transcription using Whisper
- 📊 Interactive timeline visualization
- 🎨 Color-coded segment mapping
- ⏯️ Segment-level playback controls
- ⚡ Fast video processing with OpenCV and FFmpeg
- 🖥️ Desktop application built with PyQt6

## Tech Stack

### Core Libraries

- Python
- OpenCV
- PySceneDetect
- OpenAI Whisper
- Librosa
- PyQt6
- FFmpeg
- NumPy

## Installation

### Clone the Repository

```bash
git clone https://github.com/jassu75/Context-Aware-Video-Segmentation.git
```

### Create Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python main.py
```

## Processing Pipeline

1. Load video input
2. Detect scene transitions
3. Analyze motion and frame structure
4. Extract audio features
5. Generate speech transcripts using Whisper
6. Fuse multimodal signals for segmentation
7. Display segmented output in synchronized player

## License

MIT License

## Authors

Developed by Tejas Kangod, Jesus Ramos, and Michael Maher
