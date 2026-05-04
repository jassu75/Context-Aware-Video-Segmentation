import sys
import os
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout

from frontend.uploadScreen.upload_screen import UploadScreen
from backend.pipeline import analyze_video


class VideoAnalyzer(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Analyzer")
        self.showMaximized()

        self.load_styles()

        self.setup_upload_screen()

    def setup_upload_screen(self):
       
        layout = QVBoxLayout()

        self.upload_screen = UploadScreen(self.handle_video_selected)
        layout.addWidget(self.upload_screen)

        self.setLayout(layout)

    def handle_video_selected(self, file_path):
        non_content_list, non_content_json_path = analyze_video(file_path)
        print(f'Non-Content Data from video analyzer: {non_content_list}')
        print(f'Non-Content Segment Data available at {non_content_json_path}')

    def load_styles(self):
        try:
            path = os.path.join(os.path.dirname(__file__), "main.qss")
            with open(path, "r") as f:
                self.setStyleSheet(f.read())
        except Exception as e:
            print("Failed to load main stylesheet:", e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoAnalyzer()
    window.show()
    sys.exit(app.exec())