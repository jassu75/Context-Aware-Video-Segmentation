import sys
import os
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QStackedWidget
from PyQt6.QtCore import QThread, pyqtSignal

from frontend.uploadScreen.upload_screen import UploadScreen
from frontend.loaderScreen.loader_screen import LoaderScreen
from frontend.videoPlayerScreen.video_player_screen import PlayerWindow
from backend.pipeline import analyze_video


class AnalysisWorker(QThread):
    finished = pyqtSignal(str, str)
    error    = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            _, json_path = analyze_video(self.video_path)
            self.finished.emit(self.video_path, str(json_path))
        except Exception as e:
            self.error.emit(str(e))


class VideoAnalyzer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Analyzer")
        self.showMaximized()
        self.load_styles()

        self.stack   = QStackedWidget()
        self.upload  = UploadScreen(self.handle_video_selected)
        self.loader  = LoaderScreen()
        self.stack.addWidget(self.upload)   
        self.stack.addWidget(self.loader)   

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.stack)

        self.worker = None
        self.player = None

    def handle_video_selected(self, file_path):
        self.stack.setCurrentIndex(1)          
        self.loader.start()

        self.worker = AnalysisWorker(file_path)
        self.worker.finished.connect(self.on_analysis_done)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()

    def on_analysis_done(self, video_path, json_path):
        self.loader.stop()
        self.player = PlayerWindow(video_path, json_path)
        self.player.show()
        self.close()

    def on_analysis_error(self, msg):
        from PyQt6.QtWidgets import QMessageBox
        self.loader.stop()
        self.stack.setCurrentIndex(0)
        QMessageBox.critical(self, "Analysis Error", msg)

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