import os
from PyQt6.QtWidgets import (
    QWidget, QPushButton,
    QVBoxLayout, QFileDialog
)
from PyQt6.QtCore import Qt


class UploadScreen(QWidget):
    def __init__(self, handle_video_select):
        super().__init__()

        self.handle_video_select = handle_video_select
        self.load_styles()
        self.setup_upload_screen()

    def setup_upload_screen(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.button = QPushButton("Select Video")
        self.button.setObjectName("selectButton")
        self.button.setFixedSize(220, 70)
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button.clicked.connect(self.open_file_dialog)
        

        layout.addWidget(self.button, alignment=Qt.AlignmentFlag.AlignCenter)
        self.setLayout(layout)

    def load_styles(self):
        try:
            path = os.path.join(os.path.dirname(__file__), "upload_screen.qss")
            with open(path, "r") as f:
                self.setStyleSheet(f.read())
        except Exception as e:
            print("Failed to load upload_screen stylesheet:", e)

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)"
        )

        if file_path:
            print("Selected file:", file_path)
            self.handle_video_select(file_path)