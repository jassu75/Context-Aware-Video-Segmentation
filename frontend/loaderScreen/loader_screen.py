import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import QPainter, QColor, QPen


class SpinnerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(64, 64)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def start(self):
        self._timer.start(16)

    def stop(self):
        self._timer.stop()

    def _tick(self):
        self._angle = (self._angle + 6) % 360
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(6, 6, 52, 52)

        pen = QPen(QColor("#2a2d33"), 5)
        p.setPen(pen)
        p.drawArc(rect, 0, 360 * 16)

        pen.setColor(QColor("#4a9eff"))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawArc(rect, -self._angle * 16, 100 * 16)


class LoaderScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._load_styles()

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)

        self.spinner = SpinnerWidget()
        layout.addWidget(self.spinner, alignment=Qt.AlignmentFlag.AlignCenter)

        self.label = QLabel("Analyzing video…")
        self.label.setObjectName("loaderLabel")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)

        self.sub = QLabel("This may take a moment")
        self.sub.setObjectName("loaderSub")
        self.sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.sub)

    def _load_styles(self):
        try:
            path = os.path.join(os.path.dirname(__file__), "loader_screen.qss")
            with open(path, "r") as f:
                self.setStyleSheet(f.read())
        except Exception as e:
            print("Failed to load loader stylesheet:", e)

    def start(self):
        self.spinner.start()

    def stop(self):
        self.spinner.stop()