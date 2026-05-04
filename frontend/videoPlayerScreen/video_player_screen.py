import sys
import os
import json

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QFrame, QScrollArea, QSizePolicy,
    QFileDialog, QSlider, QMessageBox
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal, QSize
from PyQt6.QtGui import QPainter, QColor, QFont, QIcon


# ── Helpers ─────────────────────────────────────────────────────────────────────

def fmt_time(secs: float) -> str:
    secs = max(0, int(secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def build_segments(non_content: list, total_s: float) -> list:
    nc = sorted(non_content, key=lambda x: x["start_seconds"])
    result, cursor = [], 0.0
    for seg in nc:
        s, e = float(seg["start_seconds"]), float(seg["end_seconds"])
        if s > cursor:
            result.append({"start_seconds": cursor, "end_seconds": s,
                            "content_type": "Content", "is_content": True})
        result.append({"start_seconds": s, "end_seconds": e,
                        "content_type": seg["content_type"], "is_content": False})
        cursor = e
    if cursor < total_s:
        result.append({"start_seconds": cursor, "end_seconds": total_s,
                       "content_type": "Content", "is_content": True})
    return result


# ── Clickable Timeline ──────────────────────────────────────────────────────────

class TimelineBar(QWidget):
    seeked = pyqtSignal(float)

    C_CONTENT    = QColor("#3a9e4e")
    C_NONCONTENT = QColor("#c0392b")
    C_BG         = QColor("#0d0f13")
    C_HEAD       = QColor("#ffffff")
    C_HEAD_RING  = QColor("#eaecf2")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.segments: list  = []
        self.total: float    = 1.0
        self.position: float = 0.0
        self.setFixedHeight(32)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def load(self, segments, total):
        self.segments = segments
        self.total    = max(total, 1.0)
        self.update()

    def set_position(self, frac: float):
        self.position = max(0.0, min(1.0, frac))
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H  = self.width(), self.height()
        BAR_H = 10
        y     = (H - BAR_H) // 2

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self.C_BG)
        p.drawRoundedRect(0, y, W, BAR_H, BAR_H // 2, BAR_H // 2)

        for seg in self.segments:
            x1 = int(seg["start_seconds"] / self.total * W)
            x2 = int(seg["end_seconds"]   / self.total * W)
            p.setBrush(self.C_CONTENT if seg["is_content"] else self.C_NONCONTENT)
            p.drawRect(x1, y, max(x2 - x1, 1), BAR_H)

        px = int(self.position * W)
        p.setBrush(self.C_HEAD)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(px - 7, H // 2 - 7, 14, 14)

    def _emit(self, x):
        self.seeked.emit(max(0.0, min(1.0, x / self.width())))

    def mousePressEvent(self, e):
        self._emit(e.position().x())

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            self._emit(e.position().x())


# ── Segment Card ────────────────────────────────────────────────────────────────

class SegmentCard(QFrame):
    action_clicked = pyqtSignal(dict)

    def __init__(self, seg: dict, parent=None):
        super().__init__(parent)
        self.seg = seg
        self.setObjectName("segmentCard")
        self.setProperty("active",  "false")
        self.setProperty("content", "true" if seg["is_content"] else "false")
        self._build()

    def _build(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 7, 10, 7)
        lay.setSpacing(8)

        left = QVBoxLayout()
        left.setSpacing(3)

        self.name_lbl = QLabel(self.seg["content_type"])
        self.name_lbl.setObjectName("cardName")

        tag_txt  = "Content" if self.seg["is_content"] else "Non-Content"
        tag_type = "content" if self.seg["is_content"] else "noncontent"
        self.tag = QLabel(tag_txt)
        self.tag.setObjectName("cardTag")
        self.tag.setProperty("type", tag_type)
        self.tag.setAlignment(Qt.AlignmentFlag.AlignCenter)

        left.addWidget(self.name_lbl)
        left.addWidget(self.tag)

        right = QVBoxLayout()
        right.setSpacing(3)
        right.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.time_lbl = QLabel(
            f"{fmt_time(self.seg['start_seconds'])} – {fmt_time(self.seg['end_seconds'])}"
        )
        self.time_lbl.setObjectName("cardTime")
        self.time_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)

        btn_txt  = "▶  Play" if self.seg["is_content"] else "▶  Skip"
        btn_type = "content" if self.seg["is_content"] else "noncontent"
        self.btn = QPushButton(btn_txt)
        self.btn.setObjectName("cardBtn")
        self.btn.setProperty("type", btn_type)
        self.btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn.clicked.connect(lambda: self.action_clicked.emit(self.seg))

        right.addWidget(self.time_lbl)
        right.addWidget(self.btn)

        lay.addLayout(left)
        lay.addStretch()
        lay.addLayout(right)

    def set_active(self, active: bool):
        self.setProperty("active", "true" if active else "false")
        self.style().unpolish(self)
        self.style().polish(self)


# ── Main Window ─────────────────────────────────────────────────────────────────

class PlayerWindow(QMainWindow):

    def __init__(self, video_path: str = "", json_path: str = ""):
        super().__init__()
        self.setWindowTitle("Segmentation Player  —  CSCI 576")
        self.setMinimumSize(1020, 660)

        self.segments:     list  = []
        self.total_s:      float = 0.0
        self.content_only: bool  = False
        self.skip_nc:      bool  = False
        self._scrubbing:   bool  = False

        self._build_ui()
        self._setup_player()

        if video_path and json_path:
            self._load(video_path, json_path)
        else:
            self._prompt_open()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        # Header
        hdr = QHBoxLayout()
        self.title_lbl = QLabel("No file loaded")
        self.title_lbl.setObjectName("titleLabel")
        self.open_btn = QPushButton("📂  Open…")
        self.open_btn.setObjectName("openBtn")
        self.open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_btn.clicked.connect(self._prompt_open)
        hdr.addWidget(self.title_lbl)
        hdr.addStretch()
        hdr.addWidget(self.open_btn)
        root.addLayout(hdr)

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addLayout(self._build_left(), stretch=1)
        body.addLayout(self._build_right())        
        root.addLayout(body)

    def _build_left(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(8)

        # Video
        self.video_widget = QVideoWidget()
        self.video_widget.setObjectName("videoWidget")
        self.video_widget.setMinimumHeight(300)
        self.video_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        col.addWidget(self.video_widget)

        # Now-playing badge
        self.now_lbl = QLabel("Load a video to begin")
        self.now_lbl.setObjectName("nowLabel")
        self.now_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        col.addWidget(self.now_lbl)

        # Transport
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)

        self.play_btn = QPushButton("▶")
        self.play_btn.setObjectName("transportBtn")
        self.play_btn.setFixedSize(40, 40)
        self.play_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.play_btn.clicked.connect(self._toggle_play)

        self.prev_btn = QPushButton("⏮")
        self.prev_btn.setObjectName("transportBtn")
        self.prev_btn.setFixedSize(40, 40)
        self.prev_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.prev_btn.clicked.connect(self._prev_segment)

        self.next_btn = QPushButton("⏭")
        self.next_btn.setObjectName("transportBtn")
        self.next_btn.setFixedSize(40, 40)
        self.next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.next_btn.clicked.connect(self._next_segment)

        self.time_lbl = QLabel("00:00 / 00:00")
        self.time_lbl.setObjectName("timeLabel")

        # Volume
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setObjectName("volSlider")
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(80)
        self.vol_slider.setFixedWidth(90)
        self.vol_slider.valueChanged.connect(
            lambda v: self.audio_out.setVolume(v / 100.0)
        )
        vol_lbl = QLabel("🔊")
        vol_lbl.setObjectName("volIcon")

        ctrl.addWidget(self.prev_btn)
        ctrl.addWidget(self.play_btn)
        ctrl.addWidget(self.next_btn)
        ctrl.addStretch()
        ctrl.addWidget(self.time_lbl)
        ctrl.addStretch()
        ctrl.addWidget(vol_lbl)
        ctrl.addWidget(self.vol_slider)
        col.addLayout(ctrl)

        # Timeline
        self.timeline = TimelineBar()
        self.timeline.seeked.connect(self._on_seek)
        col.addWidget(self.timeline)

        # Tick labels
        self.tick_container = QWidget()
        self.tick_layout    = QHBoxLayout(self.tick_container)
        self.tick_layout.setSpacing(0)
        self.tick_layout.setContentsMargins(0, 0, 0, 0)
        col.addWidget(self.tick_container)

        # Action buttons
        bot = QHBoxLayout()
        bot.setSpacing(12)

        self.play_content_btn = QPushButton("▶  Play Content Only")
        self.play_content_btn.setObjectName("playContentBtn")
        self.play_content_btn.setProperty("active", "false")
        self.play_content_btn.setFixedHeight(44)
        self.play_content_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.play_content_btn.clicked.connect(self._toggle_content_only)

        self.skip_nc_btn = QPushButton("▶  Skip Non-Content")
        self.skip_nc_btn.setObjectName("skipNcBtn")
        self.skip_nc_btn.setProperty("active", "false")
        self.skip_nc_btn.setFixedHeight(44)
        self.skip_nc_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.skip_nc_btn.clicked.connect(self._toggle_skip_nc)

        bot.addWidget(self.play_content_btn)
        bot.addWidget(self.skip_nc_btn)
        col.addLayout(bot)

        return col

    def _build_right(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(6)

        heading = QLabel("Segment Overview")
        heading.setObjectName("overviewHeading")
        col.addWidget(heading)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setObjectName("segmentScroll")
        scroll.setFixedWidth(280) 

        self.cards_container = QWidget()
        self.cards_layout    = QVBoxLayout(self.cards_container)
        self.cards_layout.setSpacing(6)
        self.cards_layout.setContentsMargins(0, 0, 4, 0)
        self.cards_layout.addStretch()

        scroll.setWidget(self.cards_container)
        col.addWidget(scroll)
        return col

    # ── Media player setup ───────────────────────────────────────────────────

    def _setup_player(self):
        self.player    = QMediaPlayer()
        self.audio_out = QAudioOutput()
        self.audio_out.setVolume(0.8)
        self.player.setAudioOutput(self.audio_out)
        self.player.setVideoOutput(self.video_widget)

        self.player.durationChanged.connect(self._on_duration_changed)
        self.player.positionChanged.connect(self._on_position_changed)
        self.player.playbackStateChanged.connect(self._on_state_changed)

        # Fallback poll timer (position updates can be infrequent)
        self._poll = QTimer(self)
        self._poll.setInterval(300)
        self._poll.timeout.connect(self._poll_position)
        self._poll.start()

    # ── Load ─────────────────────────────────────────────────────────────────

    def _prompt_open(self):
        vid, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video Files (*.mp4 *.mkv *.avi *.mov *.webm *.m4v);;All Files (*)"
        )
        if not vid:
            return
        jsn, _ = QFileDialog.getOpenFileName(
            self, "Open Segments JSON", os.path.dirname(vid),
            "JSON Files (*.json);;All Files (*)"
        )
        if not jsn:
            return
        self._load(vid, jsn)

    def _load(self, video_path: str, json_path: str):
        try:
            with open(json_path, "r") as f:
                raw = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "JSON Error", str(e))
            return

        self.player.setSource(QUrl.fromLocalFile(os.path.abspath(video_path)))
        self._pending_json = raw
        self._video_name   = os.path.basename(video_path)
        self.title_lbl.setText(f"<b>{self._video_name}</b>  —  waiting for duration…")

    def _on_duration_changed(self, ms: int):
        if ms <= 0:
            return
        self.total_s = ms / 1000.0
        raw = getattr(self, "_pending_json", [])
        self.segments = build_segments(raw, self.total_s)
        self.cards: list[SegmentCard] = []

        # Rebuild tick labels
        while self.tick_layout.count():
            item = self.tick_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for seg in self.segments:
            span = seg["end_seconds"] - seg["start_seconds"]
            lbl  = QLabel(seg["content_type"])
            lbl.setObjectName("segTickLabel")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            self.tick_layout.addWidget(lbl, stretch=max(int(span / self.total_s * 1000), 1))

        while self.cards_layout.count() > 1:
            item = self.cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for seg in self.segments:
            card = SegmentCard(seg)
            card.action_clicked.connect(self._on_card_action)
            self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)
            self.cards.append(card)

        self.timeline.load(self.segments, self.total_s)
        self.title_lbl.setText(
            f"<b>{self._video_name}</b>  ·  {fmt_time(self.total_s)}"
        )

    # ── Playback control ─────────────────────────────────────────────────────

    def _toggle_play(self):
        state = self.player.playbackState()
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _on_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_btn.setText("⏸")
        else:
            self.play_btn.setText("▶")

    def _prev_segment(self):
        if not self.segments:
            return
        pos_s = self.player.position() / 1000.0
        cur   = self._seg_at(pos_s)
        idx   = self.segments.index(cur)
        self._seek_s(self.segments[max(idx - 1, 0)]["start_seconds"])

    def _next_segment(self):
        if not self.segments:
            return
        pos_s = self.player.position() / 1000.0
        cur   = self._seg_at(pos_s)
        idx   = self.segments.index(cur)
        if idx + 1 < len(self.segments):
            self._seek_s(self.segments[idx + 1]["start_seconds"])

    def _on_seek(self, frac: float):
        if self.total_s > 0:
            self._seek_s(frac * self.total_s)

    def _seek_s(self, secs: float):
        self.player.setPosition(int(secs * 1000))

    def _on_card_action(self, seg: dict):
        self._seek_s(seg["start_seconds"])
        if self.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            self.player.play()

    def _toggle_content_only(self):
        self.content_only = not self.content_only
        self.skip_nc      = self.content_only

        if self.content_only:
            self.play_content_btn.setText("✓  Content Only")
            self.play_content_btn.setProperty("active", "true")
            if self.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
                self.player.play()
        else:
            self.play_content_btn.setText("▶  Play Content Only")
            self.play_content_btn.setProperty("active", "false")

        self._repolish(self.play_content_btn)

    def _toggle_skip_nc(self):
        self.skip_nc = not self.skip_nc
        if not self.skip_nc and self.content_only:
            self.content_only = False
            self.play_content_btn.setText("▶  Play Content Only")
            self.play_content_btn.setProperty("active", "false")
            self._repolish(self.play_content_btn)
        self.skip_nc_btn.setProperty("active", "true" if self.skip_nc else "false")
        self._repolish(self.skip_nc_btn)

    # ── Position tracking ────────────────────────────────────────────────────

    def _poll_position(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._on_position_changed(self.player.position())

    def _on_position_changed(self, ms: int):
        if self.total_s <= 0 or not self.segments:
            return
        pos_s = ms / 1000.0

        if self.skip_nc and self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            cur = self._seg_at(pos_s)
            if not cur["is_content"]:
                next_content = next(
                    (s for s in self.segments
                     if s["start_seconds"] >= cur["end_seconds"] and s["is_content"]),
                    None
                )
                if next_content:
                    self._seek_s(next_content["start_seconds"])
                    return
                else:
                    self.player.pause()
                    return

        self.timeline.set_position(pos_s / self.total_s)
        self.time_lbl.setText(f"{fmt_time(pos_s)} / {fmt_time(self.total_s)}")

        cur = self._seg_at(pos_s)
        self.now_lbl.setText(
            f"▶  {cur['content_type']}  ·  "
            f"{fmt_time(cur['start_seconds'])} – {fmt_time(cur['end_seconds'])}"
        )
        for seg, card in zip(self.segments, self.cards):
            card.set_active(seg is cur)

    def _seg_at(self, pos_s: float) -> dict:
        for seg in self.segments:
            if seg["start_seconds"] <= pos_s < seg["end_seconds"]:
                return seg
        return self.segments[-1] if self.segments else {
            "start_seconds": 0, "end_seconds": 0,
            "content_type": "", "is_content": True
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _repolish(self, w):
        w.style().unpolish(w)
        w.style().polish(w)

    def keyPressEvent(self, e):
        key = e.key()
        if key == Qt.Key.Key_Space:
            self._toggle_play()
        elif key == Qt.Key.Key_Right:
            self._seek_s(self.player.position() / 1000.0 + 10)
        elif key == Qt.Key.Key_Left:
            self._seek_s(max(0, self.player.position() / 1000.0 - 10))
        elif key == Qt.Key.Key_Period:
            self._next_segment()
        elif key == Qt.Key.Key_Comma:
            self._prev_segment()
        else:
            super().keyPressEvent(e)


# ── Entry point ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    qss_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "player.qss")
    if os.path.exists(qss_path):
        with open(qss_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
    else:
        print(f"[warn] player.qss not found — running unstyled")

    args = sys.argv[1:]
    win  = PlayerWindow(
        video_path=args[0] if len(args) > 0 else "",
        json_path =args[1] if len(args) > 1 else "",
    )
    win.show()
    sys.exit(app.exec())