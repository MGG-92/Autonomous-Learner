"""Native Qt desktop entry point with isolated local neural/professor computation."""
import math
from pathlib import Path
import sys

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap, QPolygonF, QRadialGradient
from PySide6.QtWidgets import (QApplication, QCheckBox, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSlider,
    QVBoxLayout, QWidget, QSplitter)

from embeddings import EmbeddingStore
from compute import Compute
from chat import Chat


STYLE = """
QMainWindow,QWidget{background:#0a1016;color:#dfebef;font-family:'Segoe UI';font-size:12px}
QLabel{background:transparent} QFrame#header{border-bottom:1px solid #24323c}
QFrame#workspace{background:#101820;border-top:1px solid #2a3945}
QFrame#inspector{background:#111d27;border:1px solid #2c4250;border-radius:12px}
QFrame#transport{background:#15232e;border:1px solid #304956;border-radius:10px}
QPushButton{background:#15232d;border:1px solid #304653;border-radius:7px;padding:8px 13px;color:#a6bac4}
QPushButton:hover{background:#203642;border-color:#597d88;color:#efffff}
QPushButton:pressed{background:#294a50} QPushButton:disabled{color:#4a606d;border-color:#233541}
QPushButton:checked{background:#203e40;border-color:#7bbaad;color:#aaf4df}
QPushButton#primary{background:#98e6cf;color:#102e28;border:0;font-weight:600;font-size:13px;padding:14px 22px}
QPushButton#primary:hover{background:#bbf3e3} QPushButton#primary:disabled{background:#375b53;color:#8ba59d}
QPushButton#quiet{border:0;background:transparent;color:#8ba4b1;padding:5px 9px}
QPushButton#quiet:hover{color:#c7efdf;background:#172b35}
QPlainTextEdit{background:#111e28;border:1px solid #344a58;border-radius:9px;padding:10px;color:#e2ecef;font-size:14px;selection-background-color:#365d60}
QPlainTextEdit:focus{border:1px solid #8bcfbe}
QScrollArea{border:0;background:transparent} QScrollArea>QWidget>QWidget{background:transparent}
QScrollBar:horizontal{height:5px;background:#101820} QScrollBar::handle:horizontal{background:#3a535f;border-radius:2px}
QScrollBar::add-line:horizontal,QScrollBar::sub-line:horizontal{width:0}
QScrollBar:vertical{width:7px;background:#111d27;margin:5px 0}
QScrollBar::handle:vertical{background:#3a535f;min-height:24px;border-radius:3px}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0}
QScrollBar::add-page:vertical,QScrollBar::sub-page:vertical{background:transparent}
QSlider::groove:horizontal{height:3px;background:#3b515d;border-radius:1px}
QSlider::handle:horizontal{width:10px;height:10px;margin:-4px 0;background:#9ce4d1;border-radius:5px}
QCheckBox{background:transparent;color:#9bb3bf;spacing:8px}
QCheckBox::indicator{width:13px;height:13px;border:1px solid #4b6776;border-radius:3px;background:#13222c}
QCheckBox::indicator:checked{background:#7bbba9;border-color:#a9e5d4}
QToolTip{background:#172c36;color:#e2f1ef;border:1px solid #486774;padding:6px}
QSpinBox,QDoubleSpinBox{background:#15232d;border:1px solid #304653;border-radius:6px;padding:6px;color:#b9d5d9}
QSplitter::handle{background:#24323c;width:1px}
"""


def label(text, size=11, color="#8fa8b5", bold=False):
    item = QLabel(text)
    font = QFont("Segoe UI", size)
    font.setWeight(QFont.Weight.DemiBold if bold else QFont.Weight.Normal)
    item.setFont(font)
    item.setStyleSheet(f"color:{color};background:transparent;font-family:'Segoe UI';font-size:{round(size * 1.33)}px;font-weight:{600 if bold else 400}")
    return item


def button(text, action, name=None):
    item = QPushButton(text)
    if name:
        item.setObjectName(name)
    item.clicked.connect(action)
    item.setCursor(Qt.CursorShape.PointingHandCursor)
    return item


class ModelLoader(QThread):
    ready = Signal(object)
    failed = Signal(str)

    def __init__(self, store):
        super().__init__()
        self.store = store

    def run(self):
        try:
            self.ready.emit(self.store.metadata())
        except Exception:
            self.failed.emit("Local embeddings unavailable. Run prepare_model.py using the README setup instructions.")


class Volume(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(330)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.tokens, self.index, self.all_tokens = [], 0, False
        self.response, self.guides = False, True
        self.yaw, self.pitch, self.zoom, self.fit_zoom = .55, -.3, 1., 1.
        self.drag_position = None
        self.neural = False
        self.reserve_inspector = False
        self.activity = []
        self.input_activity = []
        self.output_activity = []
        self.show_embeddings = False
        self.neuron_positions = np.stack(np.meshgrid(*([np.linspace(-1, 1, 25)] * 3), indexing="ij"), -1).reshape(-1, 3)
        yz = np.stack(np.meshgrid(np.linspace(-.9, .9, 16), np.linspace(-.9, .9, 32), indexing="ij"), -1).reshape(-1, 2)
        self.input_positions = np.column_stack((np.full(512, -1.45), yz))
        self.output_positions = np.column_stack((np.full(512, 1.45), yz))

    def reset_camera(self):
        self.yaw, self.pitch, self.zoom = .55, -.3, self.fit_zoom
        self.update()

    def project(self, points):
        xyz = np.asarray(points, dtype=float).reshape(-1, 3)
        cy, sy, cp, sp = math.cos(self.yaw), math.sin(self.yaw), math.cos(self.pitch), math.sin(self.pitch)
        x = xyz[:, 0] * cy + xyz[:, 2] * sy
        z = -xyz[:, 0] * sy + xyz[:, 2] * cy
        y = xyz[:, 1] * cp - z * sp
        depth = xyz[:, 1] * sp + z * cp
        factor = min(self.width(), self.height()) * .34 * self.zoom * 4 / np.maximum(.2, 4 - depth)
        center = (self.width() - (250 if self.tokens and self.reserve_inspector else 0)) / 2
        return [QPointF(center + xx * f, self.height() * .47 - yy * f) for xx, yy, f in zip(x, y, factor)]

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        gradient = QRadialGradient(self.width() * .43, self.height() * .48, self.width() * .7)
        gradient.setColorAt(0, QColor("#132632"))
        gradient.setColorAt(.65, QColor("#0b141d"))
        gradient.setColorAt(1, QColor("#080e15"))
        painter.fillRect(self.rect(), gradient)
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor("#839fae"))
        painter.drawText(28, 30, "●   INPUT 512   →   RECURRENT CORE 15,625   →   OUTPUT 512" if self.neural else "●   NEURAL VOLUME LAB")
        if self.neural:
            painter.setPen(QPen(QColor(105, 150, 168, 26), 1.2))
            painter.drawPoints(QPolygonF(self.project(self.neuron_positions)))
            if self.activity:
                ids = [pair[0] for pair in self.activity]
                points = self.project(self.neuron_positions[ids])
                for point, (_, value) in zip(points, self.activity):
                    color = QColor("#9ae7d2")
                    color.setAlpha(min(240, 50 + round(value * 180)))
                    painter.setPen(QPen(color, min(6., 2 + value * 3), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                    painter.drawPoint(point)
            for positions, values, tint, name in ((self.input_positions, self.input_activity, "#e8ba86", "INPUT"),
                                                   (self.output_positions, self.output_activity, "#a8c9fa", "OUTPUT")):
                points = self.project(positions)
                color = QColor(tint)
                color.setAlpha(45)
                painter.setPen(QPen(color, 2.))
                painter.drawPoints(QPolygonF(points))
                for point, value in zip(points, values):
                    if value > 1e-5:
                        color.setAlpha(min(245, 65 + round(value * 90)))
                        painter.setPen(QPen(color, min(5., 2 + value), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                        painter.drawPoint(point)
                painter.setPen(QColor(tint))
                anchor = positions[0].copy()
                anchor[1] = 1.05
                painter.drawText(self.project([anchor])[0], name + " / 512")
        if self.guides:
            radius = min(1, 1 / self.zoom)
            corners = [(x * radius, y * radius, z * radius) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
            painter.setPen(QPen(QColor(103, 153, 175, 28), 1))
            for i, a in enumerate(corners):
                for b in corners[i + 1:]:
                    if sum(x != y for x, y in zip(a, b)) == 1:
                        painter.drawLine(*self.project([a, b]))
            for axis in range(3):
                a, b = [0., 0., 0.], [0., 0., 0.]
                a[axis], b[axis] = -radius, radius
                pa, pb = self.project([a, b])
                painter.setPen(QPen(QColor(103, 153, 175, 42), 1))
                painter.drawLine(pa, pb)
                painter.setPen(QColor("#7b97a6"))
                painter.drawText(pb + QPointF(7, -7), "XYZ"[axis])
        if not self.tokens and not self.neural:
            painter.setPen(QColor("#8bb9b1"))
            painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(QRectF(0, self.height() * .35, self.width(), 30), Qt.AlignmentFlag.AlignCenter, "YOUR LEARNING NETWORK")
            painter.setPen(QColor("#e0ecf1"))
            painter.setFont(QFont("Segoe UI", 30, QFont.Weight.Light))
            painter.drawText(QRectF(0, self.height() * .43, self.width(), 55), Qt.AlignmentFlag.AlignCenter, "Start the brain.")
            painter.setFont(QFont("Segoe UI", 11))
            painter.setPen(QColor("#8fa7b4"))
            painter.drawText(QRectF(0, self.height() * .43 + 65, self.width(), 30), Qt.AlignmentFlag.AlignCenter,
                             "Send a message. Reward replies. Teach corrections.")
        elif self.tokens and self.show_embeddings:
            indices = [i for i in range(len(self.tokens)) if i != self.index] if self.all_tokens else []
            indices.append(self.index)
            color = QColor("#f2bd95" if self.response else "#9ae7d2")
            for index in indices:
                selected = index == self.index
                points = self.project(self.tokens[index]["points"])
                color.setAlpha(190 if selected else 28)
                painter.setPen(QPen(color, 1.25 if selected else .8))
                painter.drawPolyline(QPolygonF(points))
                if selected:
                    color.setAlpha(220)
                    painter.setPen(QPen(color, 2.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                    painter.drawPoints(QPolygonF(points))
                    painter.setFont(QFont("Consolas", 8))
                    painter.setBrush(color)
                    for point, name in ((points[0], "START"), (points[-1], "END")):
                        painter.drawEllipse(point, 3, 3)
                        painter.drawText(point + QPointF(8, -8), name)
        painter.setPen(QColor("#63818f"))
        painter.setFont(QFont("Segoe UI", 8))
        hint = "DRAG TO ORBIT    ·    SCROLL TO ZOOM"
        if self.width() > 1200:
            hint += "    ·    ARROWS TO ROTATE"
        painter.drawText(28, self.height() - 25, hint)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.drag_position is not None:
            delta = event.position() - self.drag_position
            self.yaw += delta.x() * .006
            self.pitch = max(-1.45, min(1.45, self.pitch + delta.y() * .006))
            self.drag_position = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        self.drag_position = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def wheelEvent(self, event):
        self.zoom = max(.25, min(100, self.zoom * math.exp(event.angleDelta().y() * .001)))
        self.update()
        event.accept()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Left: self.yaw -= .12
        elif key == Qt.Key.Key_Right: self.yaw += .12
        elif key == Qt.Key.Key_Up: self.pitch = max(-1.45, self.pitch - .12)
        elif key == Qt.Key.Key_Down: self.pitch = min(1.45, self.pitch + .12)
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal): self.zoom = min(100, self.zoom * 1.15)
        elif key == Qt.Key.Key_Minus: self.zoom = max(.25, self.zoom / 1.15)
        elif key == Qt.Key.Key_0: self.reset_camera()
        else: return super().keyPressEvent(event)
        self.update()


class Window(QMainWindow):
    def __init__(self, store=None, checkpoint=None):
        super().__init__()
        self.setWindowTitle("Neural Volume Lab · Learned I/O")
        self.setMinimumSize(1120, 760)
        self.resize(1320, 920)
        self.store = store or EmbeddingStore()
        self.data, self.sequence, self.index = None, "message", 0
        self.metadata = None
        self.compute = Compute(self, checkpoint=checkpoint)
        self.compute.result.connect(self.compute_result)
        self.compute.failed.connect(self.compute_failed)
        self.training_batches = 0
        self.network_ready = False
        self.expert_mode = False
        self.brain_running = False
        self.start_requested = False
        self.coached_message_id = None
        self.brain_timer = QTimer(self)
        self.brain_timer.timeout.connect(self.brain_tick)
        self.neural_timer = QTimer(self)
        self.neural_timer.timeout.connect(lambda: self.send_compute("step"))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance)
        self.root = QWidget()
        self.setCentralWidget(self.root)
        layout = QVBoxLayout(self.root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QFrame(objectName="header")
        head = QHBoxLayout(header)
        head.setContentsMargins(28, 20, 28, 20)
        head.addWidget(label("nv", 27, "#98e6cf"))
        titles = QVBoxLayout()
        titles.setSpacing(4)
        titles.addWidget(label("N E U R A L   V O L U M E   L A B", 10, "#d9e7ea", True))
        titles.addWidget(label("NEURAL MESSAGING / 04", 8, "#7995a5"))
        head.addLayout(titles)
        head.addStretch()
        self.mode_button = button("Inspect network", lambda: self.set_expert_mode(not self.expert_mode))
        head.addWidget(self.mode_button)
        self.status = label("●  LOADING LOCAL EMBEDDINGS", 9, "#c2ad85")
        head.addWidget(self.status)
        layout.addWidget(header)
        self.volume = Volume()
        self.volume.installEventFilter(self)
        self.split = QSplitter(Qt.Orientation.Horizontal)
        self.split.addWidget(self.volume)
        self.chat = Chat()
        self.split.addWidget(self.chat)
        self.split.setStretchFactor(0, 3)
        self.split.setStretchFactor(1, 2)
        self.split.setSizes([760, 520])
        self.chat.brain_toggled.connect(self.toggle_brain)
        self.chat.duration_changed.connect(self.set_tick_duration)
        self.chat.action.connect(self.chat_action)
        self.chat.editing.connect(self.chat_editing)
        layout.addWidget(self.split, 1)
        self.build_inspector()
        self.build_transport()
        self.build_workspace(layout)
        self.workspace.hide()
        self.error = label("", 10, "#f0b3a3")
        self.error.setWordWrap(True)
        self.error.setContentsMargins(28, 8, 28, 8)
        self.error.hide()
        layout.addWidget(self.error)
        self.loader = ModelLoader(self.store)
        self.loader.ready.connect(self.model_ready)
        self.loader.failed.connect(self.model_failed)
        self.loader.start()

    def build_inspector(self):
        self.inspector = QScrollArea(self.volume)
        self.inspector.setFixedWidth(230)
        self.inspector.setWidgetResizable(True)
        self.inspector.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.inspector_content = QFrame(objectName="inspector")
        self.inspector.setWidget(self.inspector_content)
        info = QVBoxLayout(self.inspector_content)
        info.setContentsMargins(20, 20, 20, 18)
        info.setSpacing(13)
        self.token_number = label("SELECTED TOKEN", 8)
        self.token_text = label("", 22, "#98e6cf")
        self.token_text.setWordWrap(True)
        self.token_piece = label("", 9)
        self.token_piece.setWordWrap(True)
        self.token_details = label("", 10, "#b2c7d0")
        self.token_details.setTextFormat(Qt.TextFormat.PlainText)
        self.raw = label("", 9, "#b3cbc9")
        self.raw.setFont(QFont("Consolas", 9))
        self.raw.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        for widget in (self.token_number, self.token_text, self.token_piece, self.token_details,
                       label("FIRST RAW COORDINATES", 8), self.raw):
            info.addWidget(widget)
        note = label("XYZ triples are drawings only. Full embeddings enter a learned 512-unit input layer; positions do not choose neurons.", 9, "#7d99a8")
        note.setWordWrap(True)
        info.addWidget(note)
        self.inspector.hide()

    def build_transport(self):
        self.transport = QFrame(self.volume, objectName="transport")
        row = QHBoxLayout(self.transport)
        row.setContentsMargins(8, 7, 8, 7)
        row.setSpacing(8)
        self.previous = button("←", lambda: self.step(-1))
        self.previous.setAccessibleName("Previous token")
        self.play = button("Play", self.toggle_play)
        self.next = button("→", lambda: self.step(1))
        self.next.setAccessibleName("Next token")
        self.position = label("0 / 0", 10)
        self.speed = QSlider(Qt.Orientation.Horizontal)
        self.speed.setRange(1, 12)
        self.speed.setValue(4)
        self.speed.setFixedWidth(75)
        self.speed.setAccessibleName("Playback tokens per second")
        self.speed.valueChanged.connect(self.change_speed)
        self.pace = label("1 token/s", 9)
        self.all = QCheckBox("All tokens")
        self.all.toggled.connect(self.toggle_all)
        for widget in (self.previous, self.play, self.next, self.position, label("Pace", 9), self.speed, self.pace, self.all):
            row.addWidget(widget)
        self.transport.hide()
        self.guides = QCheckBox("Guides", self.volume)
        self.guides.setChecked(True)
        self.guides.toggled.connect(self.toggle_guides)
        self.reset = button("Reset view", self.volume.reset_camera)
        self.reset.setParent(self.volume)

    def build_workspace(self, parent_layout):
        workspace = QFrame(objectName="workspace")
        self.workspace = workspace
        area = QVBoxLayout(workspace)
        area.setContentsMargins(28, 14, 28, 14)
        area.setSpacing(10)
        row = QHBoxLayout()
        self.message_tab = button("Message  0", lambda: self.switch("message"))
        self.response_tab = button("Response reference  0", lambda: self.switch("response"))
        for tab in (self.message_tab, self.response_tab): tab.setCheckable(True)
        self.message_tab.setChecked(True)
        self.response_tab.setEnabled(False)
        row.addWidget(self.message_tab)
        row.addWidget(self.response_tab)
        row.addStretch()
        row.addWidget(button("Clear space", self.clear, "quiet"))
        area.addLayout(row)
        self.strip = QScrollArea()
        self.strip.setWidgetResizable(True)
        self.strip.setFixedHeight(48)
        self.strip.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.strip.viewport().setStyleSheet("background:transparent")
        self.chips_widget = QWidget()
        self.chips_widget.setStyleSheet("background:transparent")
        self.chips_layout = QHBoxLayout(self.chips_widget)
        self.chips_layout.setContentsMargins(0, 3, 0, 3)
        self.chips_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.strip.setWidget(self.chips_widget)
        self.chips = []
        self.placeholder = label("Your tokens will appear here.", 10, "#647f8f")
        self.chips_layout.addWidget(self.placeholder)
        area.addWidget(self.strip)
        composer = QHBoxLayout()
        composer.setSpacing(15)
        self.message = QPlainTextEdit()
        self.message.setPlaceholderText("Write something to give it a shape…")
        self.message.setAccessibleName("Message")
        self.response = QPlainTextEdit()
        self.response.setPlaceholderText("An expected response, for comparison…")
        self.response.setAccessibleName("Expected response, editable training target")
        for title, field in (("MESSAGE", self.message), ("EXPECTED RESPONSE   /   EDIT BEFORE TRAINING", self.response)):
            column = QVBoxLayout()
            column.setSpacing(7)
            column.addWidget(label(title, 8))
            field.setFixedHeight(72)
            column.addWidget(field)
            composer.addLayout(column, 1)
        self.visualize = button("Visualize   ↗", self.visualize_text, "primary")
        self.visualize.setMinimumHeight(72)
        self.visualize.setEnabled(False)
        column = QVBoxLayout()
        column.addSpacing(20)
        column.addWidget(self.visualize)
        composer.addLayout(column)
        area.addLayout(composer)
        self.build_neural_controls(area)
        footer = QHBoxLayout()
        self.model_label = label("Local processing · frozen embeddings · compute starts on request", 8, "#6f8b9b")
        footer.addWidget(self.model_label)
        footer.addStretch()
        self.about_button = button("How this is drawn", self.toggle_about, "quiet")
        footer.addWidget(self.about_button)
        area.addLayout(footer)
        self.about = label("", 9, "#a1b7c1")
        self.about.setWordWrap(True)
        self.about.hide()
        area.addWidget(self.about)
        parent_layout.addWidget(workspace)

    def build_neural_controls(self, area):
        row = QHBoxLayout()
        self.start_network = button("Start network", lambda: self.send_compute("init"))
        self.inject = button("Inject token", self.inject_token)
        self.neural_step = button("Step", lambda: self.send_compute("step"))
        self.neural_run = button("Run", self.toggle_neural_run)
        self.silence = button("Silence", self.silence_network)
        self.fit = button("Train pair", self.train_pair)
        self.predict_button = button("Predict", lambda: self.send_compute("predict", message=self.message.toPlainText()))
        self.teacher = button("Ask professor", lambda: self.send_compute("teacher", message=self.message.toPlainText()))
        self.rest_button = button("Train rest", lambda: self.send_compute("rest", epochs=10))
        self.erase = button("Reset learning", self.reset_learning)
        self.compute_buttons = [self.inject, self.neural_step, self.neural_run, self.silence,
                                self.fit, self.predict_button, self.teacher, self.rest_button, self.erase]
        for widget in [self.start_network, *self.compute_buttons]:
            widget.setEnabled(False)
            row.addWidget(widget)
        area.addLayout(row)
        controls = QHBoxLayout()
        controls.addWidget(label("Firing steps/s", 8))
        self.firing_rate = QSlider(Qt.Orientation.Horizontal)
        self.firing_rate.setRange(1, 20)
        self.firing_rate.setValue(4)
        self.firing_rate.setFixedWidth(100)
        self.firing_rate.valueChanged.connect(lambda value: self.neural_timer.setInterval(round(1000 / value)))
        controls.addWidget(self.firing_rate)
        self.neural_status = label("15,625 neurons · sparse recurrence · not started", 8)
        controls.addWidget(self.neural_status, 1)
        self.stop_training = button("Stop training", self.cancel_training, "quiet")
        self.stop_training.setEnabled(False)
        controls.addWidget(self.stop_training)
        area.addLayout(controls)
        self.prediction = label("NETWORK OUTPUT   /   untrained · no prediction", 10, "#98e6cf")
        self.prediction.setTextFormat(Qt.TextFormat.PlainText)
        self.prediction.setWordWrap(True)
        self.target_source = label("Manual lab target · chat coaching has separate, labeled reward/example updates", 8)
        self.response.textChanged.connect(lambda: self.target_source.setText("Target source: edited/user supplied · train explicitly"))
        area.addWidget(self.prediction)
        area.addWidget(self.target_source)

    def send_compute(self, op, **arguments):
        if self.compute.busy:
            return False
        self.error.hide()
        if self.compute.request(op, **arguments):
            self.neural_status.setText({"teacher": "Professor generating offline…", "train": "Optimizing neural parameters…"}.get(op, "Computing " + op + "…"))
            self.update_compute_controls(True)
            status = {"init": "Loading your saved brain…", "enqueue": "Sending your message…",
                      "coach": "Professor reviewing the reply…", "correct": "Learning your correction and saving…",
                      "feedback": "Applying your reward and saving…", "speak": "Network composing another reply…"}.get(op)
            if status:
                self.chat.progress.setText(status)
            return True
        return False

    def update_compute_controls(self, busy=False):
        self.start_network.setEnabled(not busy and not self.network_ready and self.metadata is not None)
        for widget in self.compute_buttons:
            widget.setEnabled(self.network_ready and not busy and self.training_batches == 0)
        self.stop_training.setEnabled(self.training_batches > 0)
        self.chat.available(self.network_ready, busy, self.brain_running)
        if self.metadata is None:
            self.chat.start.setEnabled(False)

    def compute_failed(self, message):
        self.training_batches = 0
        self.brain_timer.stop()
        self.brain_running = False
        if self.compute.process.state() == self.compute.process.ProcessState.NotRunning:
            self.network_ready = False
            self.volume.activity = []
            self.volume.update()
            self.prediction.setText("NETWORK OUTPUT   /   compute stopped · restart to restore saved learning")
        self.neural_timer.stop()
        self.neural_run.setText("Run")
        self.error.setText(message)
        self.error.show()
        self.neural_status.setText("Operation failed · see error below")
        self.chat.progress.setText(message)
        self.update_compute_controls()

    def compute_result(self, payload):
        op, result = payload["op"], payload["result"]
        if op in ("init", "reset"):
            self.network_ready = True
            self.volume.neural = True
            self.volume.fit_zoom = self.volume.zoom = .72
            self.prediction.setText(f"NETWORK OUTPUT   /   learned vocabulary {result['vocabulary']} tokens" if result.get("vocabulary") else "NETWORK OUTPUT   /   untrained · target vocabulary empty")
        if "activity" in result:
            self.volume.activity = result["activity"]
            self.volume.input_activity = result.get("input_activity", [])
            self.volume.output_activity = result.get("output_activity", [])
            self.volume.update()
            self.neural_status.setText(f"15,625 neurons · active {result['active']:,} · peak {result['peak']:.4f} · updates {result['updates']}")
        elif op == "teacher":
            self.response.setPlainText(result["proposal"])
            self.target_source.setText("Target source: frozen local SmolLM2-135M · UNVERIFIED · edit, then Train pair")
            self.neural_status.setText("Professor ready · 16-token generation limit · offline CPU")
        elif op in ("train", "rest"):
            loss = result.get("loss", result.get("rest_loss"))
            self.neural_status.setText(f"{op.title()} loss {loss:.5f} · updates {result['updates']} · training metric only")
        if op == "predict":
            self.prediction.setText("NETWORK OUTPUT   /   " + (result["prediction"] or "[EOS / no response]"))
        if op == "enqueue":
            self.chat.queued(self.chat_message, result)
            self.message.setPlainText(self.chat_message)
            # Reference text from a prior conversation must not enter visualization.
            self.response.clear()
            self.visualize_text()
            self.volume.tokens = []  # No vertex is displayed as introduced before a tick.
            self.inspector.hide()
            self.transport.hide()
            self.volume.update()
        elif op in ("live_tick", "speak"):
            self.chat.show_tick(result)
            if result.get("token") and self.data:
                self.volume.tokens = self.data["message"]["tokens"][:result["received"]]
                self.select(min(result["received"] - 1, len(self.volume.tokens) - 1))
        elif op == "feedback":
            self.chat.rated = True
            self.chat.add_bubble("YOUR REWARD · policy update", f"{result['reward']:+d} at impact {result['strength']:g}. " +
                ("Feedback applied to the sampled reply." if result["update_attempted"] else "No learned words yet; start with a correction or professor example."), "#98e6cf")
            self.chat.progress.setText("Reward processed. The delivered message stays unchanged.")
        elif op == "coach":
            self.chat.coached(result)
        elif op == "correct":
            self.chat.corrected = self.chat.edit_turn_id == self.chat.turn_id
            edited = self.chat.corrected_text.toPlainText()
            self.chat.close_editor()
            self.chat.add_bubble("YOUR CORRECTION · example teaching", edited + "\n\nTaught this reply. Resume the brain when ready.", "#98e6cf")
            self.chat.progress.setText("Correction learned. Resume the brain when ready.")
        elif op == "reset":
            self.chat.clear_session()
            self.coached_message_id = None
        if op == "init" and self.start_requested:
            self.start_requested = False
            self.brain_running = True
        storage = result.get("storage")
        if storage:
            self.chat.saved_status(storage)
            if storage.get("error"):
                self.brain_running = False
                self.brain_timer.stop()
                self.error.setText(storage["error"])
                self.error.show()
        if op in ("init", "enqueue", "live_tick", "speak", "feedback", "coach", "correct") and self.brain_running:
            self.brain_timer.start(round(self.chat.duration.value() * 1000))
        self.update_compute_controls()
        if (op in ("live_tick", "speak") and result.get("reply", False) and self.chat.coach.isChecked()
                and self.coached_message_id != result["turn_id"]):
            self.coached_message_id = result["turn_id"]
            self.brain_timer.stop()
            self.send_compute("coach", turn_id=result["turn_id"])
        if op == "admit" and self.training_batches > 0:
            self.send_compute("train", epochs=10)
        elif op == "train" and self.training_batches > 0:
            self.training_batches -= 1
            if self.training_batches:
                QTimer.singleShot(0, lambda: self.send_compute("train", epochs=10) if self.training_batches else None)
            else:
                self.update_compute_controls()
                self.send_compute("predict", message=self.training_message)

    def set_expert_mode(self, enabled):
        self.expert_mode = enabled
        self.volume.reserve_inspector = enabled
        self.volume.show_embeddings = enabled
        self.workspace.setVisible(enabled)
        self.chat.setVisible(not enabled)
        self.mode_button.setText("Back to chat" if enabled else "Inspect network")
        self.inspector.setVisible(enabled and bool(self.volume.tokens))
        self.transport.setVisible(enabled and bool(self.volume.tokens))
        self.position_overlays()
        self.volume.update()

    def toggle_brain(self):
        if not self.network_ready:
            if self.compute.busy:
                return
            self.start_requested = True
            if not self.send_compute("init"):
                self.start_requested = False
            return
        if self.chat.editor.isVisible():
            self.chat.progress.setText("Finish or cancel the correction before resuming.")
            return
        self.brain_running = not self.brain_running
        self.neural_timer.stop()
        self.neural_run.setText("Run")
        if self.brain_running:
            self.brain_timer.start(round(self.chat.duration.value() * 1000))
            self.chat.progress.setText("Brain running · next tick after the selected duration")
        else:
            self.brain_timer.stop()
            self.chat.progress.setText("Brain paused · state and learning preserved")
        self.update_compute_controls(self.compute.busy)

    def set_tick_duration(self, seconds):
        self.brain_timer.setInterval(round(seconds * 1000))

    def brain_tick(self):
        if self.brain_running and not self.compute.busy:
            self.send_compute("live_tick")

    def chat_action(self, op, arguments):
        if self.compute.busy:
            return
        if op == "enqueue":
            try:
                self.chat_message = arguments["message"]
                if len(self.store.encode(self.chat_message)["tokens"]) > 16:
                    raise ValueError("Use at most 16 tokens in a sentence for this prototype")
            except Exception as error:
                self.chat.progress.setText(str(error))
                return
        self.brain_timer.stop()
        self.send_compute(op, **arguments)

    def chat_editing(self, editing):
        if editing:
            self.brain_running = False
            self.brain_timer.stop()
            self.neural_timer.stop()
            self.neural_run.setText("Run")
        self.update_compute_controls(self.compute.busy)

    def inject_token(self):
        self.send_compute("step", message=self.message.toPlainText(), index=self.index)

    def toggle_neural_run(self):
        self.brain_running = False
        self.brain_timer.stop()
        self.chat.available(self.network_ready, self.compute.busy, False)
        if self.neural_timer.isActive():
            self.neural_timer.stop()
            self.neural_run.setText("Run")
        else:
            self.neural_timer.start(round(1000 / self.firing_rate.value()))
            self.neural_run.setText("Pause")

    def silence_network(self):
        self.brain_running = False
        self.brain_timer.stop()
        self.neural_timer.stop()
        self.neural_run.setText("Run")
        self.send_compute("silence")

    def train_pair(self):
        self.brain_running = False
        self.brain_timer.stop()
        self.neural_timer.stop()
        self.neural_run.setText("Run")
        self.training_message = self.message.toPlainText()
        self.training_batches = 10
        if not self.send_compute("admit", message=self.training_message, target=self.response.toPlainText()):
            self.training_batches = 0

    def cancel_training(self):
        self.training_batches = 0
        self.stop_training.setEnabled(False)
        # A bounded ten-epoch request finishes; no optimizer state is discarded.
        if not self.compute.busy:
            self.update_compute_controls()

    def reset_learning(self):
        from PySide6.QtWidgets import QMessageBox
        answer = QMessageBox.question(self, "Reset learning?", "Erase this session's learned weights, target vocabulary and replay examples? Silence preserves them.")
        if answer == QMessageBox.StandardButton.Yes:
            self.silence_network()
            # Reset after silence completes on the next idle event.
            self.compute.result.connect(self.finish_reset)

    def finish_reset(self, payload):
        if payload["op"] == "silence":
            self.compute.result.disconnect(self.finish_reset)
            self.send_compute("reset")

    def eventFilter(self, watched, event):
        if watched is self.volume and event.type() == QEvent.Type.Resize and hasattr(self, "reset"):
            self.position_overlays()
        return super().eventFilter(watched, event)

    def position_overlays(self):
        self.inspector.setFixedHeight(min(self.inspector_content.sizeHint().height() + 4, max(180, self.volume.height() - 140)))
        self.inspector.move(self.volume.width() - 254, 55)
        self.transport.adjustSize()
        self.transport.move(max(24, (self.volume.width() - self.transport.width()) // 2), self.volume.height() - 66)
        self.reset.adjustSize()
        self.reset.move(self.volume.width() - self.reset.width() - 25, self.volume.height() - 55)
        self.guides.adjustSize()
        self.guides.move(self.reset.x() - self.guides.width() - 17, self.volume.height() - 45)

    def model_ready(self, metadata):
        self.metadata = metadata
        self.status.setText("●  LOCAL MODEL READY   /   OFFLINE")
        self.status.setStyleSheet("color:#9adfc9")
        self.model_label.setText(f"SmolLM2-135M · {metadata['dimensions']} dimensions · frozen embeddings · local CPU compute")
        self.about.setText(f"Frozen {metadata['dimensions']}-component embeddings, revision {metadata['revision']}. A learned projection feeds 512 input units; every core neuron connects to all input/output units. The core has 15,625 recurrent ReLU units and 12 recurrent inputs each; the learned output layer has 512 units. Coordinates are visual aids only and never choose neurons. Messaging waits for the full sentence and sends one stable network reply; idle ticks do not rewrite it. Try again explores another reply for feedback. Rewards use an immediate policy update; edited targets use supervised learning. Professor votes are unverified, not ground truth. Up to 16 output tokens and 511 learned target tokens. Learning, optimizer and bounded numeric replay autosave privately on the checkout drive; chat history is not recorded. Startup restores learning with activity silent. No consciousness or general reasoning is established.")
        self.visualize.setEnabled(True)
        self.update_compute_controls()

    def model_failed(self, message):
        self.status.setText("●  MODEL UNAVAILABLE")
        self.error.setText(message)
        self.error.show()

    def visualize_text(self):
        self.stop()
        self.error.hide()
        try:
            message = self.store.encode(self.message.toPlainText())
            response = self.store.encode(self.response.toPlainText()) if self.response.toPlainText().strip() else None
        except Exception as error:
            self.error.setText(str(error))
            self.error.show()
            return
        self.data = {"message": message, "response": response}
        self.message_tab.setText(f"Message  {len(message['tokens'])}")
        self.response_tab.setText(f"Response reference  {len(response['tokens']) if response else 0}")
        self.response_tab.setEnabled(response is not None)
        extent = max(abs(v) for seq in (message, response) if seq for token in seq["tokens"] for p in token["points"] for v in p)
        self.volume.fit_zoom = .72 if self.volume.neural else min(50, .85 / max(extent, .01))
        self.volume.reset_camera()
        self.all.setChecked(False)
        self.switch("message")
        self.inspector.setVisible(self.expert_mode)
        self.transport.setVisible(self.expert_mode)
        self.position_overlays()

    def switch(self, sequence):
        if not self.data or not self.data[sequence]: return
        self.stop()
        self.sequence = sequence
        self.message_tab.setChecked(sequence == "message")
        self.response_tab.setChecked(sequence == "response")
        self.volume.tokens = self.data[sequence]["tokens"]
        self.volume.response = sequence == "response"
        while self.chips_layout.count():
            item = self.chips_layout.takeAt(0)
            item.widget().hide()
            item.widget().deleteLater()
        self.chips = []
        for index, token in enumerate(self.volume.tokens):
            text = token["text"].replace(" ", "␠").replace("\n", "↵").replace("\t", "⇥")
            chip = button(text or "∅", lambda checked=False, i=index: self.select_manual(i))
            chip.setCheckable(True)
            chip.setToolTip(f"Token {index + 1} · ID {token['id']} · raw piece: {token['piece']}")
            chip.setFont(QFont("Consolas", 11))
            self.chips_layout.addWidget(chip)
            self.chips.append(chip)
        self.select(0)

    def select(self, index):
        if not self.volume.tokens: return
        self.index = max(0, min(len(self.volume.tokens) - 1, index))
        self.volume.index = self.index
        token = self.volume.tokens[self.index]
        self.token_number.setText(f"SELECTED TOKEN     {self.index + 1:02d} / {len(self.volume.tokens)}")
        # Plain text labels prevent input such as <b> from becoming UI markup.
        self.token_text.setTextFormat(Qt.TextFormat.PlainText)
        self.token_piece.setTextFormat(Qt.TextFormat.PlainText)
        self.token_text.setText(repr(token["text"]))
        self.token_text.setStyleSheet(f"color:{'#f2bd95' if self.volume.response else '#98e6cf'};font-size:29px;background:transparent")
        self.token_piece.setText(f"Raw piece: {token['piece']}")
        self.token_details.setText(f"Token ID                 {token['id']}\nDimensions             {len(token['vector'])}\nVertices / segments    {len(token['points'])} / {len(token['points']) - 1}")
        self.raw.setText("\n".join("(" + ", ".join(f"{v:.5f}" for v in token["vector"][i:i + 3]) + ")" for i in range(0, min(9, len(token["vector"])), 3)))
        self.position.setText(f"{self.index + 1} / {len(self.volume.tokens)}")
        self.previous.setEnabled(self.index > 0)
        self.next.setEnabled(self.index < len(self.volume.tokens) - 1)
        self.play.setEnabled(len(self.volume.tokens) > 1 and not self.all.isChecked())
        for i, chip in enumerate(self.chips): chip.setChecked(i == self.index)
        self.strip.ensureWidgetVisible(self.chips[self.index], 10, 0)
        self.position_overlays()
        self.volume.update()

    def select_manual(self, index):
        self.stop()
        self.select(index)

    def step(self, amount):
        self.stop()
        self.select(self.index + amount)

    def stop(self):
        self.timer.stop()
        self.play.setText("Play")

    def toggle_play(self):
        if self.timer.isActive(): return self.stop()
        if not self.volume.tokens or self.all.isChecked(): return
        if self.index == len(self.volume.tokens) - 1: self.select(0)
        self.timer.start(round(1000 / (self.speed.value() / 4)))
        self.play.setText("Pause")

    def advance(self):
        self.select(self.index + 1)
        if self.index == len(self.volume.tokens) - 1: self.stop()

    def change_speed(self, value):
        self.pace.setText(f"{value / 4:g} token/s")
        if self.timer.isActive(): self.timer.setInterval(round(1000 / (value / 4)))

    def toggle_all(self, enabled):
        self.stop()
        self.volume.all_tokens = enabled
        if self.volume.tokens: self.select(self.index)

    def toggle_guides(self, enabled):
        self.volume.guides = enabled
        self.volume.update()

    def toggle_about(self):
        self.about.setVisible(not self.about.isVisible())

    def clear(self):
        self.stop()
        self.data, self.volume.tokens = None, []
        self.all.setChecked(False)
        self.volume.fit_zoom = 1
        self.volume.reset_camera()
        self.inspector.hide()
        self.transport.hide()
        self.response_tab.setEnabled(False)
        self.message_tab.setChecked(True)
        self.response_tab.setChecked(False)
        self.message_tab.setText("Message  0")
        self.response_tab.setText("Response reference  0")
        while self.chips_layout.count():
            widget = self.chips_layout.takeAt(0).widget()
            widget.hide()
            widget.deleteLater()
        self.chips = []
        self.chips_layout.addWidget(label("Your tokens will appear here.", 10, "#647f8f"))
        self.error.hide()

    def closeEvent(self, event):
        self.stop()
        self.brain_running = False
        self.training_batches = 0
        self.start_requested = False
        self.brain_timer.stop()
        self.neural_timer.stop()
        self.compute.close()
        self.loader.wait()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Neural Volume Lab")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    icon = QPixmap(64, 64)
    icon.fill(QColor("#10232c"))
    painter = QPainter(icon)
    painter.setPen(QColor("#98e6cf"))
    painter.setFont(QFont("Segoe UI", 30, QFont.Weight.Light))
    painter.drawText(icon.rect(), Qt.AlignmentFlag.AlignCenter, "nv")
    painter.end()
    app.setWindowIcon(QIcon(icon))
    verification_flags = ("--verify-output", "--verify-learning-output", "--verify-chat-output")
    isolated_checkpoint = None
    for flag in verification_flags:
        if flag in sys.argv:
            isolated_checkpoint = Path(sys.argv[sys.argv.index(flag) + 1]).resolve() / "brain.pt"
    window = Window(checkpoint=isolated_checkpoint)
    available = app.primaryScreen().availableGeometry()
    window.resize(min(1320, round(available.width() * .94)), min(920, round(available.height() * .92)))
    if "--verify-output" in sys.argv:
        from check_desktop import verify
        index = sys.argv.index("--verify-output")
        destination = Path(sys.argv[index + 1])
        window.loader.ready.connect(lambda _: QTimer.singleShot(100, lambda: verify(window, app, destination)))
        window.loader.failed.connect(lambda _: app.exit(1))
    if "--verify-learning-output" in sys.argv:
        from check_learning_desktop import verify_learning
        destination = Path(sys.argv[sys.argv.index("--verify-learning-output") + 1])
        window.loader.ready.connect(lambda _: QTimer.singleShot(100, lambda: verify_learning(window, app, destination)))
        window.loader.failed.connect(lambda _: app.exit(1))
    if "--verify-chat-output" in sys.argv:
        from check_chat_desktop import verify_chat
        destination = Path(sys.argv[sys.argv.index("--verify-chat-output") + 1])
        window.loader.ready.connect(lambda _: QTimer.singleShot(100, lambda: verify_chat(window, app, destination)))
        window.loader.failed.connect(lambda _: app.exit(1))
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
