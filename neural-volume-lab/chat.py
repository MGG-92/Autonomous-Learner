"""Chat-first native controls; displayed replies always come from the neural worker."""
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QPlainTextEdit, QDoubleSpinBox, QSpinBox, QScrollArea, QCheckBox, QFrame)


class Composer(QPlainTextEdit):
    submitted = Signal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.submitted.emit()
            event.accept()
        else:
            super().keyPressEvent(event)


class Reply(QLabel):
    edit_requested = Signal()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.edit_requested.emit()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)


class Chat(QWidget):
    action = Signal(str, object)
    brain_toggled = Signal()
    duration_changed = Signal(float)
    editing = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(400)
        self.turn_id = None
        self.message_id = None
        self.latest_reply = None
        self.reply_text = ""
        self.tick_count = 0
        self.input_pending = False
        self.rated = self.corrected = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(12)
        title = QLabel("CONVERSATION")
        title.setStyleSheet("color:#b9d5d9;font-size:13px;letter-spacing:2px")
        layout.addWidget(title)
        controls = QHBoxLayout()
        self.start = QPushButton("Start brain")
        self.start.clicked.connect(self.brain_toggled.emit)
        self.start.setEnabled(False)
        self.duration = QDoubleSpinBox()
        self.duration.setRange(.2, 60)
        self.duration.setValue(5)
        self.duration.setSingleStep(.5)
        self.duration.setSuffix(" s / tick")
        self.duration.setAccessibleName("Seconds per neural tick")
        self.duration.valueChanged.connect(self.duration_changed.emit)
        controls.addWidget(self.start, 1)
        controls.addWidget(self.duration)
        layout.addLayout(controls)
        self.progress = QLabel("Start the brain, then send a short sentence.")
        self.progress.setWordWrap(True)
        self.progress.setStyleSheet("color:#92acb7;font-size:12px")
        layout.addWidget(self.progress)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.transcript = QWidget()
        self.messages = QVBoxLayout(self.transcript)
        self.messages.setContentsMargins(0, 0, 5, 0)
        self.messages.setSpacing(12)
        self.messages.addStretch()
        self.scroll.setWidget(self.transcript)
        self.follow_tail = True
        bar = self.scroll.verticalScrollBar()
        bar.rangeChanged.connect(lambda _minimum, maximum: bar.setValue(maximum) if self.follow_tail else None)
        bar.valueChanged.connect(lambda value: setattr(self, "follow_tail", value >= bar.maximum() - 24))
        layout.addWidget(self.scroll, 1)
        self.add_bubble("HOW THIS WORKS", "One token enters on each tick. The network's attempted reply evolves with its real activity. Reward it, or double-click its reply to teach a correction.", "#92acb7")
        self.live_title = QLabel("CURRENT NETWORK REPLY")
        self.live_title.setStyleSheet("color:#98e6cf;font-size:10px")
        layout.addWidget(self.live_title)
        self.live = Reply("Waiting for input and a tick…")
        self.live.setTextFormat(Qt.TextFormat.PlainText)
        self.live.setWordWrap(True)
        self.live.setMinimumHeight(42)
        self.live.setMaximumHeight(85)
        self.live.setStyleSheet("color:#dce9ed;background:#13252c;border:1px solid #36535b;border-radius:8px;padding:10px;font-size:15px")
        self.live.setToolTip("Double-click to correct this exact reply")
        self.live.edit_requested.connect(lambda: self.open_editor())
        layout.addWidget(self.live)
        ratings = QHBoxLayout()
        self.good = QPushButton("Good reply  +")
        self.bad = QPushButton("Needs work  −")
        self.edit = QPushButton("Edit reply")
        self.good.clicked.connect(lambda: self.rate(1))
        self.bad.clicked.connect(lambda: self.rate(-1))
        self.edit.clicked.connect(self.open_editor)
        for item in (self.good, self.bad, self.edit):
            item.setEnabled(False)
            ratings.addWidget(item)
        layout.addLayout(ratings)
        strength_row = QHBoxLayout()
        strength_row.addWidget(QLabel("Your feedback impact"))
        self.strength = QSpinBox()
        self.strength.setRange(1, 10)
        self.strength.setValue(5)
        self.strength.setAccessibleName("Human feedback strength, one to ten")
        strength_row.addWidget(self.strength)
        strength_row.addStretch()
        self.coach = QCheckBox("Professor coach")
        self.coach.setChecked(True)
        self.coach.setToolTip("After a sentence arrives: unverified reward, and a short example when new words are needed. Your corrections take priority.")
        strength_row.addWidget(self.coach)
        layout.addLayout(strength_row)
        self.editor = QFrame(objectName="inspector")
        editor_layout = QVBoxLayout(self.editor)
        editor_layout.addWidget(QLabel("YOUR CORRECTION · example teaching, not a reward"))
        self.corrected_text = QPlainTextEdit()
        self.corrected_text.setMaximumHeight(75)
        self.corrected_text.setAccessibleName("Corrected neural reply")
        editor_layout.addWidget(self.corrected_text)
        buttons = QHBoxLayout()
        self.save = QPushButton("Teach this reply")
        self.save.clicked.connect(lambda: self.action.emit("correct", {"turn_id": self.edit_turn_id, "target": self.corrected_text.toPlainText()}))
        self.cancel = QPushButton("Cancel")
        self.cancel.clicked.connect(self.close_editor)
        buttons.addWidget(self.save)
        buttons.addWidget(self.cancel)
        editor_layout.addLayout(buttons)
        layout.addWidget(self.editor)
        self.editor.hide()
        self.composer = Composer()
        self.composer.setPlaceholderText("Talk to the learner…  Enter to send")
        self.composer.setFixedHeight(72)
        self.composer.setAccessibleName("Chat message")
        self.composer.submitted.connect(self.submit)
        layout.addWidget(self.composer)
        self.send = QPushButton("Send sentence  ↗", objectName="primary")
        self.send.clicked.connect(self.submit)
        self.send.setEnabled(False)
        layout.addWidget(self.send)
        footer = QLabel("Short turns · 16 tokens · session-only learning\nProfessor feedback is fallible. The network speaks for itself.")
        footer.setStyleSheet("color:#7894a4;font-size:11px")
        footer.setWordWrap(True)
        layout.addWidget(footer)

    def add_bubble(self, role, text, color="#dce9ed", reply=False):
        frame = QFrame(objectName="inspector")
        content = QVBoxLayout(frame)
        content.setContentsMargins(14, 11, 14, 13)
        role_label = QLabel(role)
        role_label.setStyleSheet(f"color:{color};font-size:10px")
        content.addWidget(role_label)
        body = Reply(text) if reply else QLabel(text)
        body.setTextFormat(Qt.TextFormat.PlainText)
        body.setWordWrap(True)
        body.setStyleSheet("color:#dce9ed;font-size:14px;background:transparent")
        if reply:
            body.edit_requested.connect(lambda body=body: self.open_editor(body))
            body.setToolTip("Double-click to correct this reply. Credit is retained for the last eight ticks.")
        content.addWidget(body)
        self.messages.insertWidget(self.messages.count() - 1, frame)
        while self.messages.count() > 41:
            old = self.messages.takeAt(0).widget()
            old.hide()
            old.deleteLater()
        self.bottom()
        return body

    def bottom(self):
        self.follow_tail = True
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum()))

    def submit(self):
        if self.send.isEnabled():
            self.action.emit("enqueue", {"message": self.composer.toPlainText()})

    def queued(self, text, result):
        self.message_id = result["message_id"]
        self.add_bubble("YOU", text, "#b7cbd4")
        self.latest_reply = self.add_bubble("NETWORK · live attempted reply", "Waiting for the next tick…", "#98e6cf", reply=True)
        self.composer.clear()
        self.input_pending = True
        self.turn_id = None
        self.progress.setText(f"Queued {result['queued']} tokens · one enters per tick")

    def show_tick(self, result):
        self.tick_count += 1
        self.turn_id = result["turn_id"]
        self.rated = self.corrected = False
        self.reply_text = result["prediction"]
        self.live_title.setText(f"CURRENT NETWORK REPLY · TICK {self.tick_count}")
        self.live.setText(self.reply_text or "[No words yet / end of response]")
        if self.latest_reply is not None:
            self.latest_reply.setText(self.reply_text or "[No words yet / end of response]")
            self.latest_reply.setProperty("turn_id", self.turn_id)
            self.latest_reply.setProperty("reply_text", self.reply_text)
        self.input_pending = result["received"] < result["total"]
        entered = repr(result["token"]["text"]) if result["token"] else "no new input"
        self.progress.setText(f"Tick {self.tick_count} · {entered}\nReceived {result['received']}/{result['total']} tokens · {result['active']:,} active neurons")

    def rate(self, reward):
        if self.turn_id is not None:
            self.action.emit("feedback", {"turn_id": self.turn_id, "reward": reward, "strength": self.strength.value()})

    def open_editor(self, selected=None):
        if not isinstance(selected, Reply):
            selected = self.latest_reply
        if selected is None or selected.property("turn_id") is None or not self.edit.isEnabled():
            return
        self.edit_turn_id = selected.property("turn_id")
        self.corrected_text.setPlainText(selected.property("reply_text") or "")
        self.editor.show()
        self.editing.emit(True)
        self.corrected_text.setFocus()
        self.progress.setText("Brain paused for editing. Teach your correction, then resume.")

    def close_editor(self):
        self.editor.hide()
        self.editing.emit(False)

    def available(self, ready, busy, running):
        self.start.setEnabled(ready or not busy)
        self.start.setText("Pause brain" if running else "Resume brain" if ready else "Start brain")
        self.send.setEnabled(ready and not busy and not self.input_pending and not self.editor.isVisible())
        can_rate = ready and not busy and self.turn_id is not None and self.latest_reply is not None
        self.good.setEnabled(can_rate and not self.rated)
        self.bad.setEnabled(can_rate and not self.rated)
        self.edit.setEnabled(can_rate and not self.corrected)
        self.save.setEnabled(not busy)
        self.cancel.setEnabled(not busy)

    def coached(self, result):
        extra = " + short example teaching" if result["teaching"] else " · reward only"
        self.add_bubble("PROFESSOR · unverified feedback", result["verdict"] + f"\nReward {result['reward']:+d}" + extra + "\nExample: " + result["proposal"], "#f2bd95")

    def clear_session(self):
        self.turn_id = None
        self.input_pending = False
        self.latest_reply = None
        self.reply_text = ""
        self.tick_count = 0
        self.message_id = None
        self.live.setText("Waiting for input and a tick…")
        self.live_title.setText("CURRENT NETWORK REPLY")
        self.close_editor()
        self.add_bubble("SESSION", "Learning reset. Start with a short example.")
