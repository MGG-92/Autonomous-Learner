"""Chat-first native controls; displayed replies always come from the neural worker."""
from PySide6.QtCore import Qt, Signal, QTimer, QDateTime
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QPlainTextEdit, QDoubleSpinBox, QSpinBox, QAbstractSpinBox, QScrollArea, QCheckBox, QFrame,
    QSizePolicy)


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
        self.ui_busy = False
        self.storage_announced = False
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
        self.duration.setMinimumWidth(140)
        self.duration.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.PlusMinus)
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
        self.add_bubble("HOW THIS WORKS", "Start the brain and send a message. One token enters per tick. The network sends one reply when it has received the sentence. Reward the reply or double-click it to teach a correction.", "#92acb7")
        self.live_title = QLabel("LAST DELIVERED REPLY")
        self.live_title.setStyleSheet("color:#98e6cf;font-size:10px")
        layout.addWidget(self.live_title)
        self.live = Reply("No reply yet.")
        self.live.setTextFormat(Qt.TextFormat.PlainText)
        self.live.setWordWrap(True)
        self.live.setMinimumHeight(42)
        self.live.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.live.setStyleSheet("color:#dce9ed;background:transparent;padding:10px;font-size:15px")
        self.live.setToolTip("Double-click to correct this exact reply")
        self.live.edit_requested.connect(lambda: self.open_editor())
        self.live_scroll = QScrollArea()
        self.live_scroll.setWidgetResizable(True)
        self.live_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.live_scroll.setMinimumHeight(52)
        self.live_scroll.setMaximumHeight(100)
        self.live_scroll.setStyleSheet("QScrollArea{background:#13252c;border:1px solid #36535b;border-radius:8px}")
        self.live_scroll.setWidget(self.live)
        layout.addWidget(self.live_scroll)
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
        self.strength.setMinimumWidth(66)
        self.strength.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.PlusMinus)
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
        correction_title = QLabel("YOUR CORRECTION · example teaching, not a reward")
        correction_title.setWordWrap(True)
        editor_layout.addWidget(correction_title)
        self.corrected_text = QPlainTextEdit()
        self.corrected_text.setMinimumHeight(60)
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
        self.compose_panel = QWidget()
        compose_layout = QVBoxLayout(self.compose_panel)
        compose_layout.setContentsMargins(0, 0, 0, 0)
        compose_layout.setSpacing(12)
        self.composer = Composer()
        self.composer.setPlaceholderText("Talk to the learner…  Enter to send")
        self.composer.setFixedHeight(72)
        self.composer.setAccessibleName("Chat message")
        self.composer.submitted.connect(self.submit)
        compose_layout.addWidget(self.composer)
        self.send = QPushButton("Send message  ↗", objectName="primary")
        self.send.clicked.connect(self.submit)
        self.send.setEnabled(False)
        send_row = QHBoxLayout()
        send_row.addWidget(self.send, 1)
        self.practice = QPushButton("Try again")
        self.practice.setToolTip("The network explores another reply to the last sentence. Professor review, if enabled, supplies feedback; guesses are not treated as truth.")
        self.practice.clicked.connect(lambda: self.action.emit("speak", {}))
        self.practice.setEnabled(False)
        send_row.addWidget(self.practice)
        compose_layout.addLayout(send_row)
        layout.addWidget(self.compose_panel)
        footer = QLabel("Short messages · 16 tokens · 512-unit input/output layers\nLearning restores when you start. Professor feedback is unverified.")
        self.save_note = footer
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
            body.setToolTip("Double-click to correct this reply. Credit is retained for the last eight messages.")
        content.addWidget(body)
        body.message_role = role_label
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
        self.add_bubble("YOU · " + self.timestamp(), text, "#b7cbd4")
        self.latest_reply = self.add_bubble("NETWORK · receiving", "Receiving your sentence — no reply yet…", "#98e6cf", reply=True)
        self.composer.clear()
        self.input_pending = True
        self.turn_id = None
        self.progress.setText(f"Message sent · receiving 0/{result['queued']} tokens")

    def show_tick(self, result):
        if not result.get("practice"):
            self.tick_count += 1
        if "received" in result:
            self.input_pending = result["received"] < result["total"]
        if result.get("reply"):
            self.turn_id = result["turn_id"]
            self.rated = self.corrected = False
            self.reply_text = result["prediction"]
            shown = self.reply_text or "[The network sent no words — teach a first example.]"
            if result.get("practice") or self.latest_reply is None:
                self.latest_reply = self.add_bubble("NETWORK", shown, "#98e6cf", reply=True)
            self.latest_reply.message_role.setText("NETWORK · " + self.timestamp())
            self.latest_reply.setText(shown)
            self.latest_reply.setProperty("turn_id", self.turn_id)
            self.latest_reply.setProperty("reply_text", self.reply_text)
            self.live_title.setText("LAST DELIVERED REPLY · " + self.timestamp())
            self.live.setText(shown)
            self.progress.setText(f"Reply delivered · tick {self.tick_count} · {result['active']:,} active neurons")
            self.bottom()
        elif self.input_pending:
            self.progress.setText(f"Receiving your message · {result['received']}/{result['total']} tokens\nNo reply yet. Next token after {self.duration.value():g} seconds.")
            self.live_title.setText("NETWORK IS RECEIVING")
            self.live.setText("No reply yet — receiving your sentence.")
            if self.latest_reply is not None:
                self.latest_reply.setText(f"Receiving {result['received']}/{result['total']} tokens…")
        else:
            self.progress.setText(f"Brain active · tick {self.tick_count} · {result['active']:,} active neurons\nNo new message. Delivered replies stay unchanged.")

    @staticmethod
    def timestamp():
        return QDateTime.currentDateTime().toString("HH:mm")

    def rate(self, reward):
        if self.turn_id is not None:
            self.action.emit("feedback", {"turn_id": self.turn_id, "reward": reward, "strength": self.strength.value()})

    def open_editor(self, selected=None):
        if not isinstance(selected, Reply):
            selected = self.latest_reply
        if selected is None or selected.property("turn_id") is None or self.ui_busy:
            return
        self.edit_turn_id = selected.property("turn_id")
        self.corrected_text.setPlainText(selected.property("reply_text") or "")
        self.compose_panel.hide()
        self.editor.show()
        self.editing.emit(True)
        self.corrected_text.setFocus()
        self.progress.setText("Brain paused for editing. Teach your correction, then resume.")

    def close_editor(self):
        self.editor.hide()
        self.compose_panel.show()
        self.editing.emit(False)
        self.progress.setText("Brain paused. Resume when ready.")

    def available(self, ready, busy, running):
        self.ui_busy = busy
        self.start.setEnabled(ready or not busy)
        self.start.setText("Pause brain" if running else "Resume brain" if ready else "Start brain")
        self.send.setEnabled(ready and not busy and not self.input_pending and not self.editor.isVisible())
        self.practice.setEnabled(ready and not busy and not self.input_pending and self.turn_id is not None and not self.editor.isVisible())
        can_rate = ready and not busy and self.turn_id is not None and self.latest_reply is not None
        self.good.setEnabled(can_rate and not self.rated)
        self.bad.setEnabled(can_rate and not self.rated)
        self.edit.setEnabled(can_rate and not self.corrected)
        self.save.setEnabled(not busy)
        self.cancel.setEnabled(not busy)

    def coached(self, result):
        extra = " + short example teaching" if result["teaching"] else " · reward only"
        self.add_bubble("PROFESSOR · unverified feedback", result["verdict"] + f"\nReward {result['reward']:+d}" + extra + "\nExample: " + result["proposal"], "#f2bd95")
        self.progress.setText("Example learned. Click Try again to let the network practice." if result["teaching"] else
                              "Professor review finished. Reward or correct the network's reply.")

    def saved_status(self, storage):
        if storage.get("error"):
            self.save_note.setText("Saving failed · current learning is in memory only. See the error below.")
            return
        self.save_note.setText(f"Learning saved locally · {storage['updates']} updates\n512-unit input/output with neural feedback · professor is unverified")
        if not self.storage_announced:
            self.storage_announced = True
            text = ("Restored saved learning" if storage.get("restored") else "New brain created") + f" · {storage['updates']} updates. Learning autosaves locally; the chat starts fresh."
            if storage.get("recovered"):
                text += " Recovered the previous valid checkpoint."
            if storage.get("migrated"):
                text += " Added output-to-volume connections at zero strength; existing learning was preserved."
            self.add_bubble("SAVED BRAIN", text, "#98e6cf")

    def clear_session(self):
        self.turn_id = None
        self.input_pending = False
        self.latest_reply = None
        self.reply_text = ""
        self.tick_count = 0
        self.message_id = None
        self.live.setText("No reply yet.")
        self.live_title.setText("LAST DELIVERED REPLY")
        self.close_editor()
        self.add_bubble("SESSION", "Learning reset. Start with a short example.")
