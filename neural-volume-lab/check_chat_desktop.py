"""Explicit synthetic native/real-worker tick, correction, reward and coach smoke."""
import json
from pathlib import Path
import sys
import time

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest


def verify_chat(window, app, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    events = []
    window.compute.result.connect(lambda payload: events.append(payload))

    def until(condition, seconds=45):
        deadline = time.monotonic() + seconds
        while not condition():
            app.processEvents()
            QTest.qWait(5)
            if time.monotonic() > deadline:
                raise RuntimeError("Chat verification timed out")
        app.processEvents()
        if window.error.isVisible():
            raise RuntimeError(window.error.text())

    def settle():
        until(lambda: not window.compute.busy)

    def tick():
        window.brain_running = True
        window.brain_tick()
        window.brain_running = False
        window.brain_timer.stop()
        settle()

    try:
        assert window.chat.isVisible() and not window.workspace.isVisible()
        window.chat.coach.setChecked(False)
        QTest.mouseClick(window.chat.start, Qt.MouseButton.LeftButton)
        settle()
        assert window.network_ready and window.brain_running
        assert window.brain_timer.interval() == 5000
        window.chat.composer.setPlainText("Hello world")
        QTest.mouseClick(window.chat.send, Qt.MouseButton.LeftButton)
        settle()
        assert window.chat.input_pending and not window.volume.tokens
        started = time.monotonic()
        until(lambda: window.chat.tick_count == 1, seconds=10)
        elapsed = time.monotonic() - started
        QTest.mouseClick(window.chat.start, Qt.MouseButton.LeftButton)
        assert not window.brain_running
        first = [p["result"] for p in events if p["op"] == "live_tick"][-1]
        assert first["received"] == 1 and first["total"] == 2 and first["token"]["text"] == "Hello"
        assert not first["reply"] and window.chat.turn_id is None
        assert first["active"] > 0 and len(window.volume.tokens) == 1
        assert elapsed >= 4.5  # Actual five-second native timer, not just a property check.
        window.chat.duration.setValue(2.5)
        assert window.brain_timer.interval() == 2500
        tick()
        assert not window.chat.input_pending and len(window.volume.tokens) == 2
        delivered = window.chat.latest_reply
        old_text = delivered.text()
        app.processEvents()
        QTest.mouseDClick(window.chat.latest_reply, Qt.MouseButton.LeftButton)
        assert window.chat.editor.isVisible() and not window.brain_timer.isActive()
        window.chat.corrected_text.setPlainText("Welcome")
        QTest.mouseClick(window.chat.save, Qt.MouseButton.LeftButton)
        settle()
        assert not window.chat.editor.isVisible()
        correction = [p["result"] for p in events if p["op"] == "correct"][-1]
        assert correction["kind"] == "supervised_correction"
        assert delivered.text() == old_text
        QTest.mouseClick(window.chat.practice, Qt.MouseButton.LeftButton)
        settle()
        # A separate sampled message; the corrected original stays unchanged.
        assert window.chat.latest_reply.text() == "Welcome"
        assert delivered.text() == old_text and delivered is not window.chat.latest_reply
        QTest.mouseClick(window.chat.good, Qt.MouseButton.LeftButton)
        settle()
        feedback = [p["result"] for p in events if p["op"] == "feedback"][-1]
        assert feedback["reward"] == 1 and feedback["strength"] == 5
        window.chat.coach.setChecked(True)
        QTest.mouseClick(window.chat.practice, Qt.MouseButton.LeftButton)
        settle()
        coach = [p["result"] for p in events if p["op"] == "coach"][-1]
        assert coach["proposal"].strip() and coach["source"].startswith("Unverified")
        assert coach["teaching"] is None  # Professor cannot overwrite the human example.
        # Restart the actual compute process, using the isolated on-disk brain.
        window.brain_running = False
        window.brain_timer.stop()
        window.compute.close()
        window.network_ready = False
        window.send_compute("init")
        settle()
        restored = [p["result"] for p in events if p["op"] == "init"][-1]
        assert restored["storage"]["restored"] and restored["updates"] >= 12
        window.send_compute("predict", message="Hello world")
        settle()
        assert [p["result"] for p in events if p["op"] == "predict"][-1]["prediction"] == "Welcome"
        window.brain_timer.stop()
        window.chat.duration.setValue(5)
        window.chat.progress.setText("Paused · saved learning was restored successfully")
        app.processEvents()
        assert window.grab().save(str(destination / "chat.png"))
        result = {"status": "passed", "frozen_executable": bool(getattr(sys, "frozen", False)),
            "first_tick_elapsed_seconds": elapsed, "feedback": feedback, "correction": correction,
            "professor": coach, "restored_updates": restored["updates"],
            "architecture": {k: restored[k] for k in ("neurons", "parameters", "input_units", "output_units", "input_encoding")},
            "checks": ["chat-first native view", "start brain", "real five-second tick",
                "one token per tick", "no pre-tick introduced vertices", "live neural readout",
                "adjustable tick duration", "double-click exact reply", "pause while editing",
                "supervised correction fitted", "strong scalar reward", "offline professor judgment",
                "human example priority", "stable delivered messages", "explicit practice reply",
                "learned 512-unit input and output", "checkpoint restored in a new worker", "native screenshot"],
            "limits": "Synthetic development smoke, not evidence of useful dialogue or teacher accuracy"}
        (destination / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        window.compute.close()
        app.exit(0)
    except Exception as error:
        window.brain_timer.stop()
        (destination / "failure.txt").write_text(f"{type(error).__name__}: {error}", encoding="utf-8")
        window.compute.close()
        app.exit(1)
