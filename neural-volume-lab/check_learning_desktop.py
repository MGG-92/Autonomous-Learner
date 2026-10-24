"""Opt-in actual-model native verification; synthetic text, ignored outputs only."""
import json
from pathlib import Path
import sys
import time

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest


def verify_learning(window, app, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    events = []
    window.compute.result.connect(lambda payload: events.append(payload))

    def settle(seconds=60):
        deadline = time.monotonic() + seconds
        while window.compute.busy or window.training_batches:
            app.processEvents()
            QTest.qWait(10)
            if time.monotonic() > deadline:
                raise RuntimeError("Compute verification timed out")
        app.processEvents()
        if window.error.isVisible():
            raise RuntimeError(window.error.text())

    try:
        window.set_expert_mode(True)
        window.message.setPlainText("Hello")
        window.response.setPlainText("Welcome")
        window.visualize_text()
        QTest.mouseClick(window.start_network, Qt.MouseButton.LeftButton)
        settle()
        assert window.network_ready and window.volume.neural
        assert events[-1]["result"]["neurons"] == 15625
        assert events[-1]["result"]["active"] == 0
        QTest.mouseClick(window.inject, Qt.MouseButton.LeftButton)
        settle()
        assert events[-1]["result"]["active"] > 0
        QTest.mouseClick(window.silence, Qt.MouseButton.LeftButton)
        settle()
        assert events[-1]["result"]["active"] == 0
        QTest.mouseClick(window.neural_step, Qt.MouseButton.LeftButton)
        settle()
        assert events[-1]["result"]["active"] == 0
        QTest.mouseClick(window.teacher, Qt.MouseButton.LeftButton)
        settle()
        assert window.response.toPlainText().strip()
        assert "UNVERIFIED" in window.target_source.text()
        assert window.prediction.text().endswith("target vocabulary empty")
        professor = events[-1]["result"]
        # Teacher text did not automatically enter training.
        window.response.setPlainText("Welcome")
        QTest.mouseClick(window.fit, Qt.MouseButton.LeftButton)
        settle()
        assert window.prediction.text() == "NETWORK OUTPUT   /   Welcome"
        window.message.setPlainText("Goodbye")
        window.response.setPlainText("Farewell")
        QTest.mouseClick(window.fit, Qt.MouseButton.LeftButton)
        settle()
        assert window.prediction.text() == "NETWORK OUTPUT   /   Farewell"
        window.message.setPlainText("Hello")
        window.response.setPlainText("WRONG TARGET MUST NOT ENTER PREDICT")
        QTest.mouseClick(window.predict_button, Qt.MouseButton.LeftButton)
        settle()
        assert window.prediction.text() == "NETWORK OUTPUT   /   Welcome"
        # Task rehearsal accompanies perturbed-state rest optimization.
        QTest.mouseClick(window.rest_button, Qt.MouseButton.LeftButton)
        settle()
        QTest.mouseClick(window.silence, Qt.MouseButton.LeftButton)
        settle()
        QTest.mouseClick(window.predict_button, Qt.MouseButton.LeftButton)
        settle()
        assert window.prediction.text() == "NETWORK OUTPUT   /   Welcome"
        window.response.setPlainText("Welcome")
        window.visualize_text()
        window.send_compute("predict", message="Hello")
        settle()
        app.processEvents()
        assert window.grab().save(str(destination / "learning.png"))
        result = {"status": "passed", "frozen_executable": bool(getattr(sys, "frozen", False)),
            "teacher": professor, "initialization": events[0]["result"],
            "training": [p["result"] for p in events if p["op"] == "train"],
            "rest": [p["result"] for p in events if p["op"] == "rest"],
            "checks": ["full 25^3 native activity", "vertex injection", "silence preserves weights",
                "zero stays zero", "offline professor does not auto-train", "non-copy pair fitting",
                "second pair with first-pair replay", "changed target excluded from prediction",
                "rest training with task replay", "source/package compute bridge", "native screenshot"],
            "limits": "Two taught associations only; no generalization or autonomous reasoning claim"}
        (destination / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        window.compute.close()
        app.exit(0)
    except Exception as error:
        (destination / "failure.txt").write_text(f"{type(error).__name__}: {error}", encoding="utf-8")
        window.compute.close()
        app.exit(1)
