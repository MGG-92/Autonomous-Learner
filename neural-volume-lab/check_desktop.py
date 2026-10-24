"""Explicit desktop verification mode: real model, widgets, rendering and package assets."""
import json
from pathlib import Path
import sys

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest


def verify(window, app, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        window.message.setPlainText("Hello, world. Hello!")
        window.response.setPlainText("Welcome back.")
        QTest.mouseClick(window.visualize, Qt.MouseButton.LeftButton)
        app.processEvents()
        assert [t["id"] for t in window.volume.tokens] == [19556, 28, 905, 30, 38699, 17]
        for sequence in window.data.values():
            for token in sequence["tokens"]:
                assert len(token["vector"]) == 576 and len(token["points"]) == 192
                assert [x / window.metadata["scale"] for x in token["vector"]] == [v for p in token["points"] for v in p]
        QTest.mouseClick(window.next, Qt.MouseButton.LeftButton)
        assert window.index == 1 and window.token_text.text() == "','"
        QTest.mouseClick(window.response_tab, Qt.MouseButton.LeftButton)
        assert window.sequence == "response" and window.volume.response
        QTest.mouseClick(window.message_tab, Qt.MouseButton.LeftButton)
        window.toggle_play()
        assert window.timer.isActive()
        window.advance()
        assert window.index == 1
        window.stop()
        window.all.setChecked(True)
        assert window.volume.all_tokens and not window.play.isEnabled()
        window.all.setChecked(False)
        window.select_manual(0)
        window.volume.setFocus()
        yaw = window.volume.yaw
        QTest.keyClick(window.volume, Qt.Key.Key_Right)
        assert window.volume.yaw > yaw
        window.volume.reset_camera()
        app.processEvents()
        assert window.grab().save(str(destination / "desktop.png"))
        # Check clear/validation after capturing the populated view.
        window.clear()
        assert not window.volume.tokens and not window.inspector.isVisible()
        window.message.clear()
        window.visualize_text()
        assert window.error.isVisible()
        result = {"status": "passed", "frozen_executable": bool(getattr(sys, "frozen", False)),
                  "ui": "native PySide6 QtWidgets and QPainter; no browser or server",
                  "metadata": window.metadata, "tokens": 6, "points_per_token": 192,
                  "checks": ["pinned real token IDs", "all vector components preserved", "fixed scale",
                             "message/reference switching", "single step", "play/pause", "all-token view",
                             "keyboard camera", "clear", "empty input validation", "native screenshot"],
                  "torch_loaded": "torch" in sys.modules, "transformers_loaded": "transformers" in sys.modules}
        (destination / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        app.exit(0)
    except Exception as error:
        (destination / "failure.txt").write_text(f"{type(error).__name__}: {error}", encoding="utf-8")
        app.exit(1)
