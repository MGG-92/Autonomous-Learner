"""Native widget interactions; optional desktop dependencies, no pretrained model needed."""
import importlib.util
import os
from pathlib import Path
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication


LAB = Path(__file__).resolve().parents[1] / "neural-volume-lab"


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def native(monkeypatch, qapp):
    spec = importlib.util.spec_from_file_location("native_embeddings", LAB / "embeddings.py")
    embedding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(embedding)
    monkeypatch.setitem(sys.modules, "embeddings", embedding)
    spec = importlib.util.spec_from_file_location("native_app", LAB / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    qapp.setStyleSheet(module.STYLE)

    class Store:
        def metadata(self):
            return {"revision": "software-fixture", "dimensions": 6, "scale": 6}

        def encode(self, text):
            if not text.strip(): raise ValueError("Enter a nonempty message")
            if len(text) > 4096: raise ValueError("Use at most 4096 characters")
            return {"text": text, "tokens": [{"id": i, "text": word, "piece": word,
                "vector": [-1, -2, 0, 6, 1, -3], "points": [[-1/6, -2/6, 0], [1, 1/6, -1/2]],
                "padded_components": 0} for i, word in enumerate(text.split())]}

    window = module.Window(Store())
    window.show()
    for _ in range(100):
        qapp.processEvents()
        if window.visualize.isEnabled(): break
        QTest.qWait(10)
    assert window.visualize.isEnabled()
    yield window
    window.close()
    qapp.processEvents()


def test_native_buttons_select_actual_tokens_and_reference_without_training(native, qapp):
    native.message.setPlainText("Hello world")
    native.response.setPlainText("Welcome back")
    QTest.mouseClick(native.visualize, Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert native.volume.tokens[0]["text"] == "Hello"
    assert native.token_text.text() == "'Hello'"
    assert native.volume.tokens[0]["points"] == [[-1/6, -2/6, 0], [1, 1/6, -1/2]]
    QTest.mouseClick(native.next, Qt.MouseButton.LeftButton)
    assert native.index == 1 and native.token_text.text() == "'world'"
    QTest.mouseClick(native.response_tab, Qt.MouseButton.LeftButton)
    assert native.sequence == "response" and native.volume.response
    assert native.token_text.text() == "'Welcome'"
    native.all.setChecked(True)
    assert native.volume.all_tokens and not native.play.isEnabled()
    native.clear()
    assert not native.volume.tokens and not native.inspector.isVisible()
    assert not native.timer.isActive() and not native.response_tab.isEnabled()


def test_playback_stops_at_last_token_and_camera_does_not_change_vectors(native, qapp):
    native.message.setPlainText("one two three")
    native.visualize_text()
    points = [token["points"][:] for token in native.volume.tokens]
    native.toggle_play()
    assert native.timer.isActive()
    native.advance()
    native.advance()
    assert native.index == 2 and not native.timer.isActive()
    native.volume.setFocus()
    yaw = native.volume.yaw
    QTest.keyClick(native.volume, Qt.Key.Key_Right)
    assert native.volume.yaw > yaw
    assert points == [token["points"] for token in native.volume.tokens]
    native.volume.reset_camera()
    assert native.volume.yaw == .55


def test_empty_message_preserves_existing_view_and_displays_error(native):
    native.message.setPlainText("valid input")
    native.visualize_text()
    data = native.data
    native.message.clear()
    native.visualize_text()
    assert native.data is data
    assert native.error.isVisible() and "nonempty" in native.error.text()
