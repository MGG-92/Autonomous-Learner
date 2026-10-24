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
    monkeypatch.syspath_prepend(str(LAB))
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
    window.set_expert_mode(True)
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


def test_professor_proposal_is_editable_and_never_automatically_trains(native, monkeypatch):
    requests = []
    monkeypatch.setattr(native.compute, "request", lambda op, **arguments: requests.append((op, arguments)) or True)
    native.send_compute("init")
    native.compute.result.emit({"op": "init", "result": {"active": 0, "peak": 0, "updates": 0, "activity": []}})
    native.message.setPlainText("synthetic question")
    native.send_compute("teacher", message=native.message.toPlainText())
    native.compute.result.emit({"op": "teacher", "result": {"proposal": "<b>unverified</b>"}})
    assert native.response.toPlainText() == "<b>unverified</b>"
    assert "UNVERIFIED" in native.target_source.text()
    assert [op for op, _ in requests] == ["init", "teacher"]
    assert "target vocabulary empty" in native.prediction.text()
    native.response.setPlainText("edited target")
    native.train_pair()
    assert requests[-1] == ("admit", {"message": "synthetic question", "target": "edited target"})
    native.cancel_training()
    assert native.training_batches == 0


def test_inspector_stays_above_transport_even_with_short_volume(native):
    native.message.setPlainText("Hello world")
    native.visualize_text()
    native.position_overlays()
    assert native.inspector.geometry().bottom() < native.transport.geometry().top()


def test_chat_starts_brain_queues_tokens_and_double_click_corrects_exact_reply(native, monkeypatch, qapp):
    native.set_expert_mode(False)
    native.chat.coach.setChecked(False)
    requests = []
    monkeypatch.setattr(native.compute, "request", lambda op, **args: requests.append((op, args)) or True)
    QTest.mouseClick(native.chat.start, Qt.MouseButton.LeftButton)
    assert requests[-1][0] == "init"
    native.compute.result.emit({"op": "init", "result": {"active": 0, "peak": 0, "updates": 0, "activity": []}})
    assert native.brain_running and native.brain_timer.interval() == 5000
    native.chat.duration.setValue(7.5)
    assert native.brain_timer.interval() == 7500
    native.chat.composer.setPlainText("one two")
    QTest.mouseClick(native.chat.send, Qt.MouseButton.LeftButton)
    assert requests[-1] == ("enqueue", {"message": "one two"})
    native.compute.result.emit({"op": "enqueue", "result": {"queued": 2, "message_id": 1}})
    assert not native.volume.tokens and not native.chat.send.isEnabled()
    tick = {"turn_id": 1, "message_id": 1, "prediction": "greeting", "token": {"text": "one", "id": 0},
            "received": 1, "total": 2, "input_finished": False, "active": 1, "peak": .85,
            "activity": [[0, .85]], "updates": 0}
    native.compute.result.emit({"op": "live_tick", "result": tick})
    assert len(native.volume.tokens) == 1
    assert native.chat.latest_reply.text() == "greeting"
    assert native.chat.live.text() == "greeting"
    qapp.processEvents()
    QTest.mouseDClick(native.chat.live, Qt.MouseButton.LeftButton)
    assert native.chat.editor.isVisible() and not native.brain_running
    assert not native.brain_timer.isActive()
    assert native.chat.corrected_text.toPlainText() == "greeting"
    native.chat.corrected_text.setPlainText("Hello")
    QTest.mouseClick(native.chat.save, Qt.MouseButton.LeftButton)
    assert requests[-1] == ("correct", {"turn_id": 1, "target": "Hello"})
    native.compute.result.emit({"op": "correct", "result": {"turn_id": 1, "teaching": {}}})
    assert not native.chat.editor.isVisible()
    assert native.chat.latest_reply.text() == "greeting"  # Edited text is not forged network output.


def test_chat_rating_pauses_timer_and_uses_strong_explicit_feedback(native, monkeypatch):
    native.set_expert_mode(False)
    native.network_ready = True
    native.brain_running = True
    native.chat.queued("hello", {"message_id": 1, "queued": 1})
    native.chat.show_tick({"turn_id": 12, "prediction": "hi", "received": 1, "total": 1, "token": None, "active": 1})
    native.brain_timer.start(5000)
    requests = []
    monkeypatch.setattr(native.compute, "request", lambda op, **args: requests.append((op, args)) or True)
    native.update_compute_controls()
    QTest.mouseClick(native.chat.good, Qt.MouseButton.LeftButton)
    assert requests[-1] == ("feedback", {"turn_id": 12, "reward": 1, "strength": 5})
    assert not native.brain_timer.isActive()
    native.compute.result.emit({"op": "feedback", "result": {"turn_id": 12, "reward": 1, "strength": 5, "update_attempted": True}})
    assert native.brain_timer.isActive()
    assert not native.chat.good.isEnabled() and not native.chat.bad.isEnabled()


def test_professor_runs_once_after_full_sentence_and_remains_optional(native, monkeypatch):
    requests = []
    monkeypatch.setattr(native.compute, "request", lambda op, **args: requests.append((op, args)) or True)
    native.network_ready = True
    tick = {"turn_id": 1, "message_id": 4, "prediction": "hi", "received": 1, "total": 2,
            "input_finished": False, "token": None, "activity": [], "active": 0, "peak": 0, "updates": 0}
    native.compute.result.emit({"op": "live_tick", "result": tick})
    assert requests == []
    tick.update({"received": 2, "input_finished": True})
    native.compute.result.emit({"op": "live_tick", "result": tick})
    assert requests == [("coach", {"turn_id": 1})]
    native.compute.result.emit({"op": "live_tick", "result": tick})
    assert len(requests) == 1
    native.chat.coach.setChecked(False)
    tick["message_id"] = 5
    native.compute.result.emit({"op": "live_tick", "result": tick})
    assert len(requests) == 1


def test_double_click_on_earlier_bubble_does_not_correct_the_latest_reply(native, monkeypatch, qapp):
    native.set_expert_mode(False)
    native.network_ready = True
    native.chat.queued("earlier", {"message_id": 1, "queued": 1})
    native.chat.show_tick({"turn_id": 11, "prediction": "earlier answer", "received": 1, "total": 1, "token": None, "active": 1})
    earlier = native.chat.latest_reply
    native.chat.queued("latest", {"message_id": 2, "queued": 1})
    native.chat.show_tick({"turn_id": 12, "prediction": "latest answer", "received": 1, "total": 1, "token": None, "active": 1})
    requests = []
    monkeypatch.setattr(native.compute, "request", lambda op, **args: requests.append((op, args)) or True)
    native.update_compute_controls()
    qapp.processEvents()
    earlier.edit_requested.emit()
    assert native.chat.edit_turn_id == 11
    assert native.chat.corrected_text.toPlainText() == "earlier answer"
    native.chat.corrected_text.setPlainText("edited earlier answer")
    QTest.mouseClick(native.chat.save, Qt.MouseButton.LeftButton)
    assert requests[-1] == ("correct", {"turn_id": 11, "target": "edited earlier answer"})
