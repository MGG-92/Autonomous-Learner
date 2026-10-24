"""Native asynchronous bridge to the existing E-drive Python compute runtime."""
import json
import os
from pathlib import Path
import sys

from PySide6.QtCore import QObject, QProcess, Signal


def runtime():
    # Source launch or onedir executable inside this checkout's ignored runs/.
    candidates = [Path(__file__).resolve().parent.parent, *Path(sys.executable).resolve().parents]
    for root in candidates:
        script = root / "neural-volume-lab" / "worker.py"
        python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if script.exists() and python.exists():
            return root, python, script
    raise RuntimeError("Compute runtime missing. Launch from this checkout with its .venv installed.")


class Compute(QObject):
    result = Signal(object)
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = QProcess(self)
        self.buffer = b""
        self.busy = False
        self.closing = False
        self.process.readyReadStandardOutput.connect(self.read)
        # Drain stderr without persisting prompts or filling a pipe.
        self.process.readyReadStandardError.connect(self.process.readAllStandardError)
        self.process.errorOccurred.connect(lambda error: self.fail("Compute process unavailable: " + self.process.errorString()))
        self.process.finished.connect(self.finished)

    def fail(self, message):
        self.busy = False
        self.failed.emit(message)

    def finished(self, code, status):
        if not self.closing:
            self.fail(f"Compute process stopped ({code}); in-memory learning was lost")

    def request(self, op, **arguments):
        if self.busy:
            return False
        if self.process.state() == QProcess.ProcessState.NotRunning:
            try:
                self.buffer = b""
                self.closing = False
                root, python, script = runtime()
                from PySide6.QtCore import QProcessEnvironment
                environment = QProcessEnvironment.systemEnvironment()
                local = root / "runs" / "neural-volume-lab" / "runtime"
                local.mkdir(parents=True, exist_ok=True)
                for key in ("TEMP", "TMP", "TMPDIR", "TORCH_HOME", "XDG_CACHE_HOME"):
                    environment.insert(key, str(local))
                environment.insert("HF_HOME", str(root / "runs" / "models"))
                environment.insert("HF_HUB_OFFLINE", "1")
                environment.insert("TRANSFORMERS_OFFLINE", "1")
                environment.insert("PYTHONDONTWRITEBYTECODE", "1")
                self.process.setProcessEnvironment(environment)
                self.process.setWorkingDirectory(str(root))
                self.process.start(str(python), ["-u", str(script)])
                if not self.process.waitForStarted(3000):
                    raise RuntimeError("Could not start local compute runtime")
            except Exception as error:
                self.fail(str(error))
                return False
        self.busy = True
        self.process.write((json.dumps({"op": op, **arguments}) + "\n").encode())
        return True

    def read(self):
        self.buffer += bytes(self.process.readAllStandardOutput())
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            try:
                payload = json.loads(line)
            except ValueError:
                self.fail("Invalid compute response")
                continue
            self.busy = False
            if "error" in payload:
                self.failed.emit(payload["error"])
            else:
                self.result.emit(payload)

    def close(self):
        self.closing = True
        self.process.closeWriteChannel()
        if not self.process.waitForFinished(1000):
            self.process.kill()
            self.process.waitForFinished(3000)
