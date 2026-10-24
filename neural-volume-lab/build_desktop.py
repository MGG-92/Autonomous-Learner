"""Build an offline Windows application into ignored runs; no model downloads."""
from pathlib import Path
import os
import subprocess
import sys


def build():
    lab = Path(__file__).resolve().parent
    output = lab.parent / "runs" / "neural-volume-lab"
    model = output / "model"
    if not (model / "manifest.json").exists():
        raise RuntimeError("Prepare local model assets with prepare_model.py first")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QColor, QFont, QImage, QPainter
    from PySide6.QtCore import QRectF, Qt
    app = QApplication.instance() or QApplication([])
    icon = QImage(256, 256, QImage.Format.Format_ARGB32)
    icon.fill(Qt.GlobalColor.transparent)
    painter = QPainter(icon)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#10232c"))
    painter.drawRoundedRect(QRectF(8, 8, 240, 240), 52, 52)
    painter.setPen(QColor("#98e6cf"))
    painter.setFont(QFont("Segoe UI", 104, QFont.Weight.Light))
    painter.drawText(icon.rect(), Qt.AlignmentFlag.AlignCenter, "nv")
    painter.end()
    icon_path = output / "app.ico"
    if not icon.save(str(icon_path), "ICO"):
        raise RuntimeError("Could not generate the application icon")
    mode = "--console" if "--console" in sys.argv else "--windowed"
    build_env = dict(os.environ)
    if sys.platform == "win32":
        # Unrelated toolchains on PATH can supply incompatible same-named ICU DLLs.
        # Use this environment and the OS DLL directory for dependency resolution.
        import PySide6
        system = Path(os.environ["SystemRoot"])
        build_env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent),
            str(Path(PySide6.__file__).parent), str(system / "System32"), str(system)])
    subprocess.run([sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm", mode, "--onedir",
                    "--icon", str(icon_path),
                    "--name", "NeuralVolumeLab", "--paths", str(lab),
                    "--distpath", str(output / "desktop"), "--workpath", str(output / "build"),
                    "--specpath", str(output / "build"), "--add-data", f"{model};model",
                    "--exclude-module", "torch", "--exclude-module", "transformers",
                    "--exclude-module", "PySide6.QtWebEngineCore", "--exclude-module", "PySide6.QtWebEngineWidgets",
                    str(lab / "app.py")], check=True, env=build_env)


if __name__ == "__main__":
    build()
