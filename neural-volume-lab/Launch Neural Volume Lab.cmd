@echo off
setlocal
set "LAB_ROOT=%~dp0.."
set "LAB_EXE=%LAB_ROOT%\runs\neural-volume-lab\desktop\NeuralVolumeLab\NeuralVolumeLab.exe"
if exist "%LAB_EXE%" (
    start "Neural Volume Lab" "%LAB_EXE%"
    exit /b
)
if not exist "%LAB_ROOT%\.venv\Scripts\pythonw.exe" (
    echo Desktop runtime not found. Follow neural-volume-lab\README.md setup.
    pause
    exit /b 1
)
start "Neural Volume Lab" "%LAB_ROOT%\.venv\Scripts\pythonw.exe" "%~dp0app.py"
