@echo off
setlocal
set "LAB_ROOT=%~dp0.."
set "LAB_LAUNCH=%LAB_ROOT%\runs\neural-volume-lab\launch-desktop.cmd"
if exist "%LAB_LAUNCH%" (
 call "%LAB_LAUNCH%" %*
    exit /b
)
if not exist "%LAB_ROOT%\.venv\Scripts\pythonw.exe" (
    echo Desktop runtime not found. Follow neural-volume-lab\README.md setup.
    pause
    exit /b 1
)
start "Neural Volume Lab" "%LAB_ROOT%\.venv\Scripts\pythonw.exe" "%~dp0app.py"
