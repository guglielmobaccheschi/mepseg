@echo off
rem ============================================================
rem  mepseg — avvio rapido dell'interfaccia grafica (Windows)
rem  Doppio click su questo file: il browser si apre da solo.
rem ============================================================
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m mepseg.gui.server
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python non trovato. Installare Python 3.9+ da python.org
        echo e poi eseguire:  pip install -e .[gui]
        pause
        exit /b 1
    )
    python -m mepseg.gui.server
)

if errorlevel 1 (
    echo.
    echo L'avvio e' fallito. Verificare l'installazione:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -e .[gui,e57]
    pause
)
