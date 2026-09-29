@echo off
rem ============================================================
rem  mepseg - avvio rapido dell'interfaccia grafica (Windows)
rem  Doppio click su questo file: il browser si apre da solo.
rem  Al primo avvio crea l'ambiente .venv e installa le
rem  dipendenze (qualche minuto, serve la connessione).
rem ============================================================
setlocal
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"
set "MARCATORE=.venv\installazione_completata.txt"

if exist "%MARCATORE%" goto avvia

rem ---- primo avvio (o installazione precedente interrotta) ----
echo.
echo Primo avvio di mepseg: preparo l'ambiente Python.
echo L'operazione richiede qualche minuto e la connessione a internet.
echo.

if exist "%VENV_PY%" goto installa

rem Cerca Python: prima il launcher "py", poi "python".
rem (il test con -c evita il finto python.exe del Microsoft Store)
set "PY_BASE="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul && set "PY_BASE=py -3"
if not defined PY_BASE python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul && set "PY_BASE=python"
if not defined PY_BASE goto no_python

echo Creo l'ambiente virtuale in .venv ...
%PY_BASE% -m venv .venv
if not exist "%VENV_PY%" goto errore_venv

:installa
echo Installo le dipendenze ...
"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install -e ".[gui,e57]"
if errorlevel 1 goto errore_pip
echo ok> "%MARCATORE%"
echo.
echo Installazione completata.
echo.

:avvia
echo Avvio dell'interfaccia: il browser si aprira' tra poco.
echo Per chiudere mepseg chiudere questa finestra.
"%VENV_PY%" -m mepseg.gui.server %*
if errorlevel 1 goto errore_avvio
goto fine

:no_python
echo Python 3.9 o superiore non trovato.
echo Installarlo da https://www.python.org/downloads/
echo spuntando "Add python.exe to PATH", poi rilanciare questo file.
pause
exit /b 1

:errore_venv
echo Impossibile creare l'ambiente virtuale .venv
pause
exit /b 1

:errore_pip
echo.
echo Installazione delle dipendenze fallita: vedere i messaggi sopra.
echo Controllare la connessione e rilanciare questo file;
echo l'installazione riprendera' da dove si e' fermata.
pause
exit /b 1

:errore_avvio
echo.
echo L'avvio e' fallito: vedere i messaggi sopra.
echo Per reinstallare da zero cancellare la cartella .venv e rilanciare.
pause
exit /b 1

:fine
endlocal
