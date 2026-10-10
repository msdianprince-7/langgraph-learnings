@echo off
REM Opens the notebooks in this folder with the shared venv, no activation needed.
REM Usage:  run
setlocal
set VENV_PY=%~dp0..\venv\Scripts\python.exe
"%VENV_PY%" -m jupyter notebook --notebook-dir="%~dp0" %*
