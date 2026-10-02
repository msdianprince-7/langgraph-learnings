@echo off
REM Starts the Streamlit chatbot with the shared venv, no activation needed.
REM Usage:  run
setlocal
cd /d "%~dp0"
"%~dp0..\venv\Scripts\python.exe" -m streamlit run streamlit_frontend.py %*
