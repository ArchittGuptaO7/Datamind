@echo off
REM One-click launcher for Windows: creates the venv on first run, installs packages, starts the app.
cd /d "%~dp0"
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv || py -m venv venv
)
call venv\Scripts\activate
pip install -q -r requirements.txt
streamlit run app.py
pause
