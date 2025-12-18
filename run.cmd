@echo off
setlocal
cd /d "%~dp0"

REM Ensure venv Python exists
if not exist "rag\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found at rag\Scripts\python.exe
  echo Create the venv or adjust the path, then re-run.
  exit /b 1
)

REM Default models if not provided by the environment
if not defined OLLAMA_MODEL set OLLAMA_MODEL=llama3.2:latest
if not defined EMBED_MODEL set EMBED_MODEL=nomic-embed-text

REM Launch Streamlit via venv Python (works without activate)
"rag\Scripts\python.exe" -m streamlit run app\minimal.py %*

endlocal
