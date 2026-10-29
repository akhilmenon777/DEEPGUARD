@echo off
echo ============================================================
echo  AI Deepfake Detection System Server
echo  Starting on http://localhost:8001
echo ============================================================
echo.
echo Press CTRL+C to stop the server
echo.

cd /d "%~dp0"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8001 --reload

pause
