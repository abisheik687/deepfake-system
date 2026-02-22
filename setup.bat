@echo off
REM KAVACH-AI Setup Script for Windows
REM NO API KEYS REQUIRED - All processing is local

echo ============================================================
echo   KAVACH-AI Setup - Real-Time Deepfake Detection
echo ============================================================
echo.

echo Step 1: Check Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
)
python --version
echo.

echo Step 2: Creating environment file...
if not exist .env (
    copy .env.example .env
    echo Created .env from .env.example
) else (
    echo .env already exists, skipped
)
echo.

echo Step 3: Creating virtual environment...
if not exist venv (
    python -m venv venv
    echo Virtual environment created
) else (
    echo Virtual environment already exists
)
echo.

echo Step 4: Activating virtual environment...
call venv\Scripts\activate.bat
echo.

echo Step 5: Upgrading pip...
python -m pip install --upgrade pip
echo.

echo Step 6: Installing dependencies...
echo This may take a few minutes...
pip install -r requirements.txt
echo.

echo Step 7: Creating project directories...
mkdir data 2>nul
mkdir models 2>nul
mkdir evidence 2>nul
mkdir logs 2>nul
mkdir backend\ingestion 2>nul
mkdir backend\features 2>nul
mkdir backend\models 2>nul
mkdir backend\threat 2>nul
mkdir backend\forensics 2>nul
mkdir backend\alerts 2>nul
mkdir backend\websocket 2>nul
mkdir scripts 2>nul
mkdir tests 2>nul
echo Directories created
echo.

echo Step 8: Checking FFmpeg...
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: FFmpeg not found!
    echo Download from: https://ffmpeg.org/download.html
    echo Required for video processing
) else (
    echo FFmpeg is installed
)
echo.

echo ============================================================
echo   Setup Complete!
echo ============================================================
echo.
echo Next Steps:
echo.
echo 1. Activate virtual environment (already activated):
echo    venv\Scripts\activate.bat
echo.
echo 2. Start the backend server:
echo    uvicorn backend.main:app --reload
echo.
echo 3. Access the API:
echo    - API Docs: http://localhost:8000/docs
echo    - Health: http://localhost:8000/health
echo.
echo 4. (Optional) Start with Docker:
echo    docker-compose up --build
echo.
echo NO API KEYS REQUIRED - All processing is local!
echo.
echo ============================================================
pause
