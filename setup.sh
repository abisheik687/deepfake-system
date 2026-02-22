#!/bin/bash
# KAVACH-AI Setup Script for Linux/macOS
# NO API KEYS REQUIRED - All processing is local

echo "============================================================"
echo "  KAVACH-AI Setup - Real-Time Deepfake Detection"
echo "============================================================"
echo ""

echo "Step 1: Check Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found!"
    echo "Please install Python 3.10+"
    exit 1
fi
python3 --version
echo ""

echo "Step 2: Creating environment file..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example"
else
    echo ".env already exists, skipped"
fi
echo ""

echo "Step 3: Creating virtual environment..."
if [ ! -d venv ]; then
    python3 -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi
echo ""

echo "Step 4: Activating virtual environment..."
source venv/bin/activate
echo ""

echo "Step 5: Upgrading pip..."
pip install --upgrade pip
echo ""

echo "Step 6: Installing dependencies..."
echo "This may take a few minutes..."
pip install -r requirements.txt
echo ""

echo "Step 7: Creating project directories..."
mkdir -p data models evidence logs
mkdir -p backend/{ingestion,features,models,threat,forensics,alerts,websocket}
mkdir -p scripts tests
echo "Directories created"
echo ""

echo "Step 8: Checking FFmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    echo "WARNING: FFmpeg not found!"
    echo "Install with:"
    echo "  - Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "  - macOS: brew install ffmpeg"
else
    echo "FFmpeg is installed"
    ffmpeg -version | head -n 1
fi
echo ""

echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo ""
echo "Next Steps:"
echo ""
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Start the backend server:"
echo "   uvicorn backend.main:app --reload"
echo ""
echo "3. Access the API:"
echo "   - API Docs: http://localhost:8000/docs"
echo "   - Health: http://localhost:8000/health"
echo ""
echo "4. (Optional) Start with Docker:"
echo "   docker-compose up --build"
echo ""
echo "🛡️  NO API KEYS REQUIRED - All processing is local!"
echo ""
echo "============================================================"
