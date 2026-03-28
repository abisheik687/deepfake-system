#!/bin/bash
# KAVACH-AI Setup Script v2.0 — World-Class Deepfake Detection
# NO API KEYS REQUIRED - All processing is local

echo "============================================================"
echo "  KAVACH-AI v2.0 - Mission Control Setup"
echo "============================================================"

# 1. Environment Verification
echo "Step 1: Check Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found! Please install Python 3.10+"
    exit 1
fi

# 2. Config & Env
echo "Step 2: Initializing environment configuration..."
if [ ! -f .env ]; then
    cp .env.example .env 2>/dev/null || echo "DATABASE_URL=postgresql+asyncpg://kavach:kavach@postgres:5432/kavach" > .env
    echo "Created .env configuration"
fi

# 3. Virtual Environment & Dependencies
echo "Step 3: Preparing Python environment..."
if [ ! -d venv ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Directory Structure (Architecture v2.0)
echo "Step 4: Creating World-Class directory structure..."
mkdir -p data models/checkpoints evidence/reports logs infra/k8s docs
mkdir -p backend/{api,orchestrator,detection,agents,database}
mkdir -p frontend/src/{components,pages,hooks}
mkdir -p training/{datasets,checkpoints}

# 5. Native OS Dependencies
echo "Step 5: Checking FFmpeg & Forensic tools..."
if ! command -v ffmpeg &> /dev/null; then
    echo "WARNING: FFmpeg not found! Critical for SyncNet/OpticalFlow."
fi

echo "============================================================"
echo "  KAVACH-AI v2.0 READY"
echo "============================================================"
echo ""
echo "Orchestration Commands:"
echo "----------------------"
echo "1. Start Full Stack (Docker):"
echo "   docker compose up --build"
echo ""
echo "2. Start Backend Node (Local):"
echo "   uvicorn backend.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "3. Run Adversarial Robustness Audit:"
echo "   python training/adversarial_test.py"
echo ""
echo "4. Access Dashboards:"
echo "   - Main Dashboard: http://localhost:3000"
echo "   - API Mission Control: http://localhost:8000/docs"
echo "   - MLOps Monitor (Grafana): http://localhost:3001"
echo "   - Model Registry (MLflow): http://localhost:5000"
echo ""
echo "============================================================"
