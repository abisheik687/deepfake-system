"""
KAVACH-AI Project Setup Script
Automates initial project setup with NO manual API configuration required
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def print_header(message):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"  {message}")
    print(f"{'='*60}\n")


def run_command(cmd, description):
    """Run shell command with error handling"""
    print(f"▶ {description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"  ✓ {description} complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error in {description}")
        print(f"  Error: {e.stderr}")
        return False


def main():
    print_header("KAVACH-AI Setup - Real-Time Deepfake Detection")
    
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print(f"Project directory: {project_root}")
    
    # Step 1: Check Python version
    print_header("Step 1: Checking Python Version")
    py_version = sys.version_info
    if py_version.major >= 3 and py_version.minor >= 10:
        print(f"  ✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        print(f"  ✗ Python 3.10+ required, found {py_version.major}.{py_version.minor}")
        sys.exit(1)
    
    # Step 2: Create .env file
    print_header("Step 2: Creating Environment Configuration")
    if not (project_root / ".env").exists():
        shutil.copy(project_root / ".env.example", project_root / ".env")
        print("  ✓ Created .env from .env.example")
        print("  ℹ All settings use local processing - NO API keys needed!")
    else:
        print("  ℹ .env already exists, skip...")
    
    # Step 3: Create virtual environment
    print_header("Step 3: Setting Up Virtual Environment")
    if not (project_root / "venv").exists():
        run_command(f"{sys.executable} -m venv venv", "Creating virtual environment")
    else:
        print("  ℹ Virtual environment already exists")
    
    # Step 4: Activate and install dependencies
    print_header("Step 4: Installing Python Dependencies")
    
    # Determine activation script based on OS
    if os.name == 'nt':  # Windows
        activate_cmd = str(project_root / "venv" / "Scripts" / "activate.bat")
        pip_cmd = str(project_root / "venv" / "Scripts" / "pip.exe")
    else:  # Unix/Linux/Mac
        activate_cmd = f"source {project_root / 'venv' / 'bin' / 'activate'}"
        pip_cmd = str(project_root / "venv" / "bin" / "pip")
    
    # Upgrade pip
    run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    run_command(
        f"{pip_cmd} install -r {project_root / 'requirements.txt'}",
        "Installing dependencies (this may take a few minutes)")
    
    # Step 5: Create necessary directories
    print_header("Step 5: Creating Project Directories")
    directories = [
        "data",
        "models",
        "evidence",
        "logs",
        "backend/ingestion",
        "backend/features",
        "backend/models",
        "backend/threat",
        "backend/forensics",
        "backend/alerts",
        "backend/websocket",
        "scripts",
        "tests"
    ]
    
    for dir_path in directories:
        (project_root / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py for Python packages
        if dir_path.startswith("backend/") and not dir_path.endswith("/"):
            init_file = project_root / dir_path / "__init__.py"
            if not init_file.exists():
                init_file.write_text('"""Package initialization"""\n')
    
    print("  ✓ All directories created")
    
    # Step 6: Initialize database
    print_header("Step 6: Initializing Database")
    print("  ℹ Database will be initialized on first API startup")
    
    # Step 7: Check for FFmpeg
    print_header("Step 7: Checking External Dependencies")
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True
        )
        print("  ✓ FFmpeg is installed")
    except FileNotFoundError:
        print("  ⚠ FFmpeg not found - required for video processing")
        print("  Install FFmpeg:")
        print("  - Windows: Download from https://ffmpeg.org/download.html")
        print("  - Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  - macOS: brew install ffmpeg")
    
    # Step 8: Summary
    print_header("Setup Complete!")
    print("✅ KAVACH-AI is ready to run!")
    print("\n📋 Next Steps:")
    print("\n1. Activate virtual environment:")
    if os.name == 'nt':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("\n2. Start the backend server:")
    print("   uvicorn backend.main:app --reload")
    
    print("\n3. Access the API:")
    print("   - API Docs: http://localhost:8000/docs")
    print("   - Health Check: http://localhost:8000/health")
    
    print("\n4. (Optional) Start Redis for background tasks:")
    print("   redis-server")
    
    print("\n5. (Optional) Run with Docker:")
    print("   docker-compose up --build")
    
    print("\n🛡️  NO API KEYS REQUIRED - All processing is local!")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
