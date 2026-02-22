
<div align="center">

<img src="assets/kavach_ai_banner.png" alt="KAVACH-AI — Multimodal Deepfake Detection & Forensic Analysis" width="100%" />

# 🛡️ KAVACH-AI
### Multimodal Deepfake Detection & Forensic Analysis Platform

[![Status](https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge)](https://github.com/abisheik687/deepfake-system)
[![Version](https://img.shields.io/badge/Version-1.0.0-blue?style=for-the-badge)](https://semver.org)
[![Python](https://img.shields.io/badge/Python-3.9%2B-yellow?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

[🚀 Request Demo](https://kavach-ai.demo) · [🐛 Report Bug](https://github.com/abisheik687/deepfake-system/issues) · [✨ Request Feature](https://github.com/abisheik687/deepfake-system/issues)

</div>

---

## 📖 Overview

**KAVACH-AI** *(Protection AI — कवच)* is an enterprise-grade forensic intelligence platform engineered to detect and analyze AI-generated synthetic media (deepfakes). Powered by a sophisticated **Multi-Modal Fusion Engine**, the system simultaneously interrogates video streams for spatial artifacts, audio tracks for spectral inconsistencies, and temporal sequences for unnatural motion patterns — producing a single, high-confidence forensic verdict.

Built for **law enforcement agencies**, **digital forensic laboratories**, and **media integrity units**, KAVACH-AI provides:

- 🔬 **High-Confidence Verdicts** — By triangulating data from visual, audio, and temporal AI models through weighted ensemble fusion.
- ⛓️ **Chain of Custody** — Blockchain-ready cryptographic hashing and immutable audit logs for court-admissible evidence.
- 📡 **Real-Time Surveillance** — RTSP stream ingestion for live CCTV and broadcast threat detection.
- 📄 **Automated PDF Reporting** — Forensic-grade reports with timestamps, confidence scores, and evidence hashes.

---

## ⚡ Key Features

| Feature | Description |
| :--- | :--- |
| **🎥 Multi-Modal Detection** | Simultaneous analysis of *MesoNet* (Video), *Mel-Spectrogram* (Audio), and *LSTM* (Temporal) models. |
| **🧠 AI Fusion Engine** | Weighted voting ensemble that significantly reduces false positives vs. single-model approaches. |
| **📊 Forensic Dashboard** | Interactive visualization of confidence scores, frame-by-frame anomalies, and modality heatmaps. |
| **⚡ Real-Time Monitor** | Live CCTV interface with bounding-box inference and scrolling threat alert logs. |
| **🔒 JWT Authentication** | Role-based access control (RBAC) with secure token-based authentication for labs and analysts. |
| **📂 Report Generation** | Automated PDF export with full forensic details, timestamps, and evidence integrity hashes. |
| **🐳 Dockerized Microservices** | Fully containerized architecture for reproducible, "one-click" deployment in any environment. |
| **📡 RTSP Stream Support** | Live stream ingestion for real-time deepfake detection in video surveillance pipelines. |

---

## 🏗️ System Architecture

KAVACH-AI follows a modular, event-driven microservices architecture to ensure scalability, resilience, and maintainability.

```mermaid
graph TD
    subgraph "Client Layer"
        User[Forensic Analyst]
        Browser[React 18 Frontend]
    end

    subgraph "Application Layer"
        API[FastAPI Gateway :8000]
        Auth[JWT Auth Service]
        Worker[Celery AI Worker]
    end

    subgraph "Data Layer"
        DB[(PostgreSQL :5432)]
        Redis[Redis Cache & Queue :6379]
        Storage[Evidence File Storage]
    end

    subgraph "AI Core Engine"
        Video[MesoNet — Video Model]
        Audio[Mel-Spec — Audio Model]
        Temporal[LSTM — Temporal Model]
        Fusion[Weighted Fusion Engine]
    end

    User -->|HTTPS| Browser
    Browser -->|REST API| API
    API -->|Verify| Auth
    API -->|Store Metadata| DB
    API -->|Enqueue Task| Redis
    Redis -->|Process Async| Worker
    Worker -->|Inference| Video & Audio & Temporal
    Video & Audio & Temporal -->|Results| Fusion
    Fusion -->|Final Verdict| DB
    DB -->|Report| Browser
```

---

## 📂 Project Structure

```bash
kavach-ai/
├── assets/                  # Project branding & media assets
│   └── kavach_ai_banner.png # Project logo & banner
├── backend/                 # Python/FastAPI Backend Service
│   ├── api/                 # REST API Routes & Controllers
│   │   ├── auth.py          # JWT Authentication endpoints
│   │   ├── scan.py          # File upload & analysis endpoints
│   │   └── reports.py       # Forensic report generation
│   ├── core/                # Configuration & Security utilities
│   ├── database/            # SQLAlchemy ORM Models & CRUD operations
│   ├── models/              # PyTorch AI Models (MesoNet, LSTM)
│   ├── worker.py            # Celery Async Task Definitions
│   └── main.py              # FastAPI Application Entrypoint
├── frontend/                # React 18 / Vite Frontend
│   ├── src/
│   │   ├── api/             # Axios API Client & interceptors
│   │   ├── components/      # Reusable UI Components
│   │   ├── layout/          # Dashboard shell & navigation layouts
│   │   ├── pages/           # Feature Views (Analysis, Monitor, Upload)
│   │   └── context/         # Global State Management (Auth Context)
│   └── Dockerfile           # Multi-stage Nginx production build
├── scripts/                 # Utility & setup scripts
├── tests/                   # Automated test suites
├── data/                    # Storage for uploads, DB files & evidence
├── docker-compose.yml       # Full-stack service orchestration
├── requirements.txt         # Python dependency manifest
├── setup.sh                 # Linux/macOS setup script
├── setup.bat                # Windows setup script
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- **Docker** (v24+) & **Docker Compose** (v2+)
- *Or* **Python** 3.9+ & **Node.js** 18+ *(for manual setup)*
- Minimum **8GB RAM** recommended for running AI inference models

---

### 🐳 Option A: Docker Deployment *(Recommended)*

The fastest way to get KAVACH-AI running with all services.

**1. Clone the Repository**
```bash
git clone https://github.com/abisheik687/deepfake-system.git
cd deepfake-system
```

**2. Configure Environment**
```bash
cp .env.example .env
# Edit .env to set your SECRET_KEY, database credentials, etc.
```

**3. Launch All Services**
```bash
docker-compose up --build -d
```

**4. Access the Platform**

| Service | URL |
|---|---|
| 🌐 Frontend Dashboard | [http://localhost:3000](http://localhost:3000) |
| ⚙️ API Documentation | [http://localhost:8000/docs](http://localhost:8000/docs) |
| 🔑 Default Login | `admin` / `admin` |

---

### 🔧 Option B: Manual Installation

<details>
<summary>Click to expand manual setup instructions</summary>

#### 1. Backend (FastAPI + Python)
```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r ../requirements.txt
uvicorn main:app --reload --port 8000
```

#### 2. Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev
```

#### 3. Required Services
Ensure the following services are running before starting:

| Service | Default Port | Purpose |
|---|---|---|
| PostgreSQL | 5432 | Primary relational database |
| Redis | 6379 | Task queue & caching layer |

#### 4. Database Initialization
```bash
# Run migrations
alembic upgrade head

# Create default admin user
python scripts/create_admin.py
```

</details>

---

## 📚 API Reference

The API is fully documented using **OpenAPI (Swagger UI)** at `http://localhost:8000/docs`.

### Authentication
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/auth/token` | Authenticate and retrieve a JWT access token |
| `POST` | `/auth/register` | Register a new analyst account |

### Scan & Analysis
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/scan/upload` | Upload video or audio evidence (supports chunked upload) |
| `GET` | `/scan/{id}` | Retrieve analysis results and confidence scores |
| `GET` | `/scan/history` | List all past scans for the current authenticated user |
| `DELETE` | `/scan/{id}` | Delete a scan record and its associated evidence |

### Reports
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/reports/{scan_id}/pdf` | Download a forensic-grade PDF report for a scan |

---

## 🧠 AI Models

KAVACH-AI employs a three-stage AI pipeline for comprehensive media analysis:

| Model | Modality | Architecture | Purpose |
|---|---|---|---|
| **MesoNet** | Video / Image | CNN | Detects facial manipulation artifacts in individual frames |
| **Mel-Spectrogram** | Audio | CNN + SpecAugment | Identifies GAN-synthesized voice patterns and spectral anomalies |
| **LSTM Temporal** | Temporal | Bi-LSTM | Catches inconsistent blinking, gaze, and micro-expression sequences |
| **Fusion Engine** | All | Weighted Ensemble | Combines all model outputs into a final weighted confidence verdict |

---

## 🤝 Contributing

Contributions from the security research and AI community are welcome! Please review our [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a pull request.

1. 🍴 **Fork** the repository
2. 🌿 **Create** your feature branch (`git checkout -b feature/YourFeatureName`)
3. ✅ **Commit** your changes (`git commit -m 'feat: Add YourFeatureName'`)
4. 📤 **Push** to the branch (`git push origin feature/YourFeatureName`)
5. 🔁 **Open a Pull Request** with a clear description of your changes

---

## 🛡️ Security

If you discover a security vulnerability, please **do not open a public issue**. Instead, report it responsibly via the [GitHub Security Advisory](https://github.com/abisheik687/deepfake-system/security/advisories) feature.

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

<div align="center">
  <p>Built with ❤️ for digital truth and media integrity by the <strong>KAVACH-AI Team</strong></p>
  <p>
    <a href="#">Privacy Policy</a> ·
    <a href="#">Terms of Service</a> ·
    <a href="https://github.com/abisheik687/deepfake-system">GitHub Repository</a>
  </p>
  <br/>
  <sub>⭐ If this project helps your research, please consider giving it a star!</sub>
</div>
