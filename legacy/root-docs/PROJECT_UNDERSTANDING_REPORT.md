### A. PROJECT OVERVIEW
**What this system does:** KAVACH-AI (DeepShield AI) is a real-time deepfake detection and threat intelligence system that operates entirely locally without external API dependencies. 
**Core purpose and use case:** It is designed to identify synthetic media and deepfakes across images, videos, live streams (YouTube, RTSP), and web pages (via browser extension). It provides forensic evidence, risk scoring, and threat monitoring for security personnel.
**Intended user flow:** A security officer or admin logs in to the dashboard. They can initiate manual file scans, connect to live streams, or use a browser extension to analyze content. The backend orchestrator processes the media through multiple AI models concurrently, aggregates the results, and displays actionable verdicts, confidence scores, and Grad-CAM heatmaps to the user, generating immutable alerts for high-risk detections.

### B. TECHNOLOGY STACK
*   **Backend:** Python 3.x, FastAPI (>=0.109.0), Uvicorn, Gunicorn, Celery, Redis.
*   **Frontend:** React 18, Vite, React Router, TailwindCSS, Framer Motion, Recharts.
*   **AI/ML:** PyTorch, Torchvision, OpenCV, HuggingFace Transformers, `timm`. Inference runs on local CPU/CUDA. Training uses PyTorch with AMP and TensorBoard.
*   **Database:** SQLite by default (SQLAlchemy ORM). Key models include `Stream`, `Detection`, `Alert`, `EvidenceChain`, `ScanResult`, and `User`.
*   **Infrastructure:** Docker (Dockerfile and docker-compose.yml available for containerized deployment).

### C. ARCHITECTURE MAP
*   **Communication Layer:** The React frontend communicates with the FastAPI backend via REST APIs (port 8000) and WebSockets (`/ws`) for real-time live detection frames and system alerts.
*   **Data Flow:** Input (Image/Video/Stream/Extension) → FastAPI Route → Orchestrator (`orchestrator.py`) → Cache Lookup → Task Runner (Concurrent execution) → Model Inference (`hf_registry.py` & `ml_pipeline.py`) → Ensemble Aggregator (Temperature Scaling) → Final Verdict & Heatmap → Database/Cache → Output to Client.
*   **File Structure Tree:**
    ```
    deepfake system/
    ├── backend/
    │   ├── ai/               # Model integration, HF registry, video analyzer
    │   ├── api/              # FastAPI routes (auth, alerts, scan, models)
    │   ├── detection/        # Face extraction, GradCAM, ML pipelines
    │   ├── orchestrator/     # Task runner, ensemble aggregator, cache service
    │   └── main.py, database.py, config.py, schemas.py
    ├── frontend/
    │   ├── src/              # React app (components, pages, context, layout)
    │   └── package.json, vite.config.js, tailwind.config.js
    ├── training/             # Deepfake model training scripts (train.py, unified datasets)
    ├── scripts/              # Utility scripts (download data, init DB, auth tests)
    ├── extension/            # Browser extension plugin files
    └── requirements.txt, Dockerfile, docker-compose.yml
    ```

### D. AI/ML COMPONENT ANALYSIS
*   **Models Currently Used/Referenced:**
    1.  `vit_deepfake_primary` (prithivMLmods/Deepfake-image-detect)
    2.  `vit_deepfake_secondary` (dima806/deepfake_vs_real_image_detection)
    3.  `efficientnet_b4` (timm ImageNet pretrained)
    4.  `xception` (timm ImageNet pretrained)
    5.  `frequency_dct` & `frequency_fft` (Heuristic frequency domain analysis)
*   **Implementation:** Media undergoes face extraction (MediaPipe/Haar), is resized to 224x224, and analyzed concurrently across selected model tiers. An ensemble aggregator uses temperature scaling and custom model weights to fuse the outputs. Explainability is provided via Grad-CAM heatmaps.
*   **Training Status:** HuggingFace ViT models are pre-trained on deepfake datasets. EfficientNet and Xception default to ImageNet weights unless fine-tuned `.pth` weights are present in the `models/` directory.
*   **Data Pipelines:** Training pipelines (`train.py`) support FaceForensics++, Celeb-DF v2, DFDC, and WildDeepfake datasets.

### E. IDENTIFIED ISSUES (Observations only)
*   **Backend:** The `/auth/register` endpoint allows direct admin creation without an authorization guard (noted as "Delete in prod"). The `/alerts/{alert_id}/evidence/export` endpoint is currently a stub (unimplemented).
*   **Frontend:** The presence of `build_error.log`, `build_error_5.log`, and `lint_err.txt` indicates lingering compilation, dependency, or linting errors (likely unused imports or unresolved component paths).
*   **ML Pipeline:** Concurrent execution of multiple deep neural networks on CPU (via `task_runner`) risks extreme latency and potential out-of-memory (OOM) errors in production if not strictly managed. Duplication exists between older pipeline structures (`detection/ml_pipeline.py`) and the newer orchestration layer (`orchestrator/`).
*   **Code Quality:** Logging is robust (`loguru`), but multiple legacy files (e.g., `main_legacy.py`, `database_unused/`) clutter the active architecture.

### F. SUMMARY TABLE

| Area | Status | Key Findings | Priority |
| :--- | :--- | :--- | :--- |
| **Backend** | Functional / Incomplete | Robust Orchestrator and WebSockets. Unsecured `/auth/register` and unimplemented evidence export endpoint. Dead code present. | High |
| **Frontend** | Functional / Errors Logged | Built with React/Vite. Comprehensive dashboard available but significant build/lint error logs exist in the directory. | Medium |
| **ML Pipeline** | Advanced orchestration | Concurrent HF ViT + Timm execution + Ensembling. High risk of severe CPU bottlenecks if run without GPU. | High |
| **Database** | Setup complete | SQLite with SQLAlchemy. Well-structured immutable schemas for streams, detections, and alerts. | Low |
| **Security** | Vulnerable | Missing prod-level JWT guard on admin registration. CORS is overly permissive (`"*"`). | High |
| **Deployment** | Partial | Docker setup exists but requires full environment testing and frontend static serving configuration. | Medium |
