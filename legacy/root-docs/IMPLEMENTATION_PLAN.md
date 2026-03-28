# KAVACH-AI Implementation Plan
**Complete Deepfake Detection Platform - Production Ready**

---

## 📊 Current State Analysis

### ✅ **COMPLETED COMPONENTS**

#### Backend Infrastructure (80% Complete)
- ✅ FastAPI application with lifespan management
- ✅ Authentication system (JWT + refresh tokens)
- ✅ Database models (SQLAlchemy + Alembic migrations)
- ✅ CRUD operations for detections, alerts, users
- ✅ WebSocket endpoint for real-time updates
- ✅ Rate limiting and CORS middleware
- ✅ Prometheus metrics instrumentation
- ✅ Logging with Loguru
- ✅ Configuration management with Pydantic Settings

#### ML/AI Pipeline (70% Complete)
- ✅ Orchestrator system with model registry
- ✅ Ensemble aggregator with weighted soft voting
- ✅ Temperature scaling for calibration
- ✅ Cache service for performance
- ✅ Task runner for concurrent model execution
- ✅ Face detection pipeline (MTCNN/MediaPipe)
- ✅ Feature extraction (spatial, temporal)
- ✅ Inference service with ONNX Runtime
- ✅ GradCAM heatmap generation
- ✅ AI Agency system (fact-checker, forensic analyst, journalist)

#### API Endpoints (75% Complete)
- ✅ `/api/auth/*` - Authentication (login, refresh, register)
- ✅ `/api/scan/analyze-unified` - Image/video analysis
- ✅ `/api/scan/live-unified` - Live frame analysis
- ✅ `/api/scan/extension-scan` - Browser extension support
- ✅ `/api/detections/*` - Detection history and stats
- ✅ `/api/alerts/*` - Alert management
- ✅ `/api/models/*` - Model registry
- ✅ `/api/agency/*` - AI agency endpoints
- ✅ `/health` - Health check

#### Frontend (60% Complete)
- ✅ React 18 + Vite setup
- ✅ TailwindCSS styling
- ✅ Authentication context
- ✅ API service layer (Axios)
- ✅ Dashboard page with stats
- ✅ HomePage with hero section
- ✅ LoginPage
- ✅ MainLayout with navigation
- ✅ Reusable components (VerdictBadge, StatsRow, etc.)

#### DevOps (90% Complete)
- ✅ Docker + docker-compose setup
- ✅ Multi-service architecture (backend, frontend, postgres, redis, minio, kafka)
- ✅ GPU support profile
- ✅ Health checks for all services
- ✅ Volume persistence
- ✅ Prometheus + Grafana monitoring
- ✅ MLflow model registry

---

## 🔴 **MISSING COMPONENTS & GAPS**

### Critical Missing Features

#### 1. **Audio Deepfake Detection (Module C)** ❌
**Status:** Partially implemented, needs completion
**Files to Create/Update:**
- `backend/routers/audio.py` - Audio upload endpoint
- `backend/ml/audio_model.py` - RawNet2/LCNN implementation
- `backend/services/audio_processor.py` - Mel-spectrogram extraction
**Requirements:**
- Accept MP3/WAV/OGG up to 50MB
- Librosa-based spectrogram extraction
- RawNet2 + LCNN binary classifier
- Return: authenticity score, spectrogram visualization, anomaly regions

#### 2. **Social Media URL Fetching (Module D)** ❌
**Status:** Not implemented
**Files to Create:**
- `backend/routers/social.py` - Social media scanning endpoint
- `backend/services/media_fetcher.py` - yt-dlp integration (EXISTS but needs enhancement)
- `backend/services/celery_tasks.py` - Background job processing (EXISTS but needs social media tasks)
**Requirements:**
- Support Twitter/X, YouTube, Instagram, TikTok, Facebook URLs
- yt-dlp for media extraction
- Celery queue for batch processing
- WebSocket progress updates
- Risk labels (LOW/MEDIUM/HIGH/CRITICAL)

#### 3. **Live Video Call Detection (Module E)** ❌
**Status:** Backend exists, frontend missing
**Files to Create:**
- `frontend/src/pages/LiveVideoPage.jsx` - WebRTC video capture
- `frontend/src/hooks/useWebRTC.js` - WebRTC hook
- `backend/routers/live.py` - WebSocket endpoint for live video (needs enhancement)
**Requirements:**
- WebRTC getUserMedia integration
- 2-second frame capture interval
- MobileNetV3 lightweight model (≤50ms inference)
- Real-time overlay (GREEN=real, RED=fake)
- Session log with timestamps

#### 4. **Voice Call Detection (Module F)** ❌
**Status:** Not implemented
**Files to Create:**
- `frontend/src/pages/LiveAudioPage.jsx` - Audio stream capture
- `backend/routers/audio_stream.py` - WebSocket audio endpoint
**Requirements:**
- MediaRecorder for 2-second audio chunks
- Base64 encoding for WebSocket transmission
- Per-chunk REAL/FAKE labels
- Alert banner for 3 consecutive fake chunks (>0.75)

#### 5. **Interview Proctoring Mode (Module G)** ❌
**Status:** Not implemented
**Files to Create:**
- `frontend/src/pages/InterviewPage.jsx` - Dual-panel UI
- `backend/routers/interview.py` - Combined video+audio endpoint
- `backend/services/report_generator.py` - PDF report generation (EXISTS but needs interview template)
**Requirements:**
- Simultaneous video + audio analysis
- Dual-panel UI (video feed + audio waveform)
- Integrity score (60% video, 40% audio)
- Auto-generate PDF report with timeline

#### 6. **Frontend Pages** ❌
**Missing Pages:**
- `/scanner` - Unified upload interface with drag-and-drop
- `/live-detection` - Tab switcher (Video Call | Voice Call | Interview)
- `/social-media` - URL input + scan queue table
- `/history` - Paginated scan history with filters
- `/models` - Model registry management UI
- `/settings` - User profile and preferences

#### 7. **Training Scripts** ❌
**Status:** Partial (some scripts exist in `/training` but incomplete)
**Files to Create in `scripts/train/`:**
- `download_datasets.py` - Auto-download FaceForensics++, Celeb-DF, WaveFake, ASVspoof
- `build_manifest.py` - Create train/val/test splits (70/15/15)
- `train_image.py` - Train EfficientNet-B4 + Xception
- `train_audio.py` - Train RawNet2 on audio datasets
- `export_onnx.py` - Export trained models to ONNX
- `benchmark.py` - Evaluate AUC, EER, accuracy, F1

#### 8. **Demo Mode** ❌
**Status:** Auth exists but no demo auto-login
**Files to Update:**
- `frontend/src/pages/LoginPage.jsx` - Add "Try Demo" button
- `backend/api/auth.py` - Create demo user seed
- `backend/database.py` - Seed demo data on startup

---

## 📋 **IMPLEMENTATION ROADMAP**

### **PHASE 1: Core Detection Modules (Priority: CRITICAL)**
**Timeline:** Days 1-5

#### Task 1.1: Complete Audio Detection Module ✓
**Files:**
- Create `backend/routers/audio.py`
- Enhance `backend/ml/audio_model.py`
- Update `backend/services/audio_processor.py`
**Deliverables:**
- Audio upload endpoint accepting MP3/WAV/OGG
- Mel-spectrogram extraction with Librosa
- RawNet2 inference wrapper
- Spectrogram visualization with anomaly highlighting
- Integration with orchestrator

#### Task 1.2: Implement Social Media URL Fetching ✓
**Files:**
- Create `backend/routers/social.py`
- Enhance `backend/services/media_fetcher.py`
- Add Celery tasks in `backend/tasks.py`
**Deliverables:**
- URL validation for supported platforms
- yt-dlp integration for media extraction
- Celery background job queue
- WebSocket progress notifications
- Risk label assignment

#### Task 1.3: Verify Image Detection Ensemble ✓
**Files:**
- Review `backend/orchestrator/orchestrator.py`
- Test `backend/orchestrator/ensemble_aggregator.py`
**Deliverables:**
- Confirm weighted soft voting (ViT=0.30, EfficientNet=0.25, Xception=0.20)
- Verify GradCAM heatmap generation
- Test face detection + cropping
- Validate confidence thresholds

#### Task 1.4: Enhance Video Detection ✓
**Files:**
- Update `backend/ai/video_analyzer.py`
- Enhance `backend/detection/pipeline.py`
**Deliverables:**
- Frame sampling (every 5th frame)
- Per-frame ensemble inference
- Temporal consistency check (variance > 0.15)
- Timeline chart generation
- Suspicious timestamp detection

---

### **PHASE 2: Live Detection Features (Priority: HIGH)**
**Timeline:** Days 6-10

#### Task 2.1: Build Live Video Call Detection ✓
**Frontend:**
- Create `frontend/src/pages/LiveVideoPage.jsx`
- Create `frontend/src/hooks/useWebRTC.js`
- Create `frontend/src/components/VideoFeed.jsx`
**Backend:**
- Enhance `backend/routers/live.py`
- Add MobileNetV3 lightweight model
**Deliverables:**
- WebRTC video capture (getUserMedia)
- 2-second frame capture interval
- Real-time inference (≤50ms)
- Visual overlay (GREEN/RED border)
- Session logging

#### Task 2.2: Build Voice Call Detection ✓
**Frontend:**
- Create `frontend/src/pages/LiveAudioPage.jsx`
- Create `frontend/src/hooks/useMediaRecorder.js`
**Backend:**
- Create `backend/routers/audio_stream.py`
**Deliverables:**
- MediaRecorder audio capture
- 2-second chunk processing
- Per-chunk REAL/FAKE labels
- Alert banner for consecutive fakes
- Session transcript

#### Task 2.3: Implement Interview Proctoring ✓
**Frontend:**
- Create `frontend/src/pages/InterviewPage.jsx`
**Backend:**
- Create `backend/routers/interview.py`
- Enhance `backend/reporting/pdf_generator.py`
**Deliverables:**
- Dual-panel UI (video + audio)
- Combined integrity score
- 5-second update interval
- PDF report generation
- Timeline with screenshots

---

### **PHASE 3: Frontend Pages (Priority: HIGH)**
**Timeline:** Days 11-15

#### Task 3.1: Create Scanner Page ✓
**File:** `frontend/src/pages/ScannerPage.jsx`
**Features:**
- Unified upload tabs (Image/Video/Audio/URL)
- React Dropzone drag-and-drop
- Real-time progress bar (WebSocket)
- Result card with verdict badge
- Heatmap visualization
- Download report button

#### Task 3.2: Create Live Detection Page ✓
**File:** `frontend/src/pages/LiveDetectionPage.jsx`
**Features:**
- Tab switcher (Video Call | Voice Call | Interview)
- Embedded video/audio components
- Live confidence meter
- Record/Stop controls
- Session report download

#### Task 3.3: Create Social Media Page ✓
**File:** `frontend/src/pages/SocialMediaPage.jsx`
**Features:**
- URL input field with validation
- Scan queue table with status
- Celery job progress indicator
- Expandable result rows
- Platform icons

#### Task 3.4: Create History Page ✓
**File:** `frontend/src/pages/HistoryPage.jsx`
**Features:**
- Paginated table (10/25/50 per page)
- Filter by type/date/verdict
- Search functionality
- Re-run scan button
- Download JSON/PDF report
- Delete scan option

#### Task 3.5: Create Models Page ✓
**File:** `frontend/src/pages/ModelsPage.jsx`
**Features:**
- Model registry table
- Name, version, AUC, last trained
- Toggle active/inactive
- Upload new ONNX model
- Benchmark results

#### Task 3.6: Create Settings Page ✓
**File:** `frontend/src/pages/SettingsPage.jsx`
**Features:**
- User profile section
- API key management
- Notification preferences
- Theme toggle (dark/light)
- Export user data

---

### **PHASE 4: Training Pipeline (Priority: MEDIUM)**
**Timeline:** Days 16-20

#### Task 4.1: Dataset Download Script ✓
**File:** `scripts/train/download_datasets.py`
**Features:**
- FaceForensics++ (c23) download
- Celeb-DF v2 download
- WaveFake download
- ASVspoof 2021 download
- Progress bars with tqdm
- Checksum verification

#### Task 4.2: Manifest Builder ✓
**File:** `scripts/train/build_manifest.py`
**Features:**
- Create train/val/test CSV (70/15/15)
- Stratified split by fake method
- Label encoding
- Dataset statistics

#### Task 4.3: Image Training Script ✓
**File:** `scripts/train/train_image.py`
**Features:**
- EfficientNet-B4 + Xception training
- AdamW optimizer
- Cosine LR schedule
- AMP (fp16) training
- Gradient accumulation (4 steps)
- Early stopping (patience=5)
- MLflow logging

#### Task 4.4: Audio Training Script ✓
**File:** `scripts/train/train_audio.py`
**Features:**
- RawNet2 architecture
- WaveFake + ASVspoof datasets
- Mel-spectrogram augmentation
- Binary classification
- MLflow tracking

#### Task 4.5: ONNX Export Script ✓
**File:** `scripts/train/export_onnx.py`
**Features:**
- Export all trained models to ONNX
- Dynamic batch axis
- Verify parity with PyTorch
- Optimize for inference

#### Task 4.6: Benchmark Script ✓
**File:** `scripts/train/benchmark.py`
**Features:**
- Calculate AUC, EER, accuracy, F1
- Per-model metrics
- Ensemble metrics
- Confusion matrix
- ROC curve generation

---

### **PHASE 5: Demo Mode & Polish (Priority: MEDIUM)**
**Timeline:** Days 21-23

#### Task 5.1: Implement Demo Login ✓
**Files:**
- Update `frontend/src/pages/LoginPage.jsx`
- Update `backend/database.py`
- Create `backend/seeds/demo_data.py`
**Features:**
- "Try Demo" button on landing page
- Auto-fill credentials (demo@kavach.ai / demo1234)
- Pre-seeded detection results
- Pre-seeded alerts
- Read-only demo account

#### Task 5.2: Add Pre-trained Model Fallback ✓
**Files:**
- Update `backend/ai/hf_registry.py`
- Create `scripts/download_weights.py`
**Features:**
- Auto-download from Hugging Face Hub
- Models: hbenedek/efficientnet-deepfake, etc.
- Fallback if local training not available
- Version management

#### Task 5.3: Implement TTA (Test-Time Augmentation) ✓
**Files:**
- Update `backend/orchestrator/orchestrator.py`
- Add TTA transforms
**Features:**
- 5-crop averaging
- Horizontal flip
- Color jitter
- Gaussian blur
- Ensemble TTA results

---

### **PHASE 6: Testing & Documentation (Priority: HIGH)**
**Timeline:** Days 24-28

#### Task 6.1: Create Pytest Tests ✓
**Files to Create:**
- `tests/test_image_detection.py`
- `tests/test_video_detection.py`
- `tests/test_audio_detection.py`
- `tests/test_social_media.py`
- `tests/test_live_detection.py`
- `tests/test_interview.py`
- `tests/test_ensemble.py`
**Coverage Target:** ≥80%

#### Task 6.2: Update README.md ✓
**File:** `README.md`
**Sections:**
- 3-command startup instructions
- Feature overview with screenshots
- Architecture diagram (Mermaid)
- API documentation links
- Training guide
- Troubleshooting
- Contributing guidelines

#### Task 6.3: Verify Docker Compose ✓
**File:** `docker-compose.yml`
**Checks:**
- All services start successfully
- Health checks pass
- Volume mounts correct
- Environment variables set
- GPU profile works
- One-command startup verified

#### Task 6.4: Add WebSocket Reconnection ✓
**Files:**
- Update `frontend/src/hooks/useWebSocket.js`
- Update `backend/main.py` WebSocket handler
**Features:**
- Automatic reconnection on disconnect
- Exponential backoff
- Connection status indicator
- Heartbeat/ping-pong

#### Task 6.5: File Upload Validation ✓
**Files:**
- Update `backend/api/scan.py`
- Create `backend/utils/validators.py`
**Features:**
- MIME type validation
- File size limits (10MB image, 200MB video, 50MB audio)
- Extension whitelist
- Virus scan stub (ClamAV integration point)

#### Task 6.6: Standardize API Responses ✓
**Files:**
- Create `backend/schemas/responses.py`
- Update all router files
**Format:**
```json
{
  "success": true,
  "data": {...},
  "error": null,
  "timestamp": "2026-03-26T11:18:00Z"
}
```

#### Task 6.7: Mobile Responsiveness ✓
**Files:**
- Update all frontend pages
- Add responsive breakpoints
- Test on mobile devices
**Breakpoints:**
- Mobile: 320px - 640px
- Tablet: 641px - 1024px
- Desktop: 1025px+

---

### **PHASE 7: Final Integration & Optimization (Priority: CRITICAL)**
**Timeline:** Days 29-30

#### Task 7.1: Integration Testing ✓
**Scope:**
- End-to-end user flows
- All detection modules
- WebSocket connections
- File uploads
- Report generation
- Demo mode

#### Task 7.2: Performance Optimization ✓
**Focus Areas:**
- Model inference latency
- Database query optimization
- Redis caching strategy
- WebSocket message batching
- Frontend bundle size
- Image optimization

#### Task 7.3: Load Testing ✓
**Tools:** Locust, Apache Bench
**Targets:**
- 100 concurrent users
- 1000 requests/minute
- <500ms p95 latency
- <2% error rate

---

## 🎯 **SUCCESS CRITERIA**

### Functional Requirements ✓
- [ ] All 7 detection modules working end-to-end
- [ ] Real inference (no mocks) for all modules
- [ ] All 8 frontend pages complete and functional
- [ ] Demo mode with one-click login
- [ ] Training scripts ready to run
- [ ] Docker compose one-command startup
- [ ] ≥95% accuracy on FaceForensics++ test split

### Non-Functional Requirements ✓
- [ ] <500ms inference latency for images
- [ ] <50ms inference for live video (MobileNetV3)
- [ ] WebSocket reconnection handling
- [ ] File upload validation (type, size, virus-scan stub)
- [ ] All API responses follow standard format
- [ ] Frontend fully responsive (mobile + desktop)
- [ ] ≥80% test coverage
- [ ] Complete documentation

### Quality Gates ✓
- [ ] All pytest tests passing
- [ ] No linter errors
- [ ] No security vulnerabilities (Snyk scan)
- [ ] Load testing passed
- [ ] Manual QA checklist complete

---

## 📦 **DELIVERABLES**

### Code Artifacts
1. Complete backend with all 7 detection modules
2. Complete frontend with all 8 pages
3. Training scripts (6 files)
4. Pytest test suite (≥80% coverage)
5. Updated README.md with 3-command startup
6. Docker compose configuration
7. .env.example with all variables

### Documentation
1. API documentation (auto-generated from FastAPI)
2. Architecture diagram (Mermaid)
3. Training guide
4. Deployment guide
5. User manual
6. Developer guide

### Demo Assets
1. Demo account with pre-seeded data
2. Sample images/videos for testing
3. Video walkthrough (optional)

---

## 🚀 **NEXT STEPS**

1. **Review this plan** - Confirm priorities and timeline
2. **Set up development environment** - Ensure all dependencies installed
3. **Start Phase 1** - Begin with audio detection module
4. **Daily standups** - Track progress and blockers
5. **Continuous testing** - Test each module as completed
6. **Final review** - Comprehensive QA before delivery

---

## 📞 **SUPPORT & ESCALATION**

- **Technical Blockers:** Escalate immediately if any hard technical blocker exists
- **Scope Changes:** Discuss any requirement changes before implementation
- **Timeline Adjustments:** Communicate early if timeline needs adjustment

---

**Document Version:** 1.0  
**Last Updated:** 2026-03-26  
**Status:** Ready for Implementation  
**Estimated Completion:** 30 days from start