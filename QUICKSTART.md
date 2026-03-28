# 🚀 KAVACH-AI Quick Start Guide

## ✅ Server Status: RUNNING

The KAVACH-AI backend is now running on **http://localhost:8000**

---

## 📍 Access Points

### 1. **API Documentation (Swagger UI)**
```
http://localhost:8000/docs
```
Interactive API documentation where you can:
- Test all endpoints
- View request/response formats
- Execute API calls directly

### 2. **ReDoc Documentation**
```
http://localhost:8000/redoc
```
Alternative documentation format with better readability

### 3. **Health Check**
```bash
curl http://localhost:8000/health
```

### 4. **WebSocket Endpoint**
```
ws://localhost:8000/ws
```
Real-time updates for:
- Detection results
- Alert notifications
- System status

### 5. **Prometheus Metrics**
```
http://localhost:8000/metrics
```
System performance and monitoring metrics

---

## 🧪 Quick API Tests

### Test 1: Root Endpoint
```bash
curl http://localhost:8000/
```

### Test 2: Health Check
```bash
curl http://localhost:8000/health
```

### Test 3: Available API Paths
```bash
curl -s http://localhost:8000/openapi.json | python -m json.tool | grep -A 2 '"paths"'
```

---

## 🎯 Core API Endpoints

### 1. **Audio Deepfake Detection** (`POST /api/audio`)
Detect deepfakes in audio files/calls

**Example:**
```bash
curl -X POST "http://localhost:8000/api/audio/analyze" \
  -H "Content-Type: application/json" \
  -d '{"audio_url": "https://example.com/audio.wav"}'
```

### 2. **Video Deepfake Detection** (`POST /api/live-video`)
Analyze video streams for deepfakes

**Example:**
```bash
curl -X POST "http://localhost:8000/api/live-video/analyze" \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://example.com/video.mp4"}'
```

### 3. **Social Media Analysis** (`POST /api/social`)
Detect deepfakes on social media posts

**Example:**
```bash
curl -X POST "http://localhost:8000/api/social/scan" \
  -H "Content-Type: application/json" \
  -d '{"post_url": "https://twitter.com/user/post/123"}'
```

### 4. **Unified Scanner** (`POST /api/scan`)
Comprehensive multi-modal analysis

**Example:**
```bash
curl -X POST "http://localhost:8000/api/scan/comprehensive" \
  -H "Content-Type: application/json" \
  -d '{"media_url": "https://example.com/media"}'
```

### 5. **Interview Proctoring** (`POST /api/interview`)
Real-time interview deepfake detection

**Example:**
```bash
curl -X POST "http://localhost:8000/api/interview/start" \
  -H "Content-Type: application/json" \
  -d '{"interview_id": "test-001"}'
```

### 6. **Alerts Management** (`GET/POST /api/alerts`)
View and manage detection alerts

**Example:**
```bash
curl http://localhost:8000/api/alerts/
```

### 7. **Detection History** (`GET /api/detections`)
View past detection results

**Example:**
```bash
curl "http://localhost:8000/api/detections/?limit=10"
```

---

## 🔌 WebSocket Usage

### Connect via JavaScript:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
  console.log('Connected to KAVACH-AI');
  
  // Subscribe to alerts
  ws.send(JSON.stringify({
    type: 'subscribe',
    channels: ['alerts', 'detections']
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
  
  if (data.type === 'detection') {
    console.log(`Deepfake detected! Confidence: ${data.confidence}`);
  }
};

// Send frame for real-time analysis
ws.send(JSON.stringify({
  type: 'frame',
  data: 'base64_encoded_frame_data'
}));
```

### Ping/Pong for Keep-Alive:
```javascript
// Send ping every 30 seconds
setInterval(() => {
  ws.send(JSON.stringify({ type: 'ping' }));
}, 30000);
```

---

## 🧪 Testing with Python

### Install dependencies:
```bash
pip install requests websockets
```

### Example: Detect audio deepfake
```python
import requests
import json

# Analyze audio URL
response = requests.post(
    'http://localhost:8000/api/audio/analyze',
    json={
        'audio_url': 'https://example.com/sample.wav'
    }
)

print(response.json())
```

### Example: WebSocket real-time detection
```python
import asyncio
import websockets
import json

async def detect():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as ws:
        # Subscribe to alerts
        await ws.send(json.dumps({
            'type': 'subscribe',
            'channels': ['alerts']
        }))
        
        # Receive messages
        async for message in ws:
            data = json.loads(message)
            print(f"Alert: {data}")
            
            if data.get('type') == 'detection':
                print(f"Deepfake detected! Confidence: {data['confidence']}")

asyncio.run(detect())
```

---

## 📊 System Monitoring

### View Prometheus Metrics:
```bash
curl http://localhost:8000/metrics
```

### Key Metrics:
- `kavach_detections_total` - Total detections
- `kavach_false_positives` - False positive count
- `kavach_processing_time_seconds` - Detection latency
- `kavach_active_websockets` - Connected clients
- `kavach_streams_processed` - Streams analyzed

---

## 🛑 Stopping the Server

### Find the process:
```bash
# Windows
netstat -ano | findstr :8000

# Linux/macOS
lsof -i :8000
```

### Kill the process:
```bash
# Windows (replace PID with actual process ID)
taskkill /PID <PID> /F

# Linux/macOS
kill -9 <PID>
```

---

## 🐛 Troubleshooting

### Issue: "Address already in use"
```bash
# Find and kill existing process
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Issue: Module import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Database locked
```bash
# Delete lock file if exists
del data\kavach.db.lock 2>nul
```

### Issue: Out of memory
```bash
# Restart with less workers
uvicorn backend.main:app --workers 1 --host 0.0.0.0 --port 8000
```

---

## 🎯 Next Steps

1. **Access API Docs:** http://localhost:8000/docs
2. **Try Sample Detection:** Use the `/api/scan/comprehensive` endpoint
3. **Set Up Frontend:** Navigate to frontend directory and run `npm start`
4. **Configure Alerts:** Set up Slack/email notifications
5. **Load Models:** Download pre-trained models for better accuracy

---

## 📞 API Examples

### Audio Deepfake Detection
```bash
curl -X POST "http://localhost:8000/api/audio/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://example.com/voice_sample.wav",
    "analyze_spectrogram": true,
    "check_voiceprint": true
  }'
```

### Video Deepfake Detection
```bash
curl -X POST "http://localhost:8000/api/live-video/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://example.com/video.mp4",
    "check_facial_features": true,
    "check_blink_patterns": true
  }'
```

### Unified Multi-Modal Scan
```bash
curl -X POST "http://localhost:8000/api/scan/comprehensive" \
  -H "Content-Type: application/json" \
  -d '{
    "media_url": "https://example.com/media.mp4",
    "modules": ["video", "audio", "metadata"],
    "confidence_threshold": 0.7
  }'
```

---

## ✅ System Status

- **Backend:** ✅ Running on port 8000
- **API Docs:** ✅ http://localhost:8000/docs
- **WebSocket:** ✅ ws://localhost:8000/ws
- **Database:** ✅ SQLite (async)
- **Detection Pipeline:** ✅ Operational

**KAVACH-AI is ready to use! 🎉**
