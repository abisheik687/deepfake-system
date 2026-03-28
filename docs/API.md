# 📡 API Reference — Cloud-Scale Forensic Access

This document details the premium API interfaces for **KAVACH-AI v2.0**.

## 1. Authentication & Security
Requests must include a valid JWT token in the `Authorization` header.

```http
GET /api/v1/mission/status HTTP/1.1
Authorization: Bearer <TOKEN>
```

---

## 2. Principal Endpoints

### `POST /api/v1/scan/hyper`
The primary entry point for multi-modal forensic analysis.

**Path Parameters**: `None`
**Body (Multipart)**: `media_file`, `priority_flag`, `metadata_v3`

**Status Codes**:
- `202 Accepted`: Analysis initialized in the Celery cluster.
- `400 Bad Request`: Invalid media encoding or corrupted stream.

### `GET /api/v1/agency/briefing/{alert_id}`
Retrieves the autonomous reasoning log and summary from the Mission Control Agency.

### `GET /api/v1/reports/bundle/{report_id}`
Downloads the cryptographically signed Forensic PDF Bundle.

---

## 3. Real-Time Telemetry (WebSocket)
KAVACH-AI provides low-latency telemetry for live monitoring systems.

| Event Type | Direction | Payload Description |
| :--- | :--- | :--- |
| `INQUEST_START` | Server -> Client | Notifies UI that Agency has started a new investigation. |
| `NEURAL_FRAME` | Client -> Server | Compressed frame data for real-time mobile scanning. |
| `MISSION_COMPLETED` | Server -> Client | Final verdict and PDF link delivery. |

---

## 4. Performance Benchmarks
- **Avg. Latency**: <1800ms (Video Segment Analysis).
- **Peak Throughput**: 100+ concurrent ingestion streams.
- **Uptime Node**: 99.98% (PostgreSQL High-Availability cluster).

---
*End of Specification. Version 2.0-STABLE.*
