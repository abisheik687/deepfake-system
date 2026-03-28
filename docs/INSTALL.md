# 🏗️ Production Deployment & Cluster Orchestration

This guide provides the definitive protocol for deploying **KAVACH-AI v2.0** in both development and high-availability production environments.

## 1. Minimal Hardware Requirements
To run the Hyper-Modal Ensemble at peak performance, we recommend:
- **CPU**: 8+ Cores (AMD Threadripper or Intel Scalable preferred).
- **RAM**: 32GB+ (For local model caching).
- **GPU**: NVIDIA RTX 3090/4090 or A100 (24GB+ VRAM) for parallel multi-model inference.
- **Storage**: 500GB NVMe SSD (High I/O for Kafka/MinIO).

## 2. Theoretical Deployment Architecture
KAVACH-AI utilizes a **Decentralized Forensic Hub** model where nodes can be geographically distributed but logically centralized via the **Mission Control Agency**.

---

## 3. Deployment via Docker Cluster (The "Iron Dome")

### Quick Initialization
```bash
# 1. Clone & Enter Hub
git clone https://github.com/abisheik687/kavach-ai.git && cd kavach-ai

# 2. Master Bootstrap
# This script handles Venv, Dependencies, and Directory Signatures
bash setup.sh

# 3. Launch Cluster
docker compose up -d --build
```

### Cluster Service Topology
| Service | Role | Port | Persistence |
| :--- | :--- | :--- | :--- |
| **Kavach Gateway** | FastAPI Entry Point | 8000 | N/A |
| **Forensic Worker** | Celery/Torch Inference | N/A | `evidence/` |
| **Mission Control** | Dashboard Frontend | 3000 | N/A |
| **Agency Broker** | Redis / Kafka | 6379/9092 | `redis_data` |
| **Intelligence DB** | PostgreSQL | 5432 | `pg_data` |

---

## 4. Scaling for Enterprise (Kubernetes)
For national-scale security, deploy the provided K8s manifests:

```bash
kubectl apply -f infra/k8s/namespace.yaml
kubectl apply -f infra/k8s/configmap.yaml
kubectl apply -f infra/k8s/deployment.yaml
```

### Horizontal Pod Autoscaling (HPA)
The system automatically scales forensic pods based on the **Prometheus** metrics exported by the backend, ensuring zero-latency detection during mass disinformation events.

---

## 5. Troubleshooting
- **Metric Lag**: Check `Grafana` dashboard at `http://localhost:3001` to identify bottleneck services.
- **Inference Latency**: Ensure `nvidia-container-toolkit` is correctly configured if using the `--profile gpu` flag.
- **Storage Overflow**: Adjust data retention policies in `backend/config.py`.

---
*Operational Document — KAVACH-AI Mission Control*
