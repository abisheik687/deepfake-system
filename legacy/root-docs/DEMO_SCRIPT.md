
# KAVACH-AI: Final Demo Script

**Objective**: Demonstrate the full end-to-end capability of the system in 5 minutes or less.

## 1. Login & Dashboard (30s)
*   **Action**: Open `http://localhost:3000`. Redirects to `/login`.
*   **Action**: Enter credentials (`admin` / `admin`).
*   **Talk Track**: "Access is restricted to authorized forensic personnel. We use JWT for secure, stateless authentication."
*   **Visual**: Dashboard loads with animations. Show the "System Health" and "Recent Scans" stats.

## 2. Upload & Analysis (1.5m)
*   **Action**: Click "New Scan" (Sidebar) -> Upload Page.
*   **Action**: Drag & Drop a deepfake video file (e.g., `test_deepfake.mp4`).
*   **Talk Track**: "Our chunked upload handler allows large evidence files to be ingested reliably. The system automatically hashes the file for chain-of-custody."
*   **Action**: Click "View Analysis" after upload.
*   **Visual**: Analysis Page charts animate in.
*   **Talk Track**: "The Fusion Engine combines Video, Audio, and Temporal analysis. Here we see a high confidence score of 94% FAKE, primarily driven by lip-sync anomalies detected in the Audio channel."

## 3. Real-Time Monitor (1m)
*   **Action**: Navigate to "Live Monitor".
*   **Visual**: Show the simulated video feed and the scrolling "Event Stream" on the right.
*   **Talk Track**: "For active surveillance, we support RTSP stream ingestion. The system flags threats in real-time, simulating a 'Red Alert' for high-confidence detections."

## 4. Reports & History (1m)
*   **Action**: Navigate to "Reports".
*   **Visual**: Show the table of historical scans. Use the "Filter" to search for a specific case ID.
*   **Action**: Click "Download PDF" (Mock action).
*   **Talk Track**: "Every action is logged. Analysts can retrieve past cases and export official forensic reports for court procedures."

## 5. Closing (1m)
*   **Talk Track**: "KAVACH-AI is not just a model; it's a complete, deployable platform. It runs fully containerized on Docker, ready for on-premise deployment."
*   **Q&A Prep**: Be ready to explain the "Fusion Engine" weights (Video 50%, Audio 30%, Temporal 20%).
