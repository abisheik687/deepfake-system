# Internal trace:
# - Wrong before: this diagram documented the retired browser extension, old API layout, and legacy orchestration stack that are no longer part of the active product path.
# - Fixed now: the diagram reflects the current web-only KAVACH-AI structure and the archived legacy boundary.

# KAVACH-AI / DeepShield Web Architecture

This diagram now reflects the active, web-only application structure.

---

## 1. High-Level System Architecture

```mermaid
flowchart TB
    subgraph Clients["Client Tier"]
        Web["React Web App\n(Vite + Framer Motion)"]
        Mobile["Mobile Browser\nResponsive UI"]
    end

    subgraph Backend["Backend API (FastAPI)"]
        API["main.py\n/health, /analyse"]
        Routers["routers/\nhealth.py, analyse.py"]
        Schemas["schemas/\nrequest.py, response.py"]
    end

    subgraph Core["Detection Core"]
        Loader["models/loader.py\nstartup model registry"]
        Ensemble["models/ensemble.py\nweighted voting"]
        ImagePipeline["pipelines/image_pipeline.py"]
        VideoPipeline["pipelines/video_pipeline.py"]
        AudioPipeline["pipelines/audio_pipeline.py"]
    end

    subgraph Support["Utilities"]
        Config["config.py"]
        Files["utils/file_utils.py"]
        Logger["utils/logger.py"]
    end

    Web --> API
    Mobile --> API
    API --> Routers
    Routers --> Schemas
    Routers --> ImagePipeline
    Routers --> VideoPipeline
    Routers --> AudioPipeline
    ImagePipeline --> Loader
    VideoPipeline --> Loader
    AudioPipeline --> Loader
    Loader --> Ensemble
    API --> Config
    Routers --> Files
    API --> Logger
```

---

## 2. Active Repo Structure

```text
.
+-- backend/
¦   +-- main.py
¦   +-- config.py
¦   +-- routers/
¦   +-- models/
¦   +-- pipelines/
¦   +-- schemas/
¦   +-- utils/
+-- frontend/
¦   +-- src/
¦   ¦   +-- api/
¦   ¦   +-- components/
¦   ¦   +-- hooks/
¦   ¦   +-- pages/
¦   ¦   +-- styles/
+-- legacy/
¦   +-- backend/
¦   +-- frontend/
¦   +-- root-docs/
¦   +-- scripts/
+-- docker-compose.yml
```

---

## 3. Notes

- The Chrome extension was removed from the active product path.
- Realtime/live analysis remains archived under `legacy/` and is not part of the running web application.
- The current product is intentionally centered on upload-and-analyse reliability across desktop, tablet, and mobile browsers.
