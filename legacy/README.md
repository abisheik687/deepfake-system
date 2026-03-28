# Internal trace:
# - Wrong before: inactive experiments and the old product surface were mixed directly into the active source tree.
# - Fixed now: the retired code lives under `legacy/` so the active app stays readable while historical work remains recoverable.

# Legacy Archive

This directory contains code that is no longer part of the active KAVACH-AI runtime.

Contents:

- `legacy/backend/`: archived backend modules from the older dashboard, agentic, realtime, and orchestration stack
- `legacy/frontend/`: archived frontend pages, hooks, components, and support files from the older UI
- `legacy/root-docs/`: historical reports and planning documents moved out of the repo root
- `legacy/scripts/`: older helper scripts that are not part of the current product path

The active application now lives only in:

- `backend/`
- `frontend/`
