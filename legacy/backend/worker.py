
import os
from celery import Celery
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from backend.database import engine, SessionLocal
from backend.crud import update_scan_result

load_dotenv()

# KAVACH-AI Day 8: Async Worker (Updated Day 11)
# Handles heavy AI inference & DB Persistence

BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery_app = Celery(
    "kavach_worker",
    broker=BROKER_URL,
    backend=RESULT_BACKEND
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_concurrency=1 
)

import asyncio

async def async_db_update(task_id: str, report: dict):
    from backend.database import SessionLocal
    from backend.crud import update_scan_result
    
    async with SessionLocal() as db:
        try:
            print(f"[{task_id}] Saving result to DB...")
            await update_scan_result(db, task_id, report)
        except Exception as e:
            print(f"Error saving to DB: {e}")

@celery_app.task(name="tasks.process_video")
def process_video_task(file_path: str, task_id: str):
    """
    Async wrapper for detection pipeline + DB Update.
    """
    from backend.detection.pipeline import monitor_pipeline
    
    print(f"[{task_id}] Processing video: {file_path}")
    
    # 1. Run Pipeline
    report = monitor_pipeline.process_media(file_path)
    
    # 2. Save to DB
    asyncio.run(async_db_update(task_id, report.__dict__))
    
    print(f"[{task_id}] Complete. Verdict: {report.verdict}")
    return report.__dict__
