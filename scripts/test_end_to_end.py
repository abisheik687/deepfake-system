
import sys
import os
import time
import requests
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.append(os.getcwd())

from backend.api.main import app
from backend.database import Base, get_db
from backend.database import SessionLocal, ScanResult, User

# Use TestClient for internal API calls (simulates HTTP)
client = TestClient(app)

def test_full_flow():
    print("🚀 Starting KAVACH-AI End-to-End Test...")

    # 1. Setup: Register a unique user for this test run
    email = f"officer_{int(time.time())}@kavach.ai"
    password = "SafePassword123!"
    
    print(f"\n1. Registering User: {email}...")
    try:
        resp = client.post(f"/auth/register?email={email}&password={password}")
        if resp.status_code == 200:
            print("✅ Registration Successful.")
        elif resp.status_code == 400:
            print("⚠️ User already exists (Skipping registration).")
        else:
            print(f"❌ Registration Failed: {resp.text}")
            return
    except Exception as e:
        print(f"❌ Registration Error: {e}")
        return

    # 2. Login: Get JWT Token
    print("\n2. Logging in...")
    resp = client.post(
        "/auth/token",
        data={"username": email, "password": password}
    )
    if resp.status_code != 200:
        print(f"❌ Login Failed: {resp.text}")
        return
    
    token = resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("✅ Login Successful. Token received.")

    # 3. Upload Video
    print("\n3. Uploading Suspect Video...")
    # Create a dummy video file in memory
    filename = "e2e_evidence.mp4"
    files = {"file": (filename, b"fake_video_content_bytes", "video/mp4")}
    
    resp = client.post("/scan/upload", files=files, headers=headers)
    
    if resp.status_code == 429:
        print("⚠️ Rate Limit Hit! Waiting 60s...")
        time.sleep(60)
        resp = client.post("/scan/upload", files=files, headers=headers)
        
    if resp.status_code != 200:
        print(f"❌ Upload Failed: {resp.text}")
        return
        
    data = resp.json()
    task_id = data["task_id"]
    print(f"✅ Upload Accepted. Task ID: {task_id}")
    print(f"   Status: {data['status']}")

    # 4. Check Result (Simulate Async Wait)
    print("\n4. Checking Results...")
    # Since we are using TestClient, the BackgroundTasks might not run fully async 
    # depending on how they were attached. 
    # For this E2E, we might see "queued" if the worker isn't actually running separately.
    # However, our 'worker.py' logic is triggered mostly via direct import in dev mode 
    # inside 'upload_for_analysis' if configured that way, OR we need to verify DB state.
    
    resp = client.get(f"/scan/result/{task_id}", headers=headers)
    if resp.status_code == 200:
        result = resp.json()
        print(f"✅ Result Retrieved:")
        print(f"   Status: {result.get('status')}")
        if 'report' in result:
             print(f"   Verdict: {result['report'].get('verdict', 'Pending')}")
    else:
        print(f"❌ Failed to get result: {resp.text}")

    print("\n🎉 End-to-End Test Complete!")

if __name__ == "__main__":
    test_full_flow()
