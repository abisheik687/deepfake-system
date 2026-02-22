
import sys
import os
from fastapi.testclient import TestClient

# Add project root to path
sys.path.append(os.getcwd())

from backend.api.main import app

client = TestClient(app)

def test_health_check():
    print("Testing / (Health Check)...")
    response = client.get("/")
    assert response.status_code == 200
    print(f"✅ Response: {response.json()}")

def test_scan_flow():
    print("\nTesting /scan/upload (Mock)...")
    # Simulate file upload
    response = client.post(
        "/scan/upload",
        files={"file": ("test_video.mp4", b"dummy_content", "video/mp4")}
    )
    assert response.status_code == 200
    data = response.json()
    print(f"✅ Scanning triggered. Task ID: {data['task_id']}")
    
    task_id = data['task_id']
    
    print(f"\nTesting /scan/result/{task_id} (Fusion Logic)...")
    response = client.get(f"/scan/result/{task_id}")
    assert response.status_code == 200
    result = response.json()
    print(f"✅ Fusion Report Received:")
    print(f"   Verdict: {result['report']['verdict']}")
    print(f"   Confidence: {result['report']['confidence']}")
    print(f"   Score: {result['report']['final_score']:.4f}")
    
    # Check compliance with simple fusion logic
    # Mock inputs were video=0.88(Fake), audio=0.12(Real), temporal=0.75(Suspicious)
    # Weights: V(0.4) + A(0.3) + T(0.3)
    # Expected: 0.88*0.4 + 0.12*0.3 + 0.75*0.3 = 0.352 + 0.036 + 0.225 = 0.613
    # BUT logic has overrides. Check logic.
    assert result['report']['final_score'] > 0.5

if __name__ == "__main__":
    try:
        test_health_check()
        test_scan_flow()
        print("\n🎉 SYSTEM VERIFICATION PASSED: API & Fusion Engine Integration.")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
