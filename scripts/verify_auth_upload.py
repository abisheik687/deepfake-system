
import sys
import os
from fastapi.testclient import TestClient

# Add project root to path
sys.path.append(os.getcwd())

from backend.api.main import app

client = TestClient(app)

def test_api_security():
    print("Testing API Security & Uploads...")
    
    # 1. Register Admin
    print("1. Registering User...")
    client.post("/auth/register?email=admin@kavach.ai&password=securepassword")

    # 2. Login & Get Token
    print("2. Logging in...")
    response = client.post(
        "/auth/token",
        data={"username": "admin@kavach.ai", "password": "securepassword"}
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("✅ Token Received.")

    # 3. Protected Upload
    print("3. Testing Protected Upload...")
    response = client.post(
        "/scan/upload",
        files={"file": ("secure_vid.mp4", b"data", "video/mp4")},
        headers=headers
    )
    assert response.status_code == 200
    print("✅ Upload Accepted.")

    # 4. Unprotected Access (Fail)
    print("4. Testing Unauthorized Access...")
    response = client.post(
        "/scan/upload",
        files={"file": ("hacker_vid.mp4", b"data", "video/mp4")}
    )
    assert response.status_code == 401
    print("✅ Unauthorized Request Blocked.")

if __name__ == "__main__":
    try:
        test_api_security()
        print("\n🎉 API EXPANSION VERIFIED: Auth works, endpoints secured.")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
