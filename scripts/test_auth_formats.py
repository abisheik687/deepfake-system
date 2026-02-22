import requests

# Test with the EXACT same format as frontend sends
url = "http://localhost:8000/auth/token"

# Frontend sends FormData, which axios converts to application/x-www-form-urlencoded
data = {
    'username': 'admin@kavach.ai',
    'password': 'Kavach@2026'
}

print("=" * 60)
print("Testing Frontend Auth Format")
print("=" * 60)

# Test 1: Using x-www-form-urlencoded (what frontend should send)
print("\n[Test 1] application/x-www-form-urlencoded format:")
try:
    response = requests.post(
        url,
        data=data,
        headers={'Content-Type': 'application/x-www-form-urlencoded'}
    )
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.text[:200]}")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test 2: Using FormData approach
print("\n[Test 2] FormData simulation:")
try:
    import urllib.parse
    encoded_data = urllib.parse.urlencode(data)
    response = requests.post(
        url,
        data=encoded_data,
        headers={'Content-Type': 'application/x-www-form-urlencoded'}
    )
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.text[:200]}")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Test 3: What happens with multipart/form-data
print("\n[Test 3] multipart/form-data format (WRONG for OAuth2):")
try:
    response = requests.post(url, files=data)
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.text[:200]}")
except Exception as e:
    print(f"  ❌ Error: {e}")

print("\n" + "=" * 60)
print("Summary: OAuth2 requires application/x-www-form-urlencoded")
print("=" * 60)
