import requests

# Test the auth endpoint directly
url = "http://localhost:8000/auth/token"

# Test with correct credentials
data = {
    'username': 'admin@kavach.ai',
    'password': 'Kavach@2026'
}

print("Testing auth endpoint with credentials:")
print(f"URL: {url}")
print(f"Username: {data['username']}")
print(f"Password: {data['password']}")
print()

try:
    response = requests.post(url, data=data, headers={
        'Content-Type': 'application/x-www-form-urlencoded'
    })
    
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"Response Body: {response.text}")
    print()
    
    if response.status_code == 200:
        print("✅ Authentication successful!")
        print(f"Access Token: {response.json()['access_token']}")
    else:
        print("❌ Authentication failed!")
        print(f"Error: {response.json()}")
        
except Exception as e:
    print(f"❌ Error making request: {e}")
