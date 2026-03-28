# 🔧 Demo Login Troubleshooting Guide

## Demo Credentials

**Email**: `demo@kavach.ai`  
**Password**: `kavach2026`

---

## ✅ Verification Checklist

### 1. Backend Server Running
```bash
# Check if backend is running on port 8000
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "timestamp": "..."}
```

### 2. Database Initialized
```bash
# Run database initialization
python scripts/init_db.py

# Or check if demo user exists
python -c "from backend.database import seed_demo_user; import asyncio; asyncio.run(seed_demo_user())"
```

### 3. Frontend Connected
```bash
# Check frontend environment variable
cat frontend/.env

# Should contain:
# VITE_API_URL=http://localhost:8000
```

---

## 🐛 Common Issues & Fixes

### Issue 1: "Demo login failed"

**Cause**: Backend server not running or not accessible

**Fix**:
```bash
# Start backend
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or use Docker
docker compose up backend
```

### Issue 2: "Invalid credentials"

**Cause**: Demo user not seeded in database

**Fix**:
```bash
# Seed demo user manually
python scripts/init_db.py

# Or recreate database
rm kavach.db
python scripts/init_db.py
```

### Issue 3: CORS errors in browser console

**Cause**: Frontend and backend on different origins

**Fix**: Check [`backend/main.py`](backend/main.py) CORS configuration:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Issue 4: Network timeout

**Cause**: Slow network or backend overloaded

**Fix**: Increase timeout in [`frontend/src/services/api.js`](frontend/src/services/api.js):
```javascript
const api = axios.create({
    baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
    timeout: 30000, // Increase from 10000 to 30000
});
```

---

## 🧪 Manual Testing

### Test 1: Direct API Call
```bash
# Test login endpoint directly
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"demo@kavach.ai","password":"kavach2026"}'

# Expected response:
# {"access_token":"...", "token_type":"bearer", "user":{...}}
```

### Test 2: Browser Console
```javascript
// Open browser console on login page and run:
fetch('http://localhost:8000/auth/login', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        username: 'demo@kavach.ai',
        password: 'kavach2026'
    })
})
.then(r => r.json())
.then(console.log)
.catch(console.error);
```

### Test 3: Check Demo User in Database
```bash
# SQLite
sqlite3 kavach.db "SELECT email, full_name, role FROM users WHERE email='demo@kavach.ai';"

# Expected output:
# demo@kavach.ai|Demo Officer|demo
```

---

## 🔄 Reset Demo User

If demo login is completely broken, reset it:

```bash
# Delete and recreate demo user
python -c "
from backend.database import get_db, User
from backend.config import pwd_context
import asyncio

async def reset_demo():
    async for db in get_db():
        # Delete existing
        result = await db.execute(select(User).filter(User.email=='demo@kavach.ai'))
        user = result.scalars().first()
        if user:
            await db.delete(user)
            await db.commit()
        
        # Create new
        demo = User(
            email='demo@kavach.ai',
            hashed_password=pwd_context.hash('kavach2026'),
            full_name='Demo Officer',
            role='demo'
        )
        db.add(demo)
        await db.commit()
        print('✓ Demo user reset successfully')

asyncio.run(reset_demo())
"
```

---

## 📝 Logging

Enable debug logging to see what's happening:

**Backend** ([`backend/main.py`](backend/main.py)):
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Frontend** ([`frontend/src/context/AuthContext.jsx`](frontend/src/context/AuthContext.jsx)):
```javascript
const login = async (username, password) => {
    console.log('Attempting login:', username); // Add this
    const response = await api.post('/auth/login', { username, password });
    console.log('Login response:', response.data); // Add this
    // ... rest of code
};
```

---

## ✅ Success Indicators

When demo login works correctly, you should see:

1. **Backend logs**:
   ```
   INFO: POST /auth/login 200 OK
   ```

2. **Browser console**:
   ```
   Login successful
   Redirecting to /dashboard
   ```

3. **Browser storage**:
   ```javascript
   localStorage.getItem('token') // Should return JWT token
   ```

4. **Dashboard loads** with demo user data

---

## 🆘 Still Not Working?

If demo login still fails after trying all fixes:

1. **Check browser console** for error messages
2. **Check backend logs** for authentication errors
3. **Verify network tab** in browser DevTools
4. **Try manual login** with the same credentials in the form
5. **Restart both frontend and backend** servers
6. **Clear browser cache** and localStorage
7. **Try a different browser** to rule out browser-specific issues

---

## 📞 Support

If the issue persists, provide these details:
- Browser console errors
- Backend server logs
- Network tab screenshot
- Steps to reproduce

**Demo credentials are hardcoded and should always work if the backend is running and database is initialized.**