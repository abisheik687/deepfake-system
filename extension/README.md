# KAVACH.AI Deepfake Shield — Chrome Extension

Detects deepfakes and fake news **while you scroll** Instagram, Facebook, WhatsApp Web, X/Twitter, LinkedIn, and Reddit. Powered by the KAVACH.AI backend with optional AI explanations via Gemini or OpenAI.

---

## ⚡ Quick Setup (5 minutes, free)

### Step 1 — Generate Icons
```powershell
cd "e:\Users\Abisheik\downloads\deepfake system\deepfake system"
py extension/generate_icons.py
```

### Step 2 — Start the Backend
```powershell
py -m uvicorn backend.main:app --reload --port 8000
```

### Step 3 — Load Extension in Chrome
1. Open Chrome → address bar → type `chrome://extensions/`
2. Enable **Developer mode** toggle (top-right)
3. Click **"Load unpacked"**
4. Select the folder:
   ```
   e:\Users\Abisheik\downloads\deepfake system\deepfake system\extension
   ```
5. The KAVACH.AI shield icon appears in your Chrome toolbar ✅

---

## 🔑 Configure AI Explanation (Optional)

For "Why is this fake?" explanations using live web search:

1. Click the **KAVACH.AI shield icon** in the Chrome toolbar
2. Select your AI provider:
   - **Gemini** (recommended) — free tier, live Google Search grounding
   - **OpenAI** — GPT-4o-mini
3. Paste your API key
4. Click **Save Settings**

### Getting a free Gemini API key
1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"** → copy it → paste into KAVACH.AI popup

---

## 🧠 How It Works

```
Social Media Page (Chrome)
    │  MutationObserver watches for new images as you scroll
    ▼
content.js grabs each <img> src URL
    │  chrome.runtime.sendMessage({ type: "ANALYZE_IMAGE", url })
    ▼
background.js → POST http://localhost:8000/api/analyze-url
    │  Backend downloads image, runs OpenCV face detection
    ▼
Verdict: REAL ✅ / FAKE ⚠️ / NO_FACE 👁️
    │  Badge overlaid on the image
    ▼
User clicks FAKE badge → AI explanation panel opens
    │  background.js → POST /api/explain → Gemini (with Google Search)
    ▼
"This appears to be a deepfake because..." (live web-sourced)
```

---

## 🌐 Supported Sites

| Site | Status | Notes |
|------|--------|-------|
| Instagram.com | ✅ | Feed, Reels, Stories thumbnails |
| Facebook.com | ✅ | Feed posts, profile pictures |
| WhatsApp Web | ✅ | Contact and group photos |
| X / Twitter | ✅ | Profile pictures, post images |
| LinkedIn | ✅ | Profile pictures, posts |
| Reddit | ✅ | Post images |

> **Note:** This works only on the **web** versions of these platforms. Native mobile apps are sandboxed and cannot be accessed by browser extensions.

---

## 📁 File Structure

```
extension/
├── manifest.json       Chrome Extension Manifest V3
├── background.js       Service worker — API calls
├── content.js          Page scanner — image detection & badges
├── content.css         Badge & panel styles
├── popup.html          Settings popup UI
├── popup.js            Settings logic
├── generate_icons.py   Icon generator script
└── icons/
    ├── icon16.png
    ├── icon48.png
    └── icon128.png
```

---

## 🔌 Backend API Endpoints (for the extension)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /api/analyze-url` | `{ url }` | Download & analyze image URL |
| `POST /api/explain` | `{ image_url, verdict, provider, api_key, ... }` | AI fake news explanation |

---

## ❓ FAQ

**Q: Does it send my images to any server?**
A: Images are sent to your **local** KAVACH.AI backend only (localhost:8000). No data leaves your machine unless you use the AI explanation feature (which calls Gemini/OpenAI).

**Q: Does it slow down my browser?**
A: Each image is analyzed at most once with a queue throttle of 800ms between requests. Minimal impact.

**Q: What if the backend is offline?**
A: The extension shows a yellow "🔌 N/A" badge and the popup shows "Backend Offline". Start the backend and reload the tab.
