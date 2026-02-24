/**
 * KAVACH.AI Extension Popup — Settings & Stats
 */

const BACKEND = 'http://localhost:8000';

// ─── Load saved settings ────────────────────────────────────────────────────
chrome.storage.local.get(['apiKey', 'apiProvider', 'enabled', 'stats'], ({ apiKey, apiProvider, enabled, stats }) => {
    if (apiKey) document.getElementById('api-key').value = apiKey;
    if (apiProvider) document.getElementById('api-provider').value = apiProvider;
    document.getElementById('enabled-toggle').checked = enabled !== false;

    // Stats
    const s = stats || { scanned: 0, fake: 0, real: 0 };
    document.getElementById('stat-scanned').textContent = s.scanned;
    document.getElementById('stat-fake').textContent = s.fake;
    document.getElementById('stat-real').textContent = s.real;
});

// ─── Backend health check ────────────────────────────────────────────────────
async function checkBackend() {
    const dot = document.getElementById('backend-dot');
    const status = document.getElementById('backend-status');
    try {
        const res = await fetch(`${BACKEND}/health`, { signal: AbortSignal.timeout(3000) });
        if (res.ok) {
            dot.className = 'status-dot green';
            status.textContent = 'AI Engine Online';
        } else {
            throw new Error();
        }
    } catch {
        dot.className = 'status-dot red';
        status.textContent = 'Backend Offline';
    }
}
checkBackend();

// ─── Provider hint ────────────────────────────────────────────────────────────
document.getElementById('api-provider').addEventListener('change', (e) => {
    const hint = document.getElementById('api-hint');
    if (e.target.value === 'gemini') {
        hint.innerHTML = 'Get a free Gemini key at <a href="https://aistudio.google.com/app/apikey" target="_blank">aistudio.google.com</a>';
    } else {
        hint.innerHTML = 'Get your OpenAI key at <a href="https://platform.openai.com/api-keys" target="_blank">platform.openai.com</a>';
    }
});

// ─── Save settings ────────────────────────────────────────────────────────────
document.getElementById('save-btn').addEventListener('click', () => {
    const apiKey = document.getElementById('api-key').value.trim();
    const apiProvider = document.getElementById('api-provider').value;
    const enabled = document.getElementById('enabled-toggle').checked;

    chrome.storage.local.set({ apiKey, apiProvider, enabled }, () => {
        const msg = document.getElementById('saved-msg');
        msg.textContent = '✅ Settings saved!';
        setTimeout(() => msg.textContent = '', 2500);
    });
});

// ─── Enable toggle ────────────────────────────────────────────────────────────
document.getElementById('enabled-toggle').addEventListener('change', (e) => {
    chrome.storage.local.set({ enabled: e.target.checked });
});
