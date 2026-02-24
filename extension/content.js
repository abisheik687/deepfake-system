/**
 * KAVACH.AI Extension — Content Script v1.3 (Direct Fetch Architecture)
 *
 * KEY CHANGE: Content script now calls the KAVACH.AI backend DIRECTLY via
 * fetch() — no background service worker in the analysis path.
 *
 * Why this works:
 *   • Chrome allows content scripts to fetch URLs listed in host_permissions
 *   • http://localhost:8000/* is in host_permissions → direct fetch works
 *   • Eliminates all message-passing reliability issues
 *
 * For CDN images (Instagram, Facebook):
 *   Content script tries to fetch the image bytes (with credentials).
 *   If CDN blocks it, falls back to sending the URL to the backend for download.
 *
 * For video frames (YouTube Shorts, TikTok, Reels):
 *   Canvas captures a frame as base64 → POSTs directly to backend.
 */

const BACKEND = 'http://localhost:8000';
const MIN_SIZE_PX = 60;
const ANALYSIS_DELAY_MS = 1000;
const VIDEO_INTERVAL_MS = 4000;
const MAX_QUEUE = 50;

// Off-screen canvas for video frames
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');

// ─── State ────────────────────────────────────────────────────────────────────
const seenImgs = new WeakSet();
const seenVids = new WeakSet();
const queue = [];
let running = false;
let enabled = true;

chrome.storage.local.get('enabled', r => { if (r.enabled === false) enabled = false; });
chrome.storage.onChanged.addListener(c => { if (c.enabled) enabled = c.enabled.newValue; });

// ─── Stats ────────────────────────────────────────────────────────────────────
function recordStat(verdict) {
    chrome.storage.local.get('stats', r => {
        const s = r.stats || { scanned: 0, fake: 0, real: 0 };
        s.scanned++;
        if (verdict === 'FAKE') s.fake++;
        if (verdict === 'REAL') s.real++;
        chrome.storage.local.set({ stats: s });
    });
}

// ─── URL extraction ───────────────────────────────────────────────────────────
function getSrc(el) {
    const candidates = [
        el.currentSrc, el.src,
        el.dataset?.src, el.dataset?.lazySrc,
        el.getAttribute('data-srcset')?.split(',')[0]?.trim().split(' ')[0],
        el.getAttribute('srcset')?.split(',')[0]?.trim().split(' ')[0],
    ];
    for (const c of candidates) {
        if (c && c.startsWith('http')) return c;
    }
    return null;
}

// ─── Blob → base64 ────────────────────────────────────────────────────────────
function blobToBase64(blob) {
    return new Promise((res, rej) => {
        const r = new FileReader();
        r.onloadend = () => res(r.result);
        r.onerror = rej;
        r.readAsDataURL(blob);
    });
}

// ─── ══════════════════════════════════════════════════════════════════════════
//     BACKEND CALLS — direct fetch, no background worker
// ─── ══════════════════════════════════════════════════════════════════════════

async function analyzeBase64(base64, source) {
    const res = await fetch(`${BACKEND}/api/analyze-url/frame`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: base64, source }),
    });
    if (!res.ok) throw new Error(`Backend ${res.status}`);
    return res.json();
}

async function analyzeUrl(url) {
    const res = await fetch(`${BACKEND}/api/analyze-url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
    });
    if (!res.ok) throw new Error(`Backend ${res.status}`);
    return res.json();
}

// ─── Main image analysis ──────────────────────────────────────────────────────
async function runImageAnalysis(img) {
    const src = getSrc(img);
    if (!src) return null;

    // Try to fetch the image bytes directly (content script can do this)
    try {
        const resp = await fetch(src, {
            credentials: 'include',
            cache: 'force-cache',
            headers: { Accept: 'image/avif,image/webp,image/png,image/jpeg,*/*' },
        });
        if (resp.ok) {
            const blob = await resp.blob();
            if (blob.size > 1500 && blob.type.startsWith('image/')) {
                const b64 = await blobToBase64(blob);
                return await analyzeBase64(b64, location.hostname);
            }
        }
    } catch (_) { /* CDN rejected — fall through */ }

    // Fallback: send URL for backend to download
    return await analyzeUrl(src);
}

// ─── ══════════════════════════════════════════════════════════════════════════
//     BADGE — zero innerHTML (Trusted Types safe)
// ─── ══════════════════════════════════════════════════════════════════════════

const BADGE_CFG = {
    REAL: { e: '✅', l: 'REAL', c: 'kavach-real' },
    FAKE: { e: '⚠️', l: 'FAKE', c: 'kavach-fake' },
    NO_FACE: { e: '👁', l: 'No Face', c: 'kavach-no_face' },
    PENDING: { e: '⏳', l: '…', c: 'kavach-pending' },
    ERROR: { e: '❌', l: 'Error', c: 'kavach-error' },
    UNAVAILABLE: { e: '🔌', l: 'N/A', c: 'kavach-unavailable' },
};

function mk(tag, cls, txt) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt !== undefined) e.textContent = txt;
    return e;
}

function getWrap(el) {
    let p = el.parentElement;
    for (let i = 0; i < 5 && p; i++) {
        if (getComputedStyle(p).position !== 'static') return p;
        p = p.parentElement;
    }
    if (el.parentElement) el.parentElement.style.position = 'relative';
    return el.parentElement;
}

function dropBadge(el) {
    if (el._kb) { el._kb.remove(); delete el._kb; }
}

function putBadge(el, verdict, conf, result) {
    dropBadge(el);
    const wrap = getWrap(el);
    if (!wrap) return;

    const cfg = BADGE_CFG[verdict] || BADGE_CFG.ERROR;
    const pct = (verdict === 'REAL' || verdict === 'FAKE') ? ` ${Math.round(conf * 100)}%` : '';
    const b = mk('div', `kavach-badge ${cfg.c}`);
    b.appendChild(mk('span', 'kavach-icon', cfg.e));
    b.appendChild(mk('span', 'kavach-label', cfg.l + pct));
    wrap.appendChild(b);
    el._kb = b;

    if ((verdict === 'FAKE' || verdict === 'REAL') && result) {
        b.style.cursor = 'pointer';
        b.title = verdict === 'FAKE' ? 'Click for AI explanation' : 'Appears authentic';
        b.addEventListener('click', ev => { ev.stopPropagation(); ev.preventDefault(); showPanel(el, result); });
    }
}

// ─── Explanation panel — zero innerHTML ───────────────────────────────────────
function showPanel(target, result) {
    document.querySelectorAll('.kavach-panel').forEach(p => p.remove());
    const isFake = result.verdict === 'FAKE';
    const pct = Math.round(result.confidence * 100);

    const panel = mk('div', 'kavach-panel');

    // Header
    const hdr = mk('div', 'kavach-panel-header');
    hdr.appendChild(mk('span', `kavach-panel-verdict ${isFake ? 'red' : 'green'}`,
        `${isFake ? '⚠️ FAKE DETECTED' : '✅ AUTHENTIC'} — ${pct}% confidence`));
    const close = mk('button', 'kavach-panel-close', '✕');
    hdr.appendChild(close);
    panel.appendChild(hdr);

    // Body
    const body = mk('div', 'kavach-panel-body');

    const s1 = mk('div', 'kavach-panel-section');
    s1.appendChild(mk('b', null, '🤖 AI Analysis'));
    const exp = mk('div', 'kavach-explanation');
    exp.id = 'kavach-exp';
    exp.appendChild(mk('span', isFake ? 'kavach-loading' : 'kavach-muted',
        isFake ? 'Searching web for context… (requires API key)' : 'No manipulation signs detected.'));
    s1.appendChild(exp);
    body.appendChild(s1);

    const s2 = mk('div', 'kavach-panel-section kavach-signals');
    s2.appendChild(mk('b', null, '📊 Detection Signals'));
    const ul = mk('ul');
    [[`Faces`, String(result.faces?.length ?? 0)], [`Confidence`, `${pct}%`], [`Site`, location.hostname]]
        .forEach(([k, v]) => { const li = mk('li', null, `${k}: `); li.appendChild(mk('b', null, v)); ul.appendChild(li); });
    s2.appendChild(ul);
    body.appendChild(s2);
    panel.appendChild(body);

    panel.appendChild(mk('div', 'kavach-panel-footer', 'KAVACH.AI Deepfake Shield'));
    document.body.appendChild(panel);
    close.addEventListener('click', () => panel.remove());

    if (isFake) {
        chrome.runtime.sendMessage({
            type: 'EXPLAIN_FAKE',
            imageUrl: getSrc(target) || '',
            sourcePage: location.hostname,
            caption: (target.alt || '').slice(0, 200),
            verdict: result.verdict,
            confidence: result.confidence,
        }, resp => {
            const el = document.getElementById('kavach-exp');
            if (!el) return;
            el.textContent = '';
            el.appendChild(mk('span', null,
                resp?.explanation || (resp?.status === 'no_key' ? '⚙️ Add API key in extension popup.' : 'No explanation.')));
        });
    }
}

// ─── ══════════════════════════════════════════════════════════════════════════
//     IMAGE QUEUE
// ─── ══════════════════════════════════════════════════════════════════════════

function enqueue(img) {
    if (!enabled || seenImgs.has(img) || queue.length >= MAX_QUEUE) return;
    if (!getSrc(img)) return;
    seenImgs.add(img);
    queue.push(img);
    if (!running) setTimeout(drain, 150);
}

async function drain() {
    if (running) return;
    running = true;
    while (queue.length > 0) {
        const img = queue.shift();
        if (!img || !document.contains(img)) continue;
        const src = getSrc(img);
        if (!src) continue;
        const w = img.naturalWidth || img.width;
        const h = img.naturalHeight || img.height;
        if (w < MIN_SIZE_PX || h < MIN_SIZE_PX) continue;

        putBadge(img, 'PENDING', 0);
        try {
            const r = await runImageAnalysis(img);
            if (r && r.verdict !== 'SKIP') { putBadge(img, r.verdict, r.confidence, r); recordStat(r.verdict); }
            else dropBadge(img);
        } catch (e) {
            console.warn('[KAVACH] img:', e.message);
            dropBadge(img);
        }
        await new Promise(r => setTimeout(r, ANALYSIS_DELAY_MS));
    }
    running = false;
}

// ─── ══════════════════════════════════════════════════════════════════════════
//     VIDEO — YouTube Shorts, Reels, TikTok
// ─── ══════════════════════════════════════════════════════════════════════════

function watchVideo(video) {
    if (seenVids.has(video)) return;
    seenVids.add(video);

    const run = async () => {
        if (!enabled || video.paused || video.ended || video.readyState < 2) return;
        if ((video.videoWidth || 0) < MIN_SIZE_PX) return;
        try {
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 360;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const b64 = canvas.toDataURL('image/jpeg', 0.65);
            putBadge(video, 'PENDING', 0);
            const r = await analyzeBase64(b64, location.hostname);
            if (r && r.verdict !== 'SKIP') { putBadge(video, r.verdict, r.confidence, r); recordStat(r.verdict); }
            else dropBadge(video);
        } catch (e) {
            console.warn('[KAVACH] video:', e.message);
            dropBadge(video);
        }
    };

    video.addEventListener('playing', run, { once: true });
    const t = setInterval(() => {
        if (!document.contains(video)) { clearInterval(t); return; }
        run();
    }, VIDEO_INTERVAL_MS);
}

// ─── ══════════════════════════════════════════════════════════════════════════
//     OBSERVERS
// ─── ══════════════════════════════════════════════════════════════════════════

const vp = new IntersectionObserver(es => { for (const e of es) if (e.isIntersecting) enqueue(e.target); }, { threshold: 0.2 });

new MutationObserver(muts => {
    if (!enabled) return;
    for (const m of muts) {
        if (m.type === 'attributes' && m.target.tagName === 'IMG') {
            seenImgs.delete(m.target); dropBadge(m.target);
            if (getSrc(m.target)) enqueue(m.target);
            continue;
        }
        for (const n of m.addedNodes) {
            if (n.nodeType !== 1) continue;
            if (n.tagName === 'IMG') { vp.observe(n); enqueue(n); }
            if (n.tagName === 'VIDEO') { n.readyState >= 2 ? watchVideo(n) : n.addEventListener('loadeddata', () => watchVideo(n)); }
            n.querySelectorAll?.('img').forEach(i => { vp.observe(i); enqueue(i); });
            n.querySelectorAll?.('video').forEach(v => { v.readyState >= 2 ? watchVideo(v) : v.addEventListener('loadeddata', () => watchVideo(v)); });
        }
    }
}).observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['src', 'data-src', 'srcset'] });

// Initial scan
document.querySelectorAll('img').forEach(i => { vp.observe(i); enqueue(i); });
document.querySelectorAll('video').forEach(v => { v.readyState >= 2 ? watchVideo(v) : v.addEventListener('loadeddata', () => watchVideo(v)); });

console.log('[KAVACH] v1.3 active on', location.hostname, '— direct backend mode');
