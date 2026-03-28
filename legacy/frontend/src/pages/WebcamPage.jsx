import React, { useRef, useEffect, useState, useCallback } from 'react';

const API = 'http://localhost:8000';
const SAMPLE_INTERVAL_MS = 800; // send webcam frame every 800ms

export default function WebcamPage() {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const intervalRef = useRef(null);
    const historyRef = useRef([]);   // stores {t, confidence, verdict} for chart

    const [stream, setStream] = useState(null);
    const [running, setRunning] = useState(false);
    const [result, setResult] = useState(null);
    const [history, setHistory] = useState([]);   // [{t, confidence, verdict}]
    const [frameCount, setFrameCount] = useState(0);
    const [fakeCount, setFakeCount] = useState(0);
    const [error, setError] = useState('');
    const [selectedModel, setSelectedModel] = useState('vit_deepfake_primary');
    const [latency, setLatency] = useState(null);

    // Start webcam
    const startWebcam = useCallback(async () => {
        setError('');
        try {
            const mediaStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
            setStream(mediaStream);
            if (videoRef.current) videoRef.current.srcObject = mediaStream;
        } catch (e) {
            setError(`Camera access denied: ${e.message}. Allow camera in browser settings.`);
        }
    }, []);

    // Stop webcam + analysis
    const stopWebcam = useCallback(() => {
        if (stream) { stream.getTracks().forEach(t => t.stop()); setStream(null); }
        if (intervalRef.current) clearInterval(intervalRef.current);
        setRunning(false);
    }, [stream]);

    // Capture frame from video → base64
    const captureFrame = () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas) return null;
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        canvas.getContext('2d').drawImage(video, 0, 0);
        return canvas.toDataURL('image/jpeg', 0.7);
    };

    // Send frame to backend
    const analyzeFrame = useCallback(async (frameIdx) => {
        const frame = captureFrame();
        if (!frame) return;
        const t0 = Date.now();
        try {
            const r = await fetch(`${API}/live-detection-advanced`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frame, models: [selectedModel], frame_idx: frameIdx }),
            });
            if (!r.ok) return;
            const data = await r.json();
            const lat = Date.now() - t0;
            setLatency(lat);
            setResult(data);

            const entry = { t: frameIdx, confidence: data.confidence, verdict: data.verdict, ts: new Date().toLocaleTimeString() };
            historyRef.current = [...historyRef.current.slice(-29), entry];
            setHistory([...historyRef.current]);
            setFrameCount(c => c + 1);
            if (data.verdict === 'FAKE') setFakeCount(c => c + 1);
        } catch { /* backend busy — skip frame */ }
    }, [selectedModel]);

    // Toggle live detection
    const toggleDetection = () => {
        if (running) {
            clearInterval(intervalRef.current);
            setRunning(false);
        } else {
            if (!stream) { startWebcam(); return; }
            let idx = 0;
            intervalRef.current = setInterval(() => { analyzeFrame(idx++); }, SAMPLE_INTERVAL_MS);
            setRunning(true);
        }
    };

    // When stream starts, auto-begin detection
    useEffect(() => {
        if (stream && !running) {
            let idx = 0;
            intervalRef.current = setInterval(() => { analyzeFrame(idx++); }, SAMPLE_INTERVAL_MS);
            setRunning(true);
        }
    }, [stream]);

    useEffect(() => () => stopWebcam(), []);

    const isFake = result?.verdict === 'FAKE';
    const fakeRatio = frameCount > 0 ? Math.round((fakeCount / frameCount) * 100) : 0;

    return (
        <div style={{ padding: '28px', background: '#0f172a', minHeight: '100vh', color: '#e2e8f0', fontFamily: 'Inter,sans-serif' }}>

            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '24px' }}>
                <div>
                    <h1 style={{ fontSize: '24px', fontWeight: 700, color: '#00f3ff', margin: 0 }}>📷 Live Webcam Detection</h1>
                    <p style={{ color: '#64748b', margin: '4px 0 0', fontSize: '13px' }}>Real-time deepfake detection via webcam → HuggingFace ViT models</p>
                </div>
                <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)}
                    style={{ padding: '8px 12px', background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', color: '#e2e8f0', cursor: 'pointer' }}>
                    <option value="vit_deepfake_primary">ViT Deepfake Primary</option>
                    <option value="vit_deepfake_secondary">ViT Deepfake Secondary</option>
                    <option value="efficientnet_b4">EfficientNet-B4</option>
                </select>
            </div>

            {error && (
                <div style={{ padding: '12px 16px', background: '#1e293b', border: '1px solid #ef4444', borderRadius: '10px', color: '#ef4444', marginBottom: '16px', fontSize: '13px' }}>
                    ⚠️ {error}
                </div>
            )}

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 360px', gap: '20px' }}>

                {/* Video feed */}
                <div>
                    <div style={{
                        position: 'relative', borderRadius: '16px', overflow: 'hidden', background: '#020617',
                        border: `2px solid ${running ? (isFake ? '#ef4444' : '#22c55e') : '#334155'}`,
                        boxShadow: running ? `0 0 20px ${isFake ? '#ef444433' : '#22c55e33'}` : 'none',
                        transition: 'border-color 0.3s, box-shadow 0.3s'
                    }}>
                        <video ref={videoRef} autoPlay muted playsInline
                            style={{ width: '100%', display: 'block', borderRadius: '14px' }} />
                        <canvas ref={canvasRef} style={{ display: 'none' }} />

                        {/* Overlay verdict badge */}
                        {result && (
                            <div style={{
                                position: 'absolute', top: '16px', left: '16px', padding: '8px 16px', borderRadius: '999px',
                                background: isFake ? '#ef444488' : '#22c55e88', backdropFilter: 'blur(8px)',
                                color: '#fff', fontWeight: 800, fontSize: '16px', letterSpacing: '0.1em'
                            }}>
                                {isFake ? '⚠️ FAKE' : '✅ REAL'} — {Math.round((result.confidence || 0) * 100)}%
                            </div>
                        )}

                        {/* Animated pulse when running */}
                        {running && (
                            <div style={{ position: 'absolute', top: '16px', right: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <div style={{
                                    width: '10px', height: '10px', borderRadius: '50%', background: '#ef4444',
                                    animation: 'pulse 0.8s infinite', boxShadow: '0 0 8px #ef4444'
                                }} />
                                <span style={{ color: '#fff', fontSize: '12px', fontWeight: 600 }}>LIVE</span>
                            </div>
                        )}

                        {/* No-camera placeholder */}
                        {!stream && (
                            <div style={{
                                position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center',
                                background: '#020617', flexDirection: 'column', gap: '12px'
                            }}>
                                <div style={{ fontSize: '64px' }}>📷</div>
                                <div style={{ color: '#475569', fontSize: '14px' }}>Camera not started</div>
                            </div>
                        )}
                    </div>

                    {/* Controls */}
                    <div style={{ display: 'flex', gap: '12px', marginTop: '16px' }}>
                        <button onClick={stream ? toggleDetection : startWebcam}
                            style={{
                                flex: 1, padding: '12px', borderRadius: '10px', border: 'none', cursor: 'pointer', fontWeight: 700, fontSize: '14px',
                                background: running ? '#7f1d1d' : 'linear-gradient(90deg,#1e40af,#0891b2)', color: '#fff'
                            }}>
                            {!stream ? '📷 Start Camera' : running ? '⏹ Stop Detection' : '▶ Resume Detection'}
                        </button>
                        {stream && (
                            <button onClick={stopWebcam}
                                style={{
                                    padding: '12px 20px', borderRadius: '10px', border: '1px solid #334155',
                                    background: 'transparent', color: '#64748b', cursor: 'pointer', fontWeight: 600
                                }}>
                                ✕ Off
                            </button>
                        )}
                    </div>
                </div>

                {/* Right panel — Stats + mini chart */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>

                    {/* Stat cards */}
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                        {[
                            { label: 'Frames', value: frameCount },
                            { label: 'Fake Frames', value: fakeCount, color: '#ef4444' },
                            { label: 'Fake Ratio', value: `${fakeRatio}%`, color: fakeRatio > 30 ? '#ef4444' : '#22c55e' },
                            { label: 'Latency', value: latency ? `${latency}ms` : '—', color: '#f59e0b' },
                        ].map(s => (
                            <div key={s.label} style={{ background: '#1e293b', borderRadius: '10px', padding: '14px' }}>
                                <div style={{ color: '#64748b', fontSize: '11px', fontWeight: 600, textTransform: 'uppercase' }}>{s.label}</div>
                                <div style={{ color: s.color || '#e2e8f0', fontSize: '22px', fontWeight: 700, marginTop: '4px' }}>{s.value}</div>
                            </div>
                        ))}
                    </div>

                    {/* Confidence timeline mini-chart */}
                    <div style={{ background: '#1e293b', borderRadius: '14px', padding: '16px' }}>
                        <div style={{ color: '#64748b', fontSize: '12px', fontWeight: 600, marginBottom: '12px' }}>Confidence Timeline</div>
                        <div style={{ display: 'flex', alignItems: 'flex-end', gap: '2px', height: '80px' }}>
                            {history.length === 0 ? (
                                <div style={{ color: '#334155', fontSize: '12px', margin: 'auto' }}>Waiting for frames…</div>
                            ) : (
                                history.map((h, i) => (
                                    <div key={i} title={`${h.ts}: ${h.verdict} (${Math.round(h.confidence * 100)}%)`}
                                        style={{
                                            flex: 1, background: h.verdict === 'FAKE' ? '#ef4444' : '#22c55e',
                                            height: `${Math.round(h.confidence * 100)}%`, minHeight: '3px', borderRadius: '2px 2px 0 0',
                                            opacity: 0.4 + 0.6 * (i / history.length)
                                        }} />
                                ))
                            )}
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '4px', fontSize: '10px', color: '#475569' }}>
                            <span>←30 frames</span><span>now→</span>
                        </div>
                    </div>

                    {/* Current result breakdown */}
                    {result && (
                        <div style={{ background: '#1e293b', borderRadius: '14px', padding: '16px', border: `1px solid ${isFake ? '#ef444433' : '#22c55e33'}` }}>
                            <div style={{ color: '#64748b', fontSize: '12px', fontWeight: 600, marginBottom: '10px' }}>Current Frame</div>
                            <div style={{ fontSize: '20px', fontWeight: 700, color: isFake ? '#ef4444' : '#22c55e', marginBottom: '8px' }}>
                                {isFake ? '⚠️ DEEPFAKE' : '✅ AUTHENTIC'}
                            </div>
                            <div style={{ background: '#0f172a', borderRadius: '6px', height: '8px', marginBottom: '8px' }}>
                                <div style={{
                                    height: '100%', width: `${Math.round((result.confidence || 0) * 100)}%`,
                                    background: isFake ? '#ef4444' : '#22c55e', borderRadius: '6px', transition: 'width 0.3s'
                                }} />
                            </div>
                            <div style={{ color: '#64748b', fontSize: '12px' }}>{Math.round((result.confidence || 0) * 100)}% confidence · frame #{result.frame_idx}</div>
                        </div>
                    )}
                </div>
            </div>

            <style>{`@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }`}</style>
        </div>
    );
}
