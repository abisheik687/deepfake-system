import React, { useState, useCallback } from 'react';

const API = 'http://localhost:8000';

const MODELS = [
    { key: 'vit_deepfake_primary', label: 'ViT Deepfake (Primary)', desc: 'prithivMLmods/Deepfake-image-detect — real fine-tuned weights', color: '#00f3ff', icon: '🧠' },
    { key: 'vit_deepfake_secondary', label: 'ViT Deepfake (Secondary)', desc: 'dima806/deepfake_vs_real_image_detection', color: '#4ade80', icon: '🔬' },
    { key: 'efficientnet_b4', label: 'EfficientNet-B4', desc: 'timm pretrained — fine-tune ready', color: '#f59e0b', icon: '⚡' },
    { key: 'xception', label: 'XceptionNet', desc: 'FaceForensics++ benchmark architecture', color: '#a78bfa', icon: '🎯' },
];

export default function AdvancedPage() {
    const [selectedModels, setSelectedModels] = useState(['vit_deepfake_primary']);
    const [image, setImage] = useState(null);
    const [imageSrc, setImageSrc] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [detectFaces, setDetectFaces] = useState(true);
    const [returnHeatmap, setReturnHeatmap] = useState(true);
    const [metadata, setMetadata] = useState(null);
    const [tab, setTab] = useState('image'); // 'image' | 'url' | 'metadata'
    const [urlInput, setUrlInput] = useState('');

    const toggleModel = (key) => {
        setSelectedModels(prev =>
            prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key]
        );
    };

    // --- Image upload ---
    const onDrop = useCallback((e) => {
        e.preventDefault();
        const file = e.dataTransfer?.files[0] || e.target.files?.[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (ev) => { setImageSrc(ev.target.result); setImage(ev.target.result); setResult(null); };
        reader.readAsDataURL(file);
    }, []);

    const analyzeImage = async () => {
        if (!image) return;
        setLoading(true); setError(''); setResult(null);
        try {
            const r = await fetch(`${API}/analyze-image-advanced`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    data: image,
                    models: selectedModels,
                    return_heatmap: returnHeatmap,
                    detect_faces: detectFaces,
                }),
            });
            if (!r.ok) { const d = await r.json(); throw new Error(d.detail || 'Analysis failed'); }
            setResult(await r.json());
        } catch (e) { setError(e.message); }
        finally { setLoading(false); }
    };

    // --- URL analysis ---
    const analyzeUrl = async () => {
        if (!urlInput) return;
        setLoading(true); setError(''); setResult(null);
        try {
            const r = await fetch(`${API}/analyze-url-advanced`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: urlInput, models: selectedModels, return_heatmap: returnHeatmap }),
            });
            if (!r.ok) { const d = await r.json(); throw new Error(d.detail || 'Failed'); }
            setResult(await r.json());
        } catch (e) { setError(e.message); }
        finally { setLoading(false); }
    };

    // --- Model metadata ---
    const loadMetadata = async () => {
        try {
            const r = await fetch(`${API}/model-metadata`);
            setMetadata(await r.json());
        } catch { setError('Cannot reach backend'); }
    };

    const verdict = result?.verdict;
    const conf = result?.confidence;

    return (
        <div style={{ padding: '28px', background: '#0f172a', minHeight: '100vh', color: '#e2e8f0', fontFamily: 'Inter, sans-serif' }}>

            {/* Header */}
            <div style={{ marginBottom: '28px' }}>
                <h1 style={{ fontSize: '26px', fontWeight: 700, color: '#00f3ff', margin: 0 }}>
                    🔬 Advanced DeepFake Analysis
                </h1>
                <p style={{ color: '#64748b', margin: '6px 0 0', fontSize: '14px' }}>
                    Powered by real HuggingFace fine-tuned deepfake detection models
                </p>
            </div>

            {/* Tab bar */}
            <div style={{ display: 'flex', gap: '4px', marginBottom: '24px' }}>
                {[['image', '📷 Image Upload'], ['url', '🔗 URL'], ['metadata', '📋 Model Info']].map(([t, label]) => (
                    <button key={t} onClick={() => { setTab(t); if (t === 'metadata') loadMetadata(); }}
                        style={{
                            padding: '8px 20px', borderRadius: '8px', border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: '13px',
                            background: tab === t ? '#1e40af' : '#1e293b', color: tab === t ? '#fff' : '#64748b'
                        }}>
                        {label}
                    </button>
                ))}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '24px' }}>

                {/* Left panel */}
                <div>

                    {/* Image tab */}
                    {tab === 'image' && (
                        <div>
                            <div onDrop={onDrop} onDragOver={e => e.preventDefault()} onClick={() => document.getElementById('adv-file-input').click()}
                                style={{
                                    border: `2px dashed ${imageSrc ? '#00f3ff55' : '#334155'}`, borderRadius: '16px', padding: '40px', textAlign: 'center',
                                    cursor: 'pointer', minHeight: '180px', display: 'flex', alignItems: 'center', justifyContent: 'center',
                                    background: imageSrc ? '#0f172a' : '#111827', position: 'relative', overflow: 'hidden'
                                }}>
                                {imageSrc ? (
                                    <img src={imageSrc} alt="uploaded" style={{ maxHeight: '280px', maxWidth: '100%', borderRadius: '8px' }} />
                                ) : (
                                    <div style={{ color: '#64748b' }}>
                                        <div style={{ fontSize: '40px', marginBottom: '12px' }}>📷</div>
                                        <div>Drop an image here or click to upload</div>
                                        <div style={{ fontSize: '12px', marginTop: '4px' }}>JPEG · PNG · WEBP</div>
                                    </div>
                                )}
                            </div>
                            <input id="adv-file-input" type="file" accept="image/*" hidden onChange={onDrop} />

                            {imageSrc && (
                                <button onClick={analyzeImage} disabled={loading || selectedModels.length === 0}
                                    style={{
                                        marginTop: '16px', width: '100%', padding: '14px', borderRadius: '10px', border: 'none',
                                        background: loading ? '#334155' : 'linear-gradient(90deg,#1e40af,#0891b2)',
                                        color: '#fff', fontWeight: 700, fontSize: '15px', cursor: loading ? 'default' : 'pointer'
                                    }}>
                                    {loading ? '⏳ Analysing with HF models…' : '🔬 Run Advanced Analysis'}
                                </button>
                            )}
                        </div>
                    )}

                    {/* URL tab */}
                    {tab === 'url' && (
                        <div>
                            <input value={urlInput} onChange={e => setUrlInput(e.target.value)} placeholder="https://example.com/image.jpg"
                                style={{
                                    width: '100%', padding: '12px 16px', background: '#1e293b', border: '1px solid #334155',
                                    borderRadius: '10px', color: '#e2e8f0', fontSize: '14px', boxSizing: 'border-box'
                                }} />
                            <button onClick={analyzeUrl} disabled={loading || !urlInput}
                                style={{
                                    marginTop: '12px', width: '100%', padding: '12px', borderRadius: '10px', border: 'none',
                                    background: 'linear-gradient(90deg,#1e40af,#0891b2)', color: '#fff', fontWeight: 700, cursor: 'pointer'
                                }}>
                                {loading ? '⏳ Analysing…' : '🔗 Analyse from URL'}
                            </button>
                        </div>
                    )}

                    {/* Metadata tab */}
                    {tab === 'metadata' && metadata && (
                        <div style={{ display: 'grid', gap: '12px' }}>
                            {metadata.models?.map(m => (
                                <div key={m.name} style={{
                                    background: '#1e293b', borderRadius: '12px', padding: '16px',
                                    border: `1px solid ${m.loaded ? '#4ade8044' : '#33415544'}`
                                }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                        <span style={{ fontWeight: 700, color: m.loaded ? '#4ade80' : '#94a3b8' }}>{m.name}</span>
                                        <span style={{
                                            fontSize: '11px', background: m.loaded ? '#14532d' : '#1e293b',
                                            color: m.loaded ? '#4ade80' : '#64748b', padding: '2px 8px', borderRadius: '4px'
                                        }}>
                                            {m.loaded ? '● Loaded' : '○ Not loaded'}
                                        </span>
                                    </div>
                                    <div style={{ color: '#64748b', fontSize: '12px', marginTop: '6px' }}>{m.description}</div>
                                    {m.repo_id && <div style={{ color: '#334155', fontSize: '11px', marginTop: '4px' }}>🤗 {m.repo_id}</div>}
                                    <div style={{ color: '#475569', fontSize: '11px', marginTop: '4px' }}>Size: ~{m.size_mb}MB · Device: {m.device}</div>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Error */}
                    {error && <div style={{
                        marginTop: '16px', padding: '12px 16px', background: '#1e293b', borderRadius: '10px',
                        border: '1px solid #ef4444', color: '#ef4444', fontSize: '13px'
                    }}>⚠️ {error}</div>}

                    {/* Result */}
                    {result && <ResultCard result={result} />}
                </div>

                {/* Right panel — Model selector + options */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <div style={{ background: '#1e293b', borderRadius: '16px', padding: '20px' }}>
                        <h3 style={{ margin: '0 0 14px', fontSize: '14px', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                            Select Models
                        </h3>
                        {MODELS.map(m => (
                            <div key={m.key} onClick={() => toggleModel(m.key)}
                                style={{
                                    padding: '12px', borderRadius: '10px', marginBottom: '8px', cursor: 'pointer',
                                    background: selectedModels.includes(m.key) ? `${m.color}18` : '#0f172a',
                                    border: `1px solid ${selectedModels.includes(m.key) ? m.color : '#334155'}`,
                                    transition: 'all 0.15s'
                                }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <span style={{ fontSize: '18px' }}>{m.icon}</span>
                                    <div>
                                        <div style={{ fontWeight: 600, fontSize: '13px', color: selectedModels.includes(m.key) ? m.color : '#94a3b8' }}>{m.label}</div>
                                        <div style={{ fontSize: '11px', color: '#475569', marginTop: '2px' }}>{m.desc}</div>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>

                    <div style={{ background: '#1e293b', borderRadius: '16px', padding: '20px' }}>
                        <h3 style={{ margin: '0 0 14px', fontSize: '14px', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                            Options
                        </h3>
                        {[
                            [detectFaces, setDetectFaces, 'Extract faces first (more accurate)'],
                            [returnHeatmap, setReturnHeatmap, 'Generate Grad-CAM heatmap'],
                        ].map(([val, setter, label]) => (
                            <div key={label} onClick={() => setter(!val)}
                                style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px', cursor: 'pointer' }}>
                                <div style={{
                                    width: '36px', height: '20px', borderRadius: '999px', background: val ? '#1e40af' : '#334155',
                                    position: 'relative', transition: 'background 0.2s'
                                }}>
                                    <div style={{
                                        width: '16px', height: '16px', borderRadius: '50%', background: '#fff',
                                        position: 'absolute', top: '2px', left: val ? '18px' : '2px', transition: 'left 0.2s'
                                    }} />
                                </div>
                                <span style={{ fontSize: '13px', color: '#94a3b8' }}>{label}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}

function ResultCard({ result }) {
    const isFake = result.verdict === 'FAKE';
    const pct = Math.round((result.confidence || 0) * 100);

    return (
        <div style={{
            marginTop: '20px', background: '#1e293b', borderRadius: '16px', padding: '24px',
            border: `2px solid ${isFake ? '#ef4444' : '#22c55e'}`
        }}>

            {/* Verdict banner */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '20px' }}>
                <div style={{ fontSize: '40px' }}>{isFake ? '⚠️' : '✅'}</div>
                <div>
                    <div style={{ fontSize: '24px', fontWeight: 800, color: isFake ? '#ef4444' : '#22c55e' }}>{result.verdict}</div>
                    <div style={{ color: '#64748b', fontSize: '13px' }}>HuggingFace deepfake detection</div>
                </div>
                <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
                    <div style={{ fontSize: '28px', fontWeight: 700, color: isFake ? '#ef4444' : '#22c55e' }}>{pct}%</div>
                    <div style={{ color: '#64748b', fontSize: '11px' }}>confidence</div>
                </div>
            </div>

            {/* Confidence bar */}
            <div style={{ background: '#0f172a', borderRadius: '999px', height: '8px', marginBottom: '20px' }}>
                <div style={{
                    height: '100%', width: `${pct}%`, borderRadius: '999px',
                    background: isFake ? 'linear-gradient(90deg,#ef4444,#f97316)' : 'linear-gradient(90deg,#22c55e,#0ea5e9)',
                    transition: 'width 0.5s ease'
                }} />
            </div>

            {/* Per-model breakdown */}
            {result.fused_scores?.per_model?.length > 0 && (
                <div>
                    <div style={{ color: '#64748b', fontSize: '12px', fontWeight: 600, marginBottom: '8px', textTransform: 'uppercase' }}>Per-Model Breakdown</div>
                    {result.fused_scores.per_model.map(m => (
                        <div key={m.model} style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                            <span style={{ color: '#94a3b8', fontSize: '12px', width: '180px', flexShrink: 0 }}>{m.model}</span>
                            <div style={{ flex: 1, background: '#0f172a', borderRadius: '4px', height: '6px' }}>
                                <div style={{
                                    height: '100%', width: `${Math.round(m.fake_prob * 100)}%`,
                                    background: m.verdict === 'FAKE' ? '#ef4444' : '#22c55e', borderRadius: '4px'
                                }} />
                            </div>
                            <span style={{ color: m.verdict === 'FAKE' ? '#ef4444' : '#22c55e', fontSize: '12px', width: '36px', textAlign: 'right' }}>
                                {Math.round(m.fake_prob * 100)}%
                            </span>
                        </div>
                    ))}
                </div>
            )}

            {/* Heatmap */}
            {result.heatmap_b64 && (
                <div style={{ marginTop: '16px' }}>
                    <div style={{ color: '#64748b', fontSize: '12px', fontWeight: 600, marginBottom: '8px', textTransform: 'uppercase' }}>
                        🔥 Grad-CAM Heatmap
                    </div>
                    <img src={result.heatmap_b64} alt="Grad-CAM heatmap" style={{ width: '100%', borderRadius: '10px' }} />
                </div>
            )}

            {/* Latency */}
            <div style={{ marginTop: '16px', color: '#475569', fontSize: '11px' }}>
                ⏱ {result.latency_ms}ms · {result.faces_found ?? 0} face(s) detected
            </div>
        </div>
    );
}
