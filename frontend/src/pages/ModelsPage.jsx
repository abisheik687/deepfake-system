import React, { useEffect, useState } from 'react';

const API = 'http://localhost:8000';

const MODEL_INFO = {
    efficientnet_b4: { icon: '⚡', color: '#00f3ff', tagline: 'Best accuracy/speed tradeoff' },
    efficientnet_b0: { icon: '🚀', color: '#4ade80', tagline: 'Fastest on CPU (~800ms)' },
    xception: { icon: '🎯', color: '#f59e0b', tagline: 'FaceForensics++ benchmark' },
    vit_b16: { icon: '🤖', color: '#a78bfa', tagline: 'Transformer-based detection' },
    mesonet4: { icon: '💡', color: '#fb923c', tagline: 'Ultra-lightweight (~5MB)' },
};

export default function ModelsPage() {
    const [models, setModels] = useState([]);
    const [activeModel, setActiveModel] = useState(null);
    const [loading, setLoading] = useState(true);
    const [loadingModel, setLoadingModel] = useState(null);
    const [benchmark, setBenchmark] = useState(null);
    const [device, setDevice] = useState(null);
    const [error, setError] = useState('');

    useEffect(() => { fetchModels(); }, []);

    const fetchModels = async () => {
        setLoading(true);
        try {
            const r = await fetch(`${API}/api/models/`);
            const d = await r.json();
            setModels(d.models || []);
            setActiveModel(d.active_model);
            setDevice(d.device);
        } catch { setError('Backend offline'); }
        finally { setLoading(false); }
    };

    const loadModel = async (name) => {
        setLoadingModel(name);
        setError('');
        try {
            const r = await fetch(`${API}/api/models/load`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_name: name }),
            });
            if (!r.ok) { const e = await r.json(); setError(e.detail); return; }
            setActiveModel(name);
            await fetchModels();
        } catch (e) { setError(String(e)); }
        finally { setLoadingModel(null); }
    };

    const runBenchmark = async () => {
        setBenchmark(null);
        try {
            const r = await fetch(`${API}/api/models/benchmark`);
            setBenchmark(await r.json());
        } catch { setError('Benchmark failed'); }
    };

    const exportOnnx = async () => {
        try {
            const r = await fetch(`${API}/api/models/export`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
            const d = await r.json();
            alert(d.message);
        } catch { setError('Export failed'); }
    };

    return (
        <div style={{ padding: '32px', background: '#0f172a', minHeight: '100vh', color: '#e2e8f0' }}>
            {/* Header */}
            <div style={{ marginBottom: '32px' }}>
                <h1 style={{ fontSize: '28px', fontWeight: 700, color: '#00f3ff', margin: 0 }}>
                    🧠 AI Models
                </h1>
                <p style={{ color: '#94a3b8', margin: '8px 0 0' }}>
                    Select, load, benchmark and export deepfake detection models
                </p>
            </div>

            {/* Device Info */}
            {device && (
                <div style={card({ border: device.type === 'cuda' ? '#4ade80' : '#64748b' })}>
                    <span style={{ color: device.type === 'cuda' ? '#4ade80' : '#94a3b8', fontWeight: 600 }}>
                        {device.type === 'cuda' ? '🟢 GPU Detected' : '🔵 CPU Mode'}
                    </span>
                    {device.gpu && <span style={{ marginLeft: '12px', color: '#94a3b8' }}>{device.gpu.name} ({device.gpu.memory_gb} GB)</span>}
                    {device.type === 'cpu' && <span style={{ marginLeft: '12px', color: '#64748b' }}>Install CUDA for 10-20× faster inference</span>}
                </div>
            )}

            {/* Error */}
            {error && <div style={card({ border: '#ef4444', color: '#ef4444' })}>⚠️ {error}</div>}

            {/* Action buttons */}
            <div style={{ display: 'flex', gap: '12px', margin: '20px 0' }}>
                <button onClick={runBenchmark} style={btn('#1e40af')}>⏱ Benchmark Active Model</button>
                <button onClick={exportOnnx} style={btn('#7c3aed')}>📦 Export ONNX</button>
            </div>

            {/* Benchmark Result */}
            {benchmark && (
                <div style={card({ border: '#00f3ff' })}>
                    <b style={{ color: '#00f3ff' }}>Benchmark Result</b>
                    <span style={{ marginLeft: '16px' }}>Model: <b>{benchmark.model}</b></span>
                    <span style={{ marginLeft: '16px' }}>Latency: <b style={{ color: '#4ade80' }}>{benchmark.latency_ms}ms</b></span>
                    <span style={{ marginLeft: '16px' }}>Device: <b>{benchmark.device?.type}</b></span>
                </div>
            )}

            {/* Model cards */}
            {loading ? (
                <div style={{ color: '#64748b', marginTop: '40px', textAlign: 'center' }}>Loading models…</div>
            ) : (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: '20px', marginTop: '24px' }}>
                    {models.map(m => {
                        const info = MODEL_INFO[m.name] || { icon: '🔬', color: '#64748b', tagline: '' };
                        const isActive = m.name === activeModel;
                        const isLoading = loadingModel === m.name;
                        return (
                            <div key={m.name} style={{
                                background: '#1e293b',
                                border: `2px solid ${isActive ? info.color : '#334155'}`,
                                borderRadius: '16px', padding: '24px',
                                transition: 'border-color 0.2s',
                                position: 'relative', overflow: 'hidden',
                            }}>
                                {isActive && (
                                    <div style={{ position: 'absolute', top: '12px', right: '12px', background: info.color, color: '#000', fontSize: '11px', fontWeight: 700, padding: '3px 8px', borderRadius: '4px' }}>
                                        ACTIVE
                                    </div>
                                )}
                                <div style={{ fontSize: '32px', marginBottom: '8px' }}>{info.icon}</div>
                                <div style={{ fontSize: '18px', fontWeight: 700, color: info.color }}>{m.name}</div>
                                <div style={{ color: '#94a3b8', fontSize: '13px', margin: '4px 0 12px' }}>{info.tagline}</div>
                                <div style={{ color: '#64748b', fontSize: '12px', marginBottom: '16px' }}>{m.description}</div>

                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '16px' }}>
                                    <Stat label="AUC (FF++)" value={m.auc_ff ? (m.auc_ff * 100).toFixed(1) + '%' : 'N/A'} />
                                    <Stat label="CPU latency" value={m.speed_cpu_ms ? m.speed_cpu_ms + 'ms' : 'N/A'} />
                                    <Stat label="GPU latency" value={m.speed_gpu_ms ? m.speed_gpu_ms + 'ms' : 'N/A'} />
                                    <Stat label="Fine-tuned" value={m.finetuned ? '✅ Yes' : '⬜ No'} />
                                </div>

                                <button
                                    onClick={() => loadModel(m.name)}
                                    disabled={isActive || isLoading}
                                    style={{
                                        ...btn(isActive ? '#334155' : info.color.replace(')', ', 0.2)').replace('rgb', 'rgba')),
                                        width: '100%',
                                        color: isActive ? '#64748b' : info.color,
                                        border: `1px solid ${isActive ? '#334155' : info.color}`,
                                        cursor: isActive ? 'default' : 'pointer',
                                    }}
                                >
                                    {isLoading ? '⏳ Loading…' : isActive ? '✓ Active' : 'Load Model'}
                                </button>
                            </div>
                        );
                    })}
                </div>
            )}

            {/* Model comparison table */}
            <div style={{ marginTop: '48px' }}>
                <h2 style={{ color: '#e2e8f0', fontSize: '18px', fontWeight: 600 }}>📊 Model Comparison</h2>
                <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '16px' }}>
                    <thead>
                        <tr style={{ borderBottom: '1px solid #334155', color: '#64748b', fontSize: '12px', textAlign: 'left' }}>
                            <th style={th()}>Model</th>
                            <th style={th()}>Architecture</th>
                            <th style={th()}>FF++ AUC</th>
                            <th style={th()}>CPU ms</th>
                            <th style={th()}>GPU ms</th>
                            <th style={th()}>Size</th>
                            <th style={th()}>Best For</th>
                        </tr>
                    </thead>
                    <tbody>
                        {[
                            ['EfficientNet-B4', 'CNN (EfficientNet)', '99.1%', '2200', '85', '~85MB', 'Best overall accuracy'],
                            ['EfficientNet-B0', 'CNN (EfficientNet)', '96.2%', '800', '30', '~20MB', 'Fast CPU inference'],
                            ['XceptionNet', 'Depthwise CNN', '99.0%', '3000', '100', '~88MB', 'FF++ benchmark'],
                            ['ViT-B/16', 'Transformer', '98.8%', '4200', '120', '~330MB', 'Transformer baseline'],
                            ['MesoNet-4', 'Lightweight CNN', '89.5%', '150', '12', '~5MB', 'Resource-constrained'],
                        ].map(([name, arch, auc, cpu, gpu, size, best]) => (
                            <tr key={name} style={{ borderBottom: '1px solid #1e293b', fontSize: '13px' }}>
                                <td style={td()}><b style={{ color: '#00f3ff' }}>{name}</b></td>
                                <td style={td()}>{arch}</td>
                                <td style={td()}><b style={{ color: '#4ade80' }}>{auc}</b></td>
                                <td style={td()}>{cpu}ms</td>
                                <td style={td()}>{gpu}ms</td>
                                <td style={td()}>{size}</td>
                                <td style={td({ color: '#94a3b8' })}>{best}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

// ── Helpers ───────────────────────────────────────────────────────────────────
const Stat = ({ label, value }) => (
    <div style={{ background: '#0f172a', borderRadius: '8px', padding: '8px 12px' }}>
        <div style={{ color: '#64748b', fontSize: '11px' }}>{label}</div>
        <div style={{ color: '#e2e8f0', fontWeight: 600, fontSize: '14px' }}>{value}</div>
    </div>
);

const card = ({ border = '#334155', color = '#e2e8f0' } = {}) => ({
    background: '#1e293b', border: `1px solid ${border}`, borderRadius: '10px',
    padding: '14px 20px', color, marginBottom: '12px',
});
const btn = (bg) => ({
    padding: '10px 20px', borderRadius: '8px', border: 'none',
    background: bg, cursor: 'pointer', fontWeight: 600, fontSize: '14px',
    color: '#fff', transition: 'opacity 0.2s',
});
const th = () => ({ padding: '10px 16px', textTransform: 'uppercase', fontWeight: 600 });
const td = ({ color = '#e2e8f0' } = {}) => ({ padding: '12px 16px', color });
