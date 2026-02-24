import React, { useEffect, useState, useRef } from 'react';

const API = 'http://localhost:8000';

const DATASETS = [
    { name: 'FaceForensics++', short: 'ff++', videos: 4000, size: '512 GB', access: 'Free (academic request)', url: 'https://github.com/ondyari/FaceForensics', color: '#00f3ff', auc: '99.1%' },
    { name: 'Celeb-DF v2', short: 'celeb_df', videos: 6229, size: '6.4 GB', access: 'Free — GitHub', url: 'https://github.com/yuezunli/celeb-deepfakeforensics', color: '#4ade80', auc: '97.8%' },
    { name: 'DFDC', short: 'dfdc', videos: 100000, size: '470 GB', access: 'Free — Kaggle', url: 'https://www.kaggle.com/c/deepfake-detection-challenge', color: '#f59e0b', auc: '82.1%' },
    { name: 'WildDeepfake', short: 'wild', videos: 7314, size: '10 GB', access: 'Free — GitHub', url: 'https://github.com/deepfakeinthewild/deepfake-in-the-wild', color: '#a78bfa', auc: '95.4%' },
    { name: 'DeepFakeTIMIT', short: 'timit', videos: 640, size: '2.3 GB', access: 'Free (form)', url: 'https://www.idiap.ch/en/dataset/deepfaketimit', color: '#fb923c', auc: '98.2%' },
];

export default function TrainingPage() {
    const [status, setStatus] = useState(null);
    const [form, setForm] = useState({ model: 'efficientnet_b4', dataset: 'ff++', epochs: 20, batch_size: 16, learning_rate: 0.0001 });
    const [running, setRunning] = useState(false);
    const [error, setError] = useState('');
    const pollRef = useRef(null);
    const logRef = useRef(null);

    useEffect(() => { pollStatus(); return () => clearInterval(pollRef.current); }, []);

    const pollStatus = () => {
        const tick = async () => {
            try {
                const r = await fetch(`${API}/api/training/status`);
                const d = await r.json();
                setStatus(d);
                setRunning(d.status === 'running');
            } catch { /* backend offline */ }
        };
        tick();
        pollRef.current = setInterval(tick, 2000);
    };

    const startTraining = async () => {
        setError('');
        try {
            const r = await fetch(`${API}/api/training/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(form),
            });
            const d = await r.json();
            if (!r.ok) throw new Error(d.detail);
            setRunning(true);
        } catch (e) { setError(String(e)); }
    };

    const stopTraining = async () => {
        await fetch(`${API}/api/training/stop`, { method: 'POST' });
        setRunning(false);
    };

    // Auto-scroll log
    useEffect(() => {
        if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
    }, [status?.log]);

    const pct = status ? Math.round(status.progress_pct || 0) : 0;

    return (
        <div style={{ padding: '32px', background: '#0f172a', minHeight: '100vh', color: '#e2e8f0' }}>
            <h1 style={{ fontSize: '28px', fontWeight: 700, color: '#00f3ff', margin: 0 }}>🎯 Training Pipeline</h1>
            <p style={{ color: '#94a3b8', margin: '8px 0 24px' }}>
                Train deepfake detection models on open-source datasets
            </p>

            {/* Datasets grid */}
            <h2 style={{ color: '#e2e8f0', fontSize: '16px', fontWeight: 600, marginBottom: '16px' }}>📚 Supported Datasets</h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '16px', marginBottom: '40px' }}>
                {DATASETS.map(d => (
                    <div key={d.short} style={{ background: '#1e293b', border: `1px solid ${d.color}22`, borderRadius: '12px', padding: '20px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                            <div style={{ color: d.color, fontWeight: 700, fontSize: '15px' }}>{d.name}</div>
                            <a href={d.url} target="_blank" rel="noreferrer" style={{ color: '#64748b', fontSize: '12px' }}>↗ Link</a>
                        </div>
                        <div style={{ marginTop: '12px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                            {[['Videos', d.videos.toLocaleString()], ['Size', d.size], ['Access', d.access], ['Best AUC¹', d.auc]].map(([k, v]) => (
                                <div key={k} style={{ background: '#0f172a', borderRadius: '6px', padding: '6px 10px' }}>
                                    <div style={{ color: '#64748b', fontSize: '10px' }}>{k}</div>
                                    <div style={{ color: '#e2e8f0', fontSize: '12px', fontWeight: 600 }}>{v}</div>
                                </div>
                            ))}
                        </div>
                    </div>
                ))}
            </div>
            <p style={{ color: '#64748b', fontSize: '12px', marginBottom: '40px' }}>¹ AUC when EfficientNet-B4 fine-tuned on that dataset</p>

            {/* Training config */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px' }}>

                {/* Config form */}
                <div style={{ background: '#1e293b', borderRadius: '16px', padding: '28px' }}>
                    <h3 style={{ margin: '0 0 20px', color: '#e2e8f0' }}>⚙️ Training Config</h3>

                    {[
                        { label: 'Model', key: 'model', type: 'select', options: ['efficientnet_b4', 'efficientnet_b0', 'xception', 'vit_b16', 'mesonet4'] },
                        { label: 'Dataset', key: 'dataset', type: 'select', options: ['ff++', 'celeb_df', 'dfdc', 'wild', 'all'] },
                        { label: 'Epochs', key: 'epochs', type: 'number', min: 1, max: 200 },
                        { label: 'Batch Size', key: 'batch_size', type: 'number', min: 4, max: 128 },
                        { label: 'Learning Rate', key: 'learning_rate', type: 'number', step: 0.00001, min: 0.00001, max: 0.01 },
                    ].map(({ label, key, type, options, ...rest }) => (
                        <div key={key} style={{ marginBottom: '16px' }}>
                            <label style={{ color: '#94a3b8', fontSize: '12px', fontWeight: 600, display: 'block', marginBottom: '6px' }}>{label}</label>
                            {type === 'select' ? (
                                <select value={form[key]} disabled={running}
                                    onChange={e => setForm(f => ({ ...f, [key]: e.target.value }))}
                                    style={selectStyle}>
                                    {options.map(o => <option key={o} value={o}>{o}</option>)}
                                </select>
                            ) : (
                                <input type="number" value={form[key]} disabled={running} {...rest}
                                    onChange={e => setForm(f => ({ ...f, [key]: parseFloat(e.target.value) }))}
                                    style={inputStyle} />
                            )}
                        </div>
                    ))}

                    {error && <div style={{ color: '#ef4444', fontSize: '13px', marginBottom: '16px' }}>⚠️ {error}</div>}

                    <div style={{ display: 'flex', gap: '12px' }}>
                        <button onClick={startTraining} disabled={running} style={{ ...btnStyle('#1e40af'), flex: 1 }}>
                            {running ? '⏳ Training…' : '▶ Start Training'}
                        </button>
                        {running && (
                            <button onClick={stopTraining} style={{ ...btnStyle('#7f1d1d'), flex: 1 }}>⏹ Stop</button>
                        )}
                    </div>

                    <div style={{ marginTop: '16px', background: '#0f172a', borderRadius: '8px', padding: '14px', fontSize: '12px', color: '#64748b' }}>
                        <b style={{ color: '#94a3b8' }}>💡 Setup required before training:</b>
                        <pre style={{ margin: '8px 0 0', lineHeight: 1.6 }}>{`# 1. Download a dataset (e.g. Celeb-DF)
# 2. Run face extraction:
python training/preprocessing/extract_faces.py \\
  --dataset celeb_df \\
  --input path/to/downloaded_videos \\
  --output data/datasets/celeb_df
# 3. Click Start Training`}</pre>
                    </div>
                </div>

                {/* Status panel */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

                    {/* Progress */}
                    <div style={{ background: '#1e293b', borderRadius: '16px', padding: '24px' }}>
                        <h3 style={{ margin: '0 0 16px', color: '#e2e8f0' }}>📈 Progress</h3>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                            <div style={{
                                width: '12px', height: '12px', borderRadius: '50%',
                                background: status?.status === 'running' ? '#4ade80' : status?.status === 'completed' ? '#00f3ff' : '#64748b',
                                boxShadow: status?.status === 'running' ? '0 0 8px #4ade80' : 'none',
                            }} />
                            <span style={{ fontWeight: 600 }}>{status?.status || 'idle'}</span>
                            {status?.model && <span style={{ color: '#64748b' }}>— {status.model}</span>}
                        </div>

                        {/* Progress bar */}
                        <div style={{ background: '#0f172a', borderRadius: '999px', height: '8px', overflow: 'hidden' }}>
                            <div style={{ width: `${pct}%`, height: '100%', background: 'linear-gradient(90deg, #1e40af, #00f3ff)', transition: 'width 0.5s ease', borderRadius: '999px' }} />
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontSize: '12px', color: '#64748b' }}>
                            <span>Epoch {status?.epoch || 0} / {status?.total_epochs || 0}</span>
                            <span>{pct}%</span>
                        </div>

                        {/* Metrics */}
                        {(status?.loss || status?.val_auc) && (
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginTop: '16px' }}>
                                {status?.loss && <MetricCard label="Train Loss" value={status.loss.toFixed(4)} color="#f59e0b" />}
                                {status?.val_auc && <MetricCard label="Val AUC" value={status.val_auc.toFixed(4)} color="#4ade80" />}
                            </div>
                        )}
                    </div>

                    {/* Training Log */}
                    <div style={{ background: '#1e293b', borderRadius: '16px', padding: '24px', flex: 1 }}>
                        <h3 style={{ margin: '0 0 12px', color: '#e2e8f0', fontSize: '15px' }}>📋 Training Log</h3>
                        <div ref={logRef} style={{ background: '#020617', borderRadius: '8px', padding: '12px', height: '220px', overflowY: 'auto', fontFamily: 'monospace', fontSize: '12px', color: '#94a3b8', lineHeight: 1.8 }}>
                            {status?.log?.length ? status.log.map((l, i) => (
                                <div key={i} style={{ color: l.includes('AUC') ? '#4ade80' : l.includes('Error') ? '#ef4444' : '#94a3b8' }}>
                                    {l}
                                </div>
                            )) : <span style={{ color: '#334155' }}>Waiting for training job…</span>}
                        </div>
                    </div>
                </div>
            </div>

            {/* Dataset integration instructions */}
            <div style={{ background: '#1e293b', borderRadius: '16px', padding: '28px', marginTop: '32px' }}>
                <h3 style={{ margin: '0 0 16px', color: '#e2e8f0' }}>🛠️ Dataset Setup Guide</h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
                    {[
                        { title: 'FaceForensics++', steps: ['Go to github.com/ondyari/FaceForensics', 'Fill free academic access form', 'Download via their download.py script', 'Extract to data/datasets/ff++'] },
                        { title: 'Celeb-DF v2', steps: ['Go to github.com/yuezunli/celeb-deepfakeforensics', 'Fill one-page Google Form', 'Download YouTube video IDs + annotations', 'Extract to data/datasets/celeb_df'] },
                    ].map(({ title, steps }) => (
                        <div key={title}>
                            <b style={{ color: '#00f3ff' }}>{title}</b>
                            <ol style={{ color: '#94a3b8', fontSize: '13px', paddingLeft: '20px', lineHeight: 2 }}>
                                {steps.map((s, i) => <li key={i}>{s}</li>)}
                            </ol>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

const MetricCard = ({ label, value, color }) => (
    <div style={{ background: '#0f172a', borderRadius: '10px', padding: '14px', textAlign: 'center' }}>
        <div style={{ color: '#64748b', fontSize: '11px', fontWeight: 600 }}>{label}</div>
        <div style={{ color, fontSize: '22px', fontWeight: 700, marginTop: '4px' }}>{value}</div>
    </div>
);

const selectStyle = { width: '100%', padding: '10px 12px', background: '#0f172a', border: '1px solid #334155', borderRadius: '8px', color: '#e2e8f0', fontSize: '14px' };
const inputStyle = { ...selectStyle };
const btnStyle = (bg) => ({ padding: '12px', borderRadius: '10px', border: 'none', background: bg, color: '#fff', fontWeight: 700, cursor: 'pointer', fontSize: '14px' });
