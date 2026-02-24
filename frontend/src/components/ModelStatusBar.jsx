/**
 * ModelStatusBar — Active HF model indicator
 *
 * Shown at the top of Advanced pages to display the currently
 * loaded model, device, and last-inference latency.
 *
 * Props:
 *   model    : string (model key)
 *   device   : "cuda" | "cpu"
 *   latencyMs: number | null
 *   loaded   : bool
 */

import React, { useEffect, useState } from 'react';

const API = 'http://localhost:8000';

const MODEL_SHORT = {
    vit_deepfake_primary: 'ViT-DeepFake (HF)',
    vit_deepfake_secondary: 'ViT-DeepFake-2 (HF)',
    efficientnet_b4: 'EfficientNet-B4',
    xception: 'XceptionNet',
};

export default function ModelStatusBar() {
    const [meta, setMeta] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch(`${API}/model-metadata`)
            .then(r => r.json())
            .then(d => { setMeta(d); setLoading(false); })
            .catch(() => setLoading(false));
    }, []);

    if (loading) {
        return (
            <div style={barStyle('#334155')}>
                <span style={{ color: '#475569', fontSize: '12px' }}>⏳ Loading model status…</span>
            </div>
        );
    }

    if (!meta) {
        return (
            <div style={barStyle('#7f1d1d')}>
                <span style={{ color: '#ef4444', fontSize: '12px' }}>⚠️ Backend offline — start backend with uvicorn</span>
            </div>
        );
    }

    const loaded = meta.models?.filter(m => m.loaded) || [];
    const isCuda = meta.device === 'cuda';
    const gpuName = meta.gpu?.name;

    return (
        <div style={{ ...barStyle('#1e293b'), display: 'flex', alignItems: 'center', gap: '20px', justifyContent: 'space-between' }}>
            {/* Device */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{
                    width: '8px', height: '8px', borderRadius: '50%', background: isCuda ? '#4ade80' : '#f59e0b',
                    boxShadow: isCuda ? '0 0 6px #4ade80' : 'none'
                }} />
                <span style={{ color: '#94a3b8', fontSize: '12px' }}>
                    {isCuda ? `GPU: ${gpuName || 'CUDA'}` : 'CPU Mode'}
                </span>
            </div>

            {/* Loaded models */}
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <span style={{ color: '#475569', fontSize: '11px' }}>Models loaded:</span>
                {loaded.length === 0 ? (
                    <span style={{ color: '#475569', fontSize: '11px' }}>None (first inference auto-loads ViT)</span>
                ) : (
                    loaded.map(m => (
                        <span key={m.name} style={{
                            fontSize: '11px', padding: '2px 8px', borderRadius: '4px',
                            background: '#14532d', color: '#4ade80'
                        }}>
                            {MODEL_SHORT[m.name] || m.name}
                        </span>
                    ))
                )}
            </div>

            {/* HF badge */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ fontSize: '14px' }}>🤗</span>
                <span style={{ color: '#64748b', fontSize: '11px' }}>HuggingFace</span>
                <span style={{ fontSize: '11px', padding: '2px 8px', borderRadius: '4px', background: '#1e3a5f', color: '#60a5fa' }}>
                    {meta.models?.length || 0} models available
                </span>
            </div>
        </div>
    );
}

const barStyle = (bg) => ({
    background: bg, borderRadius: '10px', padding: '10px 16px',
    border: '1px solid #334155', marginBottom: '20px',
});
