/**
 * OrchestratorStatusWidget — Dashboard widget showing model health
 *
 * Polls /orchestrator-status every 30s.
 * Shows overall health ring, model list with status dots, cache info.
 *
 * Usage: <OrchestratorStatusWidget /> — drop in anywhere, self-contained.
 */

import React, { useState, useEffect } from 'react';

const API = 'http://localhost:8000';
const POLL_MS = 30_000;

const TIER_LABELS = {
    fast: { label: 'Fast', color: '#22c55e', desc: 'Frequency only' },
    hf_only: { label: 'HF Only', color: '#00f3ff', desc: 'HuggingFace ViT models' },
    balanced: { label: 'Balanced', color: '#a78bfa', desc: 'ViT + EfficientNet + Freq' },
    full: { label: 'Full', color: '#f59e0b', desc: 'All models' },
};

const KIND_ICON = {
    hf_pipeline: '🤗',
    timm: '⚡',
    frequency: '📊',
    custom: '🔧',
};

export default function OrchestratorStatusWidget() {
    const [status, setStatus] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    const fetchStatus = async () => {
        try {
            const r = await fetch(`${API}/orchestrator-status`);
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            setStatus(await r.json());
            setError('');
        } catch (e) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchStatus();
        const id = setInterval(fetchStatus, POLL_MS);
        return () => clearInterval(id);
    }, []);

    // Loading state
    if (loading) return (
        <div style={CARD_STYLE}>
            <div style={{ color: '#475569', fontSize: '13px' }}>⏳ Loading orchestrator status…</div>
        </div>
    );

    // Error state
    if (error || !status) return (
        <div style={{ ...CARD_STYLE, borderColor: '#ef444433' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#ef4444' }} />
                <span style={{ color: '#ef4444', fontSize: '13px' }}>Orchestrator offline</span>
            </div>
            <div style={{ color: '#475569', fontSize: '11px', marginTop: '4px' }}>
                Restart: <code style={{ color: '#94a3b8' }}>py -m uvicorn backend.main:app --reload</code>
            </div>
        </div>
    );

    const healthPct = status.total > 0 ? Math.round((status.healthy / status.total) * 100) : 0;
    const allOk = status.healthy === status.total;
    const accentClr = allOk ? '#22c55e' : status.healthy > 0 ? '#f59e0b' : '#ef4444';

    return (
        <div style={CARD_STYLE}>
            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <div style={{
                        width: '8px', height: '8px', borderRadius: '50%', background: accentClr,
                        boxShadow: `0 0 6px ${accentClr}`, animation: 'pulse 1.5s infinite'
                    }} />
                    <span style={{ color: '#e2e8f0', fontWeight: 700, fontSize: '14px' }}>Orchestrator</span>
                </div>
                <div style={{ fontSize: '12px', color: accentClr, fontWeight: 600 }}>
                    {status.healthy}/{status.total} models healthy
                </div>
            </div>

            {/* Health ring (simple bar) */}
            <div style={{ background: '#0f172a', borderRadius: '999px', height: '6px', marginBottom: '14px' }}>
                <div style={{
                    width: `${healthPct}%`, height: '100%', borderRadius: '999px', background: accentClr,
                    transition: 'width 0.4s ease', boxShadow: `0 0 6px ${accentClr}66`
                }} />
            </div>

            {/* Model list */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginBottom: '12px' }}>
                {status.models?.map(m => (
                    <div key={m.name} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <div style={{
                            width: '6px', height: '6px', borderRadius: '50%', flexShrink: 0,
                            background: m.healthy ? '#22c55e' : '#ef4444'
                        }} />
                        <span style={{ color: '#94a3b8', fontSize: '11px', flex: 1 }}>
                            {KIND_ICON[m.kind] || '•'} {m.name}
                        </span>
                        <span style={{ color: '#475569', fontSize: '10px' }}>w={m.weight}</span>
                        {m.last_latency_ms > 0 && (
                            <span style={{ color: '#334155', fontSize: '10px' }}>{Math.round(m.last_latency_ms)}ms</span>
                        )}
                    </div>
                ))}
            </div>

            {/* Tier presets */}
            <div>
                <div style={{ color: '#475569', fontSize: '10px', fontWeight: 600, marginBottom: '6px', textTransform: 'uppercase' }}>
                    Tier Presets
                </div>
                <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                    {status.tier_presets?.map(t => {
                        const info = TIER_LABELS[t] || { label: t, color: '#64748b' };
                        return (
                            <span key={t} style={{
                                fontSize: '10px', padding: '2px 6px', borderRadius: '4px',
                                background: `${info.color}18`, color: info.color, border: `1px solid ${info.color}33`
                            }}>
                                {info.label}
                            </span>
                        );
                    })}
                </div>
            </div>

            {/* Cache */}
            {status.cache && (
                <div style={{
                    marginTop: '10px', paddingTop: '10px', borderTop: '1px solid #1e293b',
                    fontSize: '10px', color: '#334155', display: 'flex', gap: '12px'
                }}>
                    <span>Cache: <b style={{ color: '#64748b' }}>{status.cache.backend}</b></span>
                    <span>Keys: <b style={{ color: '#64748b' }}>{status.cache.memory_keys}</b></span>
                    {status.cache.redis_ok && <span style={{ color: '#22c55e' }}>●  Redis</span>}
                </div>
            )}

            <style>{`@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }`}</style>
        </div>
    );
}

const CARD_STYLE = {
    background: '#1e293b',
    borderRadius: '14px',
    padding: '16px',
    border: '1px solid #334155',
};
