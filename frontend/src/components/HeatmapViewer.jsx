/**
 * HeatmapViewer — Grad-CAM heatmap overlay component
 *
 * Displays the Grad-CAM heatmap returned from /analyze-image-advanced
 * side-by-side with the original image, with an opacity slider overlay.
 *
 * Props:
 *   originalSrc : string  — base64 or URL of original image
 *   heatmapSrc  : string  — base64 data URI of Grad-CAM overlay (from backend)
 *   verdict     : "FAKE" | "REAL"
 *   confidence  : float   0–1
 */

import React, { useState } from 'react';

export default function HeatmapViewer({ originalSrc, heatmapSrc, verdict, confidence }) {
    const [opacity, setOpacity] = useState(0.7);
    const [viewMode, setViewMode] = useState('overlay'); // 'overlay' | 'side' | 'heatmap'
    const [showInfo, setShowInfo] = useState(false);

    const isFake = verdict === 'FAKE';
    const pct = Math.round((confidence || 0) * 100);
    const accentClr = isFake ? '#ef4444' : '#22c55e';

    if (!heatmapSrc) {
        return (
            <div style={{ background: '#1e293b', borderRadius: '12px', padding: '20px', textAlign: 'center', color: '#475569', fontSize: '13px' }}>
                🔥 Grad-CAM heatmap not available<br />
                <span style={{ fontSize: '11px' }}>Enable "Generate Grad-CAM heatmap" in options and re-analyse</span>
            </div>
        );
    }

    return (
        <div style={{ background: '#1e293b', borderRadius: '16px', padding: '20px' }}>
            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
                <div>
                    <span style={{ color: '#94a3b8', fontSize: '12px', fontWeight: 600, textTransform: 'uppercase' }}>🔥 Grad-CAM Explainability</span>
                    <div style={{ color: accentClr, fontSize: '11px', marginTop: '2px' }}>
                        {isFake ? `Model focused on manipulated regions (${pct}% fake)` : `Natural face regions detected (${pct}% real)`}
                    </div>
                </div>
                <button onClick={() => setShowInfo(!showInfo)}
                    style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer', fontSize: '16px' }} title="What is Grad-CAM?">
                    ℹ️
                </button>
            </div>

            {/* Info panel */}
            {showInfo && (
                <div style={{ background: '#0f172a', borderRadius: '8px', padding: '12px', marginBottom: '14px', fontSize: '12px', color: '#94a3b8', lineHeight: 1.7 }}>
                    <b style={{ color: '#00f3ff' }}>What is Grad-CAM?</b><br />
                    Gradient-weighted Class Activation Map highlights which regions of the image the neural network "looks at" when making its deepfake/real decision.
                    <br /><b style={{ color: '#ef4444' }}>Red/hot regions</b> = high influence on FAKE verdict.
                    <br /><b style={{ color: '#4ade80' }}>Blue/cool regions</b> = lower influence (likely authentic).
                </div>
            )}

            {/* View mode selector */}
            <div style={{ display: 'flex', gap: '4px', marginBottom: '14px' }}>
                {[['overlay', 'Overlay'], ['side', 'Side by Side'], ['heatmap', 'Heatmap Only']].map(([mode, label]) => (
                    <button key={mode} onClick={() => setViewMode(mode)}
                        style={{
                            padding: '6px 14px', borderRadius: '6px', border: 'none', cursor: 'pointer', fontSize: '12px', fontWeight: 600,
                            background: viewMode === mode ? '#1e40af' : '#0f172a', color: viewMode === mode ? '#fff' : '#64748b'
                        }}>
                        {label}
                    </button>
                ))}
            </div>

            {/* Overlay mode */}
            {viewMode === 'overlay' && (
                <div>
                    <div style={{ position: 'relative', borderRadius: '10px', overflow: 'hidden' }}>
                        <img src={originalSrc} alt="original" style={{ width: '100%', display: 'block', borderRadius: '10px' }} />
                        <img src={heatmapSrc} alt="heatmap" style={{
                            position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover',
                            borderRadius: '10px', opacity, mixBlendMode: 'multiply',
                        }} />
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginTop: '10px' }}>
                        <span style={{ color: '#64748b', fontSize: '11px' }}>Opacity</span>
                        <input type="range" min={0} max={1} step={0.05} value={opacity}
                            onChange={e => setOpacity(parseFloat(e.target.value))}
                            style={{ flex: 1, accentColor: '#00f3ff' }} />
                        <span style={{ color: '#94a3b8', fontSize: '11px', width: '32px', textAlign: 'right' }}>{Math.round(opacity * 100)}%</span>
                    </div>
                </div>
            )}

            {/* Side-by-side mode */}
            {viewMode === 'side' && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                    <div>
                        <div style={{ color: '#64748b', fontSize: '11px', marginBottom: '4px', textAlign: 'center' }}>Original</div>
                        <img src={originalSrc} alt="original" style={{ width: '100%', borderRadius: '8px' }} />
                    </div>
                    <div>
                        <div style={{ color: '#64748b', fontSize: '11px', marginBottom: '4px', textAlign: 'center' }}>Grad-CAM</div>
                        <img src={heatmapSrc} alt="heatmap" style={{ width: '100%', borderRadius: '8px' }} />
                    </div>
                </div>
            )}

            {/* Heatmap only */}
            {viewMode === 'heatmap' && (
                <img src={heatmapSrc} alt="Grad-CAM heatmap" style={{ width: '100%', borderRadius: '10px' }} />
            )}

            {/* Legend */}
            <div style={{ display: 'flex', gap: '16px', marginTop: '12px', justifyContent: 'center' }}>
                {[['🔴 High activation (FAKE signal)', '#ef4444'], ['🔵 Low activation (authentic)', '#3b82f6'], ['🟡 Medium', '#f59e0b']].map(([l, c]) => (
                    <div key={l} style={{ fontSize: '10px', color: '#64748b' }}>{l}</div>
                ))}
            </div>
        </div>
    );
}
