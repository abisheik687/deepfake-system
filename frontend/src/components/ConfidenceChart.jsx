/**
 * ConfidenceChart — Recharts real-time confidence graph
 *
 * Renders a line chart of deepfake confidence over time (frame sequence).
 * Uses recharts AreaChart for smooth animated rendering.
 *
 * Props:
 *   data        : [{t, confidence, verdict, ts}]  ← from WebcamPage history
 *   height      : number (default 200)
 *   title       : string
 *   showLegend  : bool
 */

import React from 'react';
import {
    AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, ReferenceLine, Legend,
} from 'recharts';

const FAKE_COLOR = '#ef4444';
const REAL_COLOR = '#22c55e';
const THRESHOLD = 0.50;

// Custom tooltip
const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    const d = payload[0]?.payload;
    if (!d) return null;
    const isFake = d.verdict === 'FAKE';
    return (
        <div style={{
            background: '#1e293b', border: `1px solid ${isFake ? FAKE_COLOR : REAL_COLOR}`,
            borderRadius: '8px', padding: '10px 14px', fontSize: '12px', color: '#e2e8f0',
        }}>
            <div style={{ fontWeight: 700, color: isFake ? FAKE_COLOR : REAL_COLOR, marginBottom: '4px' }}>
                {isFake ? '⚠️ FAKE' : '✅ REAL'}
            </div>
            <div>Confidence: <b>{Math.round((d.confidence || 0) * 100)}%</b></div>
            {d.ts && <div style={{ color: '#64748b', marginTop: '2px' }}>{d.ts}</div>}
        </div>
    );
};

// Dot renderer — colored by verdict
const VerdictDot = (props) => {
    const { cx, cy, payload } = props;
    if (!payload) return null;
    const color = payload.verdict === 'FAKE' ? FAKE_COLOR : REAL_COLOR;
    return <circle cx={cx} cy={cy} r={4} fill={color} stroke="#0f172a" strokeWidth={1.5} />;
};

export function ConfidenceChart({ data = [], height = 200, title = 'Confidence Over Time', showLegend = false }) {
    // Transform: add fake_pct for the area fill (always 0–100)
    const chartData = data.map((d, i) => ({
        ...d,
        frame: i,
        fake_pct: Math.round((d.confidence || 0) * 100),
    }));

    const isEmpty = chartData.length === 0;

    return (
        <div style={{ background: '#1e293b', borderRadius: '14px', padding: '16px' }}>
            {title && (
                <div style={{ color: '#94a3b8', fontSize: '12px', fontWeight: 600, marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    📊 {title}
                </div>
            )}

            {isEmpty ? (
                <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#334155', fontSize: '13px' }}>
                    No data yet — start analysis to see chart
                </div>
            ) : (
                <ResponsiveContainer width="100%" height={height}>
                    <AreaChart data={chartData} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
                        <defs>
                            <linearGradient id="fakeGrad" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={FAKE_COLOR} stopOpacity={0.3} />
                                <stop offset="95%" stopColor={FAKE_COLOR} stopOpacity={0.02} />
                            </linearGradient>
                            <linearGradient id="realGrad" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={REAL_COLOR} stopOpacity={0.3} />
                                <stop offset="95%" stopColor={REAL_COLOR} stopOpacity={0.02} />
                            </linearGradient>
                        </defs>

                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />

                        <XAxis dataKey="frame" tick={{ fontSize: 10, fill: '#475569' }} tickLine={false} axisLine={false} />
                        <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: '#475569' }} tickLine={false} axisLine={false}
                            tickFormatter={v => `${v}%`} />

                        <Tooltip content={<CustomTooltip />} />
                        {showLegend && <Legend wrapperStyle={{ fontSize: '12px', color: '#94a3b8' }} />}

                        {/* 50% threshold line */}
                        <ReferenceLine y={50} stroke="#64748b" strokeDasharray="4 4"
                            label={{ value: 'Threshold', position: 'insideTopRight', fill: '#64748b', fontSize: 10 }} />

                        <Area
                            type="monotone"
                            dataKey="fake_pct"
                            stroke={FAKE_COLOR}
                            strokeWidth={2}
                            fill="url(#fakeGrad)"
                            dot={<VerdictDot />}
                            activeDot={{ r: 6 }}
                            animationDuration={300}
                            name="Fake Probability %"
                        />
                    </AreaChart>
                </ResponsiveContainer>
            )}
        </div>
    );
}

export default ConfidenceChart;
