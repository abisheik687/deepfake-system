/**
 * StatsRow — shared stat card grid
 * Used by: Dashboard.jsx, AuditPage.jsx
 */

export function StatCard({ icon, label, value, delta, color }) {
    return (
        <div className='rounded-xl p-5 flex flex-col gap-2'
            style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
            <div className='flex items-center justify-between'>
                <span className='text-2xl'>{icon}</span>
                {delta != null && (
                    <span className={`text-xs font-medium px-2 py-0.5 rounded-full
                           ${delta >= 0 ? 'text-red-400 bg-red-900/30'
                            : 'text-green-400 bg-green-900/30'}`}>
                        {delta >= 0 ? '+' : ''}{delta}% today
                    </span>
                )}
            </div>
            <div className='text-3xl font-bold' style={{ color: color ?? 'var(--text-primary)' }}>
                {value ?? '—'}
            </div>
            <div className='text-xs' style={{ color: 'var(--text-muted)' }}>{label}</div>
        </div>
    );
}

export default function StatsRow({ stats, loading }) {
    return (
        <div className='grid grid-cols-2 lg:grid-cols-4 gap-4'>
            <StatCard
                icon='🔍'
                label='Scans Today'
                value={loading ? '…' : (stats?.scans_today ?? stats?.total_scans ?? 0)}
            />
            <StatCard
                icon='🎭'
                label='Deepfakes Detected'
                value={loading ? '…' : (stats?.deepfakes_today ?? stats?.fake_detections ?? 0)}
                color='var(--verdict-fake)'
            />
            <StatCard
                icon='✅'
                label='Real Verdicts'
                value={loading ? '…' : (stats?.real_verdicts_today ?? stats?.real_verdicts ?? 0)}
            />
            <StatCard
                icon='📊'
                label='Avg Risk Score'
                value={loading ? '…' : (
                    stats?.avg_risk_score_today != null
                        ? (stats.avg_risk_score_today * 100).toFixed(1) + '%'
                        : stats?.avg_confidence != null
                            ? (stats.avg_confidence * 100).toFixed(1) + '%'
                            : '—'
                )}
                color={
                    (stats?.avg_risk_score_today ?? stats?.avg_confidence ?? 0) > 0.5
                        ? '#E63946'
                        : undefined
                }
            />
        </div>
    );
}
