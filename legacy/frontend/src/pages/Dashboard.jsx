import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { VerdictBadge } from '../components/VerdictBadge';
import { SkeletonRow } from '../components/SkeletonRow';
import StatsRow from '../components/StatsRow';
import { detectionsAPI, alertsAPI } from '../services/api';

async function loadRecent() {
    try {
        return await detectionsAPI.getHistory({ limit: 10 });
    } catch {
        return [];
    }
}

const THREAT_LEVELS = {
    LOW: { color: '#2DC653', bg: '#0d2c18', label: 'LOW RISK' },
    MEDIUM: { color: '#F4A261', bg: '#2c1f0d', label: 'MEDIUM RISK' },
    HIGH: { color: '#E63946', bg: '#2c0d10', label: 'HIGH RISK' },
    CRITICAL: { color: '#E63946', bg: '#2c0d10', label: 'CRITICAL', pulse: true },
};

// StatCard extracted to shared StatsRow component

export default function Dashboard() {
    const navigate = useNavigate();
    const [stats, setStats] = useState(null);
    const [recent, setRecent] = useState([]);
    const [loading, setLoading] = useState(true);
    const [threatLevel, setThreatLevel] = useState('LOW');

    const loadData = useCallback(async () => {
        try {
            const [s, alerts] = await Promise.all([
                detectionsAPI.getStats(),
                alertsAPI.getAlerts({ limit: 10 }).catch(() => ({ items: [] })),
            ]);
            setStats({
                scans_today: s.total_scans_today ?? 0,
                deepfakes_today: s.fake_detections_today ?? 0,
                real_verdicts_today: s.real_verdicts_today ?? 0,
                high_risk_count: s.fake_detections_today ?? 0,
                avg_risk_score_today: s.avg_risk_score_today ?? 0,
            });
            const history = await loadRecent();
            setRecent(history);
            const items = Array.isArray(alerts) ? alerts : (alerts?.items ?? alerts ?? []);
            const highCount = items.filter(a => a.severity === 'high' || (a.confidence && a.confidence > 0.7)).length;
            if (highCount >= 5) setThreatLevel('CRITICAL');
            else if (highCount >= 2) setThreatLevel('HIGH');
            else if (highCount >= 1) setThreatLevel('MEDIUM');
            else setThreatLevel('LOW');
        } catch (_) {
            setStats({ scans_today: 0, deepfakes_today: 0, high_risk_count: 0, avg_risk_score_today: 0 });
            setRecent([]);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => { loadData(); }, [loadData]);
    useEffect(() => {
        const interval = setInterval(loadData, 30000);
        return () => clearInterval(interval);
    }, [loadData]);

    const tl = THREAT_LEVELS[threatLevel];

    return (
        <div className='p-6 max-w-7xl mx-auto'>

            {/* ── Threat level banner ── */}
            <div className='flex items-center justify-between mb-6 rounded-xl px-5 py-3'
                style={{ background: tl.bg, border: `1px solid ${tl.color}33` }}>
                <div className='flex items-center gap-3'>
                    <div className={`w-3 h-3 rounded-full ${tl.pulse ? 'animate-pulse' : ''}`}
                        style={{ background: tl.color }} />
                    <span className='font-bold tracking-widest text-sm'
                        style={{ color: tl.color }}>THREAT LEVEL: {tl.label}</span>
                </div>
                <button onClick={() => navigate('/scan')}
                    className='text-xs px-4 py-1.5 rounded-md font-semibold transition-all
                           hover:opacity-80'
                    style={{ background: 'var(--cyan)', color: '#0A1628' }}>
                    + New Scan
                </button>
            </div>

            {/* ── Stat cards ── */}
            <div className='mb-8'>
                <StatsRow stats={stats} loading={loading} />
            </div>

            {/* ── Recent activity table ── */}
            <div className='rounded-xl overflow-hidden'
                style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                <div className='px-5 py-4 flex items-center justify-between'
                    style={{ borderBottom: '1px solid var(--border)' }}>
                    <h2 className='font-semibold text-sm tracking-wide
                         uppercase' style={{ color: 'var(--text-secondary)' }}>
                        Recent Activity
                    </h2>
                    <button onClick={() => navigate('/alerts')}
                        className='text-xs hover:underline'
                        style={{ color: 'var(--cyan)' }}>View all →</button>
                </div>
                <table className='w-full'>
                    <thead>
                        <tr style={{ borderBottom: '1px solid var(--border)' }}>
                            {['Source', 'Verdict', 'Confidence', 'Models', 'Time'].map(h => (
                                <th key={h} className='px-4 py-3 text-left text-xs font-semibold
                                        uppercase tracking-wider'
                                    style={{ color: 'var(--text-muted)' }}>{h}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {loading
                            ? Array.from({ length: 5 }).map((_, i) => (
                                <SkeletonRow key={i} cols={5} />
                            ))
                            : recent.length === 0
                                ? (
                                    <tr><td colSpan={5} className='px-4 py-12 text-center'>
                                        <div className='flex flex-col items-center gap-3'>
                                            <span className='text-4xl'>🔬</span>
                                            <p style={{ color: 'var(--text-muted)' }}>No scans yet</p>
                                            <button onClick={() => navigate('/scan')}
                                                className='text-sm px-4 py-2 rounded-md font-medium'
                                                style={{ background: 'var(--cyan)', color: '#0A1628' }}>
                                                Run your first scan
                                            </button>
                                        </div>
                                    </td></tr>
                                )
                                : recent.map(row => (
                                    <tr key={row.task_id || row.id}
                                        onClick={() => navigate(`/alerts/${row.task_id || row.id}`)}
                                        className='cursor-pointer transition-colors hover:bg-white/5'
                                        style={{ borderBottom: '1px solid var(--border)' }}>
                                        <td className='px-4 py-3 text-sm max-w-xs truncate'
                                            title={row.filename}>
                                            {row.filename || 'File upload'}
                                        </td>
                                        <td className='px-4 py-3'>
                                            <VerdictBadge verdict={row.verdict} size='sm' />
                                        </td>
                                        <td className='px-4 py-3 text-sm font-mono'
                                            style={{ color: 'var(--text-secondary)' }}>
                                            {row.confidence != null ? `${(Number(row.confidence) * 100).toFixed(1)}%` : row.final_score != null ? `${(Number(row.final_score) * 100).toFixed(1)}%` : '—'}
                                        </td>
                                        <td className='px-4 py-3 text-sm'
                                            style={{ color: 'var(--text-muted)' }}>
                                            {row.meta_data?.model_breakdown ? Object.keys(row.meta_data.model_breakdown).length + ' models' : '—'}
                                        </td>
                                        <td className='px-4 py-3 text-xs'
                                            style={{ color: 'var(--text-muted)' }}>
                                            {row.timestamp ? new Date(row.timestamp).toLocaleTimeString() : row.created_at ? new Date(row.created_at).toLocaleTimeString() : '—'}
                                        </td>
                                    </tr>
                                ))
                        }
                    </tbody>
                </table>
            </div>
        </div>
    );
}
