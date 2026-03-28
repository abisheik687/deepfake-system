import { useState, useEffect } from 'react';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import { VerdictBadge } from '../components/VerdictBadge';
import { ConfidenceRing } from '../components/ConfidenceRing';
import HeatmapViewer from '../components/HeatmapViewer';
import { useToast } from '../hooks/useToast';
import { Toast } from '../components/Toast';
import AgentSummary from '../components/AgentSummary';
import { ModelBreakdownBars } from '../components/ModelBreakdownBars';
import { EvidenceTimeline } from '../components/EvidenceTimeline';
import { detectionsAPI, agencyAPI } from '../services/api';

export default function ResultsPage() {
    const { alertId } = useParams();
    const location = useLocation();
    const navigate = useNavigate();
    const { toasts, toast, remove } = useToast();
    const [data, setData] = useState(location.state?.result ?? null);
    const [loading, setLoading] = useState(!location.state?.result);
    const [gradcamMode, setGradcamMode] = useState('heatmap');
    const [exporting, setExporting] = useState(false);

    useEffect(() => {
        if (location.state?.result) {
            setData(location.state.result);
            setLoading(false);
            return;
        }
        if (!alertId) return;
        detectionsAPI.getDetection(alertId)
            .then(setData)
            .catch(() => toast('Failed to load result', 'error'))
            .finally(() => setLoading(false));
    }, [alertId, location.state?.result]);

    const handleExport = async () => {
        setExporting(true);
        try {
            const blob = await agencyAPI.getForensicReport(alertId);
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `kavach_forensic_${alertId}.pdf`;
            a.click();
            URL.revokeObjectURL(url);
            toast('Evidence PDF downloaded!', 'success');
        } catch (e) {
            toast(e?.message || 'Export failed', 'error');
        } finally {
            setExporting(false);
        }
    };

    if (loading) return (
        <div className='flex items-center justify-center h-96'>
            <div className='w-8 h-8 rounded-full border-2 animate-spin'
                style={{ borderColor: 'var(--cyan)', borderTopColor: 'transparent' }} />
        </div>
    );

    const modelScores = data?.model_scores ?? {};
    const confidence = data?.confidence ? data.confidence * 100 : 0;
    
    // Construct dynamic timeline events based on the scan data
    const timelineEvents = data ? [
        { timestamp: data.created_at || new Date().toISOString(), action: 'File Received', details: `Payload ID: ${alertId}` },
        { timestamp: data.created_at || new Date().toISOString(), action: 'SHA-256 Signature', hash: data.file_hash || 'Pending' },
        { timestamp: data.created_at || new Date().toISOString(), action: 'Multi-Model Inference', details: `${Object.keys(modelScores).length} neural modules executed` },
        { timestamp: data.created_at || new Date().toISOString(), action: 'Verdict Synthesized', details: `Final confidence string: ${(confidence).toFixed(2)}%` }
    ] : [];

    return (
        <div className='p-6 max-w-5xl mx-auto'>
            <button onClick={() => navigate(-1)} className='mb-4 text-sm hover:underline'
                style={{ color: 'var(--text-muted)' }}>← Back</button>

            {/* ── Verdict banner ── */}
            <div className='rounded-xl p-6 mb-6 flex flex-col sm:flex-row
                       items-center gap-6 justify-between'
                style={{
                    background: (data?.verdict === 'FAKE' || data?.verdict === 'DEEPFAKE') ? '#2c0d10' : data?.verdict === 'SUSPICIOUS' ? '#2c1f0d' : '#0d2c18',
                    border: `1px solid ${(data?.verdict === 'FAKE' || data?.verdict === 'DEEPFAKE') ? '#E6394633' : data?.verdict === 'SUSPICIOUS' ? '#F4A26133' : '#2DC65333'}`,
                }}>
                <VerdictBadge verdict={data?.verdict} size='xl' />
                <ConfidenceRing percentage={typeof confidence === 'number' && confidence <= 1 ? confidence * 100 : (confidence ?? 0)} verdict={data?.verdict} />
                <div className='flex flex-col gap-2 text-sm'
                    style={{ color: 'var(--text-secondary)' }}>
                    <span>Alert ID: <strong className='font-mono'>{alertId}</strong></span>
                    <span>Faces: <strong>{data?.face_count ?? data?.faces_detected ?? 0}</strong></span>
                    <span>Time: <strong>{data?.processing_time_ms}ms</strong></span>
                </div>
            </div>

            {/* ── Uncertainty Warning ── */}
            {data?.is_uncertain && (
                <div className='mb-6 p-4 rounded-xl border flex items-center gap-4 bg-yellow-500/10 border-yellow-500/30'>
                    <span className='text-2xl'>⚠️</span>
                    <div className='flex-1'>
                        <h3 className='text-sm font-bold text-yellow-500 uppercase'>High Model Disagreement</h3>
                        <p className='text-xs text-yellow-200/70'>The AI ensemble results show significant variance (Disagreement: {(data.disagreement * 100).toFixed(1)}%). Use caution when interpreting this verdict.</p>
                    </div>
                </div>
            )}

            <div className='grid grid-cols-1 lg:grid-cols-2 gap-6'>

                {/* ── Model scores ── */}
                <div className='rounded-xl overflow-hidden'
                    style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                    <div className='px-4 py-3' style={{ borderBottom: '1px solid var(--border)' }}>
                        <h2 className='text-sm font-semibold uppercase tracking-wide'
                            style={{ color: 'var(--text-secondary)' }}>Model Scores</h2>
                    </div>
                    <ModelBreakdownBars scores={modelScores} />
                </div>

                {/* ── GradCAM viewer ── */}
                <div className='rounded-xl overflow-hidden'
                    style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                    <div className='px-4 py-3 flex items-center justify-between'
                        style={{ borderBottom: '1px solid var(--border)' }}>
                        <h2 className='text-sm font-semibold uppercase tracking-wide'
                            style={{ color: 'var(--text-secondary)' }}>Explainability (GradCAM)</h2>
                        <div className='flex gap-1'>
                            {['original', 'heatmap'].map(m => (
                                <button key={m} onClick={() => setGradcamMode(m)}
                                    className='text-xs px-2 py-1 rounded transition-colors'
                                    style={{
                                        background: gradcamMode === m ? 'var(--cyan)' : 'var(--bg-primary)',
                                        color: gradcamMode === m ? '#0A1628' : 'var(--text-muted)',
                                    }}>
                                    {m}
                                </button>
                            ))}
                        </div>
                    </div>
                    <div className='p-4 flex items-center justify-center min-h-48'>
                        {data?.heatmap_b64
                            ? <HeatmapViewer base64={data.heatmap_b64} />
                            : <p className='text-sm' style={{ color: 'var(--text-muted)' }}>
                                No heatmap available
                            </p>
                        }
                    </div>
                </div>
                
                {/* ── Chain of Custody Timeline ── */}
                <div className='rounded-xl overflow-hidden lg:col-span-2'
                    style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                    <div className='px-4 py-3 flex items-center justify-between'
                        style={{ borderBottom: '1px solid var(--border)' }}>
                        <h2 className='text-sm font-semibold uppercase tracking-wide text-cyan-500'>
                            Chain of Custody
                        </h2>
                    </div>
                    <div className='p-6'>
                        <EvidenceTimeline events={timelineEvents} />
                    </div>
                </div>

            </div>

            {/* ── AI Agent Findings ── */}
            <AgentSummary 
                findings={data?.findings} 
                summary={data?.public_summary} 
                reportPath={data?.forensic_report} 
            />

            {/* ── Actions ── */}
            <div className='flex gap-3 mt-6'>
                <button onClick={handleExport} disabled={exporting}
                    className='flex-1 py-2.5 rounded-lg text-sm font-semibold
                           transition-all disabled:opacity-50'
                    style={{ background: 'var(--cyan)', color: '#0A1628' }}>
                    {exporting ? 'Exporting...' : '⬇ Download Evidence PDF'}
                </button>
                <button onClick={() => navigate('/scan')}
                    className='px-5 py-2.5 rounded-lg text-sm font-semibold
                           transition-all hover:bg-white/10'
                    style={{ border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
                    + New Scan
                </button>
            </div>

            {toasts.map(t => (
                <Toast key={t.id} message={t.message} type={t.type} onClose={() => remove(t.id)} />
            ))}
        </div>
    );
}
