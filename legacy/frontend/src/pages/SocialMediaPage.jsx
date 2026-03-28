import { useState, useEffect } from 'react';
import { Link2, Loader2, CheckCircle, XCircle, Clock, AlertTriangle, Download } from 'lucide-react';
import { motion } from 'framer-motion';
import { socialMediaAPI } from '../services/api';
import { VerdictBadge } from '../components/VerdictBadge';

const PLATFORM_ICONS = {
    youtube: '🎥',
    twitter: '🐦',
    instagram: '📷',
    tiktok: '🎵',
    facebook: '👥',
    unknown: '🔗'
};

const STATUS_CONFIG = {
    pending: { icon: Clock, color: '#F4A261', label: 'Pending' },
    processing: { icon: Loader2, color: '#3B82F6', label: 'Processing', spin: true },
    completed: { icon: CheckCircle, color: '#2DC653', label: 'Completed' },
    failed: { icon: XCircle, color: '#E63946', label: 'Failed' }
};

const RISK_LEVELS = {
    LOW: { color: '#2DC653', bg: '#0d2c18', label: 'LOW RISK' },
    MEDIUM: { color: '#F4A261', bg: '#2c1f0d', label: 'MEDIUM RISK' },
    HIGH: { color: '#E63946', bg: '#2c0d10', label: 'HIGH RISK' },
    CRITICAL: { color: '#E63946', bg: '#2c0d10', label: 'CRITICAL', pulse: true }
};

function StatusBadge({ status }) {
    const config = STATUS_CONFIG[status] || STATUS_CONFIG.pending;
    const Icon = config.icon;
    
    return (
        <div className='flex items-center gap-1.5'>
            <Icon 
                size={14} 
                className={config.spin ? 'animate-spin' : ''} 
                style={{ color: config.color }} 
            />
            <span className='text-xs font-semibold uppercase tracking-wider' style={{ color: config.color }}>
                {config.label}
            </span>
        </div>
    );
}

function RiskBadge({ level }) {
    const config = RISK_LEVELS[level] || RISK_LEVELS.LOW;
    
    return (
        <div 
            className='inline-flex items-center gap-1.5 px-2 py-1 rounded-md'
            style={{ background: config.bg, border: `1px solid ${config.color}33` }}
        >
            <div 
                className={`w-1.5 h-1.5 rounded-full ${config.pulse ? 'animate-pulse' : ''}`}
                style={{ background: config.color }}
            />
            <span className='text-xs font-bold tracking-wider' style={{ color: config.color }}>
                {config.label}
            </span>
        </div>
    );
}

export default function SocialMediaPage() {
    const [url, setUrl] = useState('');
    const [priority, setPriority] = useState('normal');
    const [queue, setQueue] = useState([]);
    const [loading, setLoading] = useState(false);
    const [platforms, setPlatforms] = useState([]);
    const [expandedId, setExpandedId] = useState(null);

    useEffect(() => {
        loadQueue();
        loadPlatforms();
        const interval = setInterval(loadQueue, 5000); // Poll every 5 seconds
        return () => clearInterval(interval);
    }, []);

    const loadQueue = async () => {
        try {
            const data = await socialMediaAPI.getQueue();
            setQueue(Array.isArray(data) ? data : data.scans || []);
        } catch (error) {
            console.error('Failed to load queue:', error);
        }
    };

    const loadPlatforms = async () => {
        try {
            const data = await socialMediaAPI.getPlatforms();
            setPlatforms(data.platforms || []);
        } catch (error) {
            console.error('Failed to load platforms:', error);
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!url.trim()) return;

        setLoading(true);
        try {
            await socialMediaAPI.scanURL(url, priority);
            setUrl('');
            await loadQueue();
        } catch (error) {
            alert(error.message || 'Failed to submit URL for scanning');
        } finally {
            setLoading(false);
        }
    };

    const toggleExpand = (id) => {
        setExpandedId(expandedId === id ? null : id);
    };

    const getRiskLevel = (confidence) => {
        if (confidence >= 0.9) return 'CRITICAL';
        if (confidence >= 0.7) return 'HIGH';
        if (confidence >= 0.5) return 'MEDIUM';
        return 'LOW';
    };

    return (
        <div className='p-6 max-w-7xl mx-auto'>
            {/* Header */}
            <div className='mb-6'>
                <h1 className='text-2xl font-bold mb-2' style={{ color: 'var(--text-primary)' }}>
                    Social Media Analysis
                </h1>
                <p className='text-sm' style={{ color: 'var(--text-muted)' }}>
                    Scan public social media URLs for deepfake content
                </p>
            </div>

            {/* Supported Platforms */}
            {platforms.length > 0 && (
                <div className='mb-6 p-4 rounded-xl' style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                    <h3 className='text-xs font-semibold uppercase tracking-wider mb-3' style={{ color: 'var(--text-secondary)' }}>
                        Supported Platforms
                    </h3>
                    <div className='flex flex-wrap gap-2'>
                        {platforms.map(platform => (
                            <div 
                                key={platform}
                                className='flex items-center gap-2 px-3 py-1.5 rounded-lg'
                                style={{ background: 'var(--bg-hover)', border: '1px solid var(--border)' }}
                            >
                                <span className='text-lg'>{PLATFORM_ICONS[platform.toLowerCase()] || PLATFORM_ICONS.unknown}</span>
                                <span className='text-xs font-semibold capitalize' style={{ color: 'var(--text-primary)' }}>
                                    {platform}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* URL Input Form */}
            <form onSubmit={handleSubmit} className='mb-6'>
                <div className='rounded-xl p-6' style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                    <div className='flex flex-col md:flex-row gap-4'>
                        <div className='flex-1'>
                            <label className='block text-xs font-semibold uppercase tracking-wider mb-2' style={{ color: 'var(--text-secondary)' }}>
                                Social Media URL
                            </label>
                            <div className='relative'>
                                <Link2 className='absolute left-3 top-1/2 -translate-y-1/2' size={18} style={{ color: 'var(--text-muted)' }} />
                                <input
                                    type='url'
                                    value={url}
                                    onChange={(e) => setUrl(e.target.value)}
                                    placeholder='https://youtube.com/watch?v=... or https://twitter.com/...'
                                    className='w-full pl-10 pr-4 py-3 rounded-lg text-sm font-mono'
                                    style={{
                                        background: 'var(--bg-hover)',
                                        border: '1px solid var(--border)',
                                        color: 'var(--text-primary)',
                                        outline: 'none'
                                    }}
                                    required
                                />
                            </div>
                        </div>

                        <div className='w-full md:w-48'>
                            <label className='block text-xs font-semibold uppercase tracking-wider mb-2' style={{ color: 'var(--text-secondary)' }}>
                                Priority
                            </label>
                            <select
                                value={priority}
                                onChange={(e) => setPriority(e.target.value)}
                                className='w-full px-4 py-3 rounded-lg text-sm font-semibold'
                                style={{
                                    background: 'var(--bg-hover)',
                                    border: '1px solid var(--border)',
                                    color: 'var(--text-primary)',
                                    outline: 'none'
                                }}
                            >
                                <option value='low'>Low</option>
                                <option value='normal'>Normal</option>
                                <option value='high'>High</option>
                                <option value='urgent'>Urgent</option>
                            </select>
                        </div>

                        <div className='flex items-end'>
                            <button
                                type='submit'
                                disabled={loading || !url.trim()}
                                className='w-full md:w-auto px-6 py-3 rounded-lg font-bold text-sm transition-all disabled:opacity-50 flex items-center justify-center gap-2'
                                style={{ background: 'var(--cyan)', color: '#0A1628' }}
                            >
                                {loading ? (
                                    <>
                                        <Loader2 size={18} className='animate-spin' />
                                        Submitting...
                                    </>
                                ) : (
                                    <>
                                        <Link2 size={18} />
                                        Scan URL
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                </div>
            </form>

            {/* Queue Table */}
            <div className='rounded-xl overflow-hidden' style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                <div className='px-6 py-4 flex items-center justify-between' style={{ borderBottom: '1px solid var(--border)' }}>
                    <h2 className='font-semibold text-sm uppercase tracking-wider' style={{ color: 'var(--text-secondary)' }}>
                        Scan Queue ({queue.length})
                    </h2>
                    <button
                        onClick={loadQueue}
                        className='text-xs px-3 py-1.5 rounded-md font-semibold transition-all'
                        style={{ background: 'var(--bg-hover)', color: 'var(--text-primary)' }}
                    >
                        Refresh
                    </button>
                </div>

                <div className='overflow-x-auto'>
                    {queue.length === 0 ? (
                        <div className='p-12 text-center'>
                            <Link2 size={48} className='mx-auto mb-4 opacity-20' style={{ color: 'var(--text-muted)' }} />
                            <p className='text-sm font-semibold mb-1' style={{ color: 'var(--text-secondary)' }}>
                                No scans in queue
                            </p>
                            <p className='text-xs' style={{ color: 'var(--text-muted)' }}>
                                Submit a social media URL to start scanning
                            </p>
                        </div>
                    ) : (
                        <table className='w-full'>
                            <thead>
                                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                                    <th className='px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider' style={{ color: 'var(--text-secondary)' }}>
                                        Platform
                                    </th>
                                    <th className='px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider' style={{ color: 'var(--text-secondary)' }}>
                                        URL
                                    </th>
                                    <th className='px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider' style={{ color: 'var(--text-secondary)' }}>
                                        Status
                                    </th>
                                    <th className='px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider' style={{ color: 'var(--text-secondary)' }}>
                                        Result
                                    </th>
                                    <th className='px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider' style={{ color: 'var(--text-secondary)' }}>
                                        Risk
                                    </th>
                                    <th className='px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider' style={{ color: 'var(--text-secondary)' }}>
                                        Actions
                                    </th>
                                </tr>
                            </thead>
                            <tbody>
                                {queue.map((scan, idx) => (
                                    <motion.tr
                                        key={scan.id || idx}
                                        initial={{ opacity: 0, y: 10 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ delay: idx * 0.05 }}
                                        className='cursor-pointer transition-colors'
                                        style={{ borderBottom: '1px solid var(--border)' }}
                                        onClick={() => toggleExpand(scan.id)}
                                    >
                                        <td className='px-6 py-4'>
                                            <div className='flex items-center gap-2'>
                                                <span className='text-2xl'>{PLATFORM_ICONS[scan.platform?.toLowerCase()] || PLATFORM_ICONS.unknown}</span>
                                                <span className='text-sm font-semibold capitalize' style={{ color: 'var(--text-primary)' }}>
                                                    {scan.platform || 'Unknown'}
                                                </span>
                                            </div>
                                        </td>
                                        <td className='px-6 py-4'>
                                            <div className='max-w-xs truncate text-xs font-mono' style={{ color: 'var(--text-muted)' }}>
                                                {scan.url}
                                            </div>
                                        </td>
                                        <td className='px-6 py-4'>
                                            <StatusBadge status={scan.status} />
                                        </td>
                                        <td className='px-6 py-4'>
                                            {scan.status === 'completed' && scan.result ? (
                                                <VerdictBadge verdict={scan.result.verdict} />
                                            ) : (
                                                <span className='text-xs' style={{ color: 'var(--text-muted)' }}>—</span>
                                            )}
                                        </td>
                                        <td className='px-6 py-4'>
                                            {scan.status === 'completed' && scan.result ? (
                                                <RiskBadge level={getRiskLevel(scan.result.confidence || 0)} />
                                            ) : (
                                                <span className='text-xs' style={{ color: 'var(--text-muted)' }}>—</span>
                                            )}
                                        </td>
                                        <td className='px-6 py-4'>
                                            {scan.status === 'completed' && scan.result && (
                                                <button
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        window.open(scan.result.report_url, '_blank');
                                                    }}
                                                    className='flex items-center gap-1 px-3 py-1.5 rounded-md text-xs font-semibold transition-all'
                                                    style={{ background: 'var(--bg-hover)', color: 'var(--text-primary)' }}
                                                >
                                                    <Download size={14} />
                                                    Report
                                                </button>
                                            )}
                                        </td>
                                    </motion.tr>
                                ))}
                            </tbody>
                        </table>
                    )}
                </div>

                {/* Expanded Details */}
                {expandedId && queue.find(s => s.id === expandedId) && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className='px-6 py-4'
                        style={{ background: 'var(--bg-hover)', borderTop: '1px solid var(--border)' }}
                    >
                        {(() => {
                            const scan = queue.find(s => s.id === expandedId);
                            return (
                                <div className='space-y-3'>
                                    <div>
                                        <h4 className='text-xs font-semibold uppercase tracking-wider mb-2' style={{ color: 'var(--text-secondary)' }}>
                                            Full URL
                                        </h4>
                                        <p className='text-sm font-mono break-all' style={{ color: 'var(--text-primary)' }}>
                                            {scan.url}
                                        </p>
                                    </div>

                                    {scan.result && (
                                        <>
                                            <div>
                                                <h4 className='text-xs font-semibold uppercase tracking-wider mb-2' style={{ color: 'var(--text-secondary)' }}>
                                                    Content Type
                                                </h4>
                                                <p className='text-sm capitalize' style={{ color: 'var(--text-primary)' }}>
                                                    {scan.result.content_type || 'Unknown'}
                                                </p>
                                            </div>

                                            <div>
                                                <h4 className='text-xs font-semibold uppercase tracking-wider mb-2' style={{ color: 'var(--text-secondary)' }}>
                                                    Confidence Score
                                                </h4>
                                                <div className='flex items-center gap-3'>
                                                    <div className='flex-1 h-2 rounded-full overflow-hidden' style={{ background: 'var(--bg-card)' }}>
                                                        <div
                                                            className='h-full transition-all'
                                                            style={{
                                                                width: `${(scan.result.confidence || 0) * 100}%`,
                                                                background: scan.result.confidence > 0.5 ? '#E63946' : '#2DC653'
                                                            }}
                                                        />
                                                    </div>
                                                    <span className='text-sm font-bold' style={{ color: 'var(--text-primary)' }}>
                                                        {Math.round((scan.result.confidence || 0) * 100)}%
                                                    </span>
                                                </div>
                                            </div>

                                            {scan.result.analysis && (
                                                <div>
                                                    <h4 className='text-xs font-semibold uppercase tracking-wider mb-2' style={{ color: 'var(--text-secondary)' }}>
                                                        Analysis
                                                    </h4>
                                                    <p className='text-sm' style={{ color: 'var(--text-primary)' }}>
                                                        {scan.result.analysis}
                                                    </p>
                                                </div>
                                            )}
                                        </>
                                    )}

                                    {scan.error && (
                                        <div className='flex items-start gap-2 p-3 rounded-lg' style={{ background: '#2c0d10', border: '1px solid #E6394633' }}>
                                            <AlertTriangle size={16} className='shrink-0 mt-0.5' style={{ color: '#E63946' }} />
                                            <div>
                                                <h4 className='text-xs font-semibold mb-1' style={{ color: '#E63946' }}>
                                                    Error
                                                </h4>
                                                <p className='text-xs' style={{ color: 'var(--text-muted)' }}>
                                                    {scan.error}
                                                </p>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            );
                        })()}
                    </motion.div>
                )}
            </div>
        </div>
    );
}

// Made with Bob
