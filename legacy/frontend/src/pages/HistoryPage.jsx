import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { History, Filter, Download, RefreshCw, Search, Calendar, FileText } from 'lucide-react';
import { motion } from 'framer-motion';
import { detectionsAPI } from '../services/api';
import { VerdictBadge } from '../components/VerdictBadge';

const DETECTION_TYPES = ['All', 'Image', 'Video', 'Audio', 'Social Media', 'Live Video', 'Live Audio', 'Interview'];
const VERDICTS = ['All', 'REAL', 'FAKE', 'UNCERTAIN'];
const ITEMS_PER_PAGE = 20;

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric', 
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function ConfidenceBar({ confidence }) {
    const percentage = Math.round(confidence * 100);
    const color = confidence > 0.5 ? '#E63946' : '#2DC653';
    
    return (
        <div className='flex items-center gap-2'>
            <div className='flex-1 h-1.5 rounded-full overflow-hidden' style={{ background: 'var(--bg-hover)' }}>
                <div
                    className='h-full transition-all'
                    style={{ width: `${percentage}%`, background: color }}
                />
            </div>
            <span className='text-xs font-bold w-10 text-right' style={{ color }}>
                {percentage}%
            </span>
        </div>
    );
}

export default function HistoryPage() {
    const navigate = useNavigate();
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);
    const [page, setPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    
    // Filters
    const [searchQuery, setSearchQuery] = useState('');
    const [typeFilter, setTypeFilter] = useState('All');
    const [verdictFilter, setVerdictFilter] = useState('All');
    const [dateFrom, setDateFrom] = useState('');
    const [dateTo, setDateTo] = useState('');

    useEffect(() => {
        loadHistory();
    }, [page, typeFilter, verdictFilter, dateFrom, dateTo]);

    const loadHistory = async () => {
        setLoading(true);
        try {
            const params = {
                limit: ITEMS_PER_PAGE,
                offset: (page - 1) * ITEMS_PER_PAGE
            };

            // Apply filters
            if (typeFilter !== 'All') {
                params.detection_type = typeFilter.toLowerCase().replace(' ', '_');
            }
            if (verdictFilter !== 'All') {
                params.verdict = verdictFilter;
            }
            if (dateFrom) {
                params.date_from = dateFrom;
            }
            if (dateTo) {
                params.date_to = dateTo;
            }

            const data = await detectionsAPI.getHistory(params);
            const items = Array.isArray(data) ? data : data.items || [];
            setHistory(items);
            
            // Calculate total pages
            const total = data.total || items.length;
            setTotalPages(Math.ceil(total / ITEMS_PER_PAGE));
        } catch (error) {
            console.error('Failed to load history:', error);
            setHistory([]);
        } finally {
            setLoading(false);
        }
    };

    const handleRefresh = () => {
        loadHistory();
    };

    const handleReset = () => {
        setSearchQuery('');
        setTypeFilter('All');
        setVerdictFilter('All');
        setDateFrom('');
        setDateTo('');
        setPage(1);
    };

    const handleViewDetails = (detection) => {
        navigate(`/results?id=${detection.id}`);
    };

    const handleExportCSV = () => {
        const csv = [
            ['ID', 'Type', 'Verdict', 'Confidence', 'Date', 'Filename'].join(','),
            ...history.map(d => [
                d.id,
                d.detection_type || 'Unknown',
                d.verdict || 'N/A',
                d.confidence || 0,
                d.created_at || '',
                d.filename || 'N/A'
            ].join(','))
        ].join('\n');

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `kavach-history-${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
        URL.revokeObjectURL(url);
    };

    // Filter history by search query (client-side)
    const filteredHistory = history.filter(item => {
        if (!searchQuery) return true;
        const query = searchQuery.toLowerCase();
        return (
            item.filename?.toLowerCase().includes(query) ||
            item.detection_type?.toLowerCase().includes(query) ||
            item.verdict?.toLowerCase().includes(query) ||
            item.id?.toString().includes(query)
        );
    });

    return (
        <div className='p-6 max-w-7xl mx-auto'>
            {/* Header */}
            <div className='flex items-center justify-between mb-6'>
                <div>
                    <h1 className='text-2xl font-bold mb-1' style={{ color: 'var(--text-primary)' }}>
                        Detection History
                    </h1>
                    <p className='text-sm' style={{ color: 'var(--text-muted)' }}>
                        View and manage all past detections
                    </p>
                </div>
                <div className='flex items-center gap-2'>
                    <button
                        onClick={handleRefresh}
                        className='flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all'
                        style={{ background: 'var(--bg-card)', color: 'var(--text-primary)', border: '1px solid var(--border)' }}
                    >
                        <RefreshCw size={16} />
                        Refresh
                    </button>
                    <button
                        onClick={handleExportCSV}
                        disabled={history.length === 0}
                        className='flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all disabled:opacity-50'
                        style={{ background: 'var(--cyan)', color: '#0A1628' }}
                    >
                        <Download size={16} />
                        Export CSV
                    </button>
                </div>
            </div>

            {/* Filters */}
            <div className='mb-6 rounded-xl p-6' style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                <div className='flex items-center gap-2 mb-4'>
                    <Filter size={18} style={{ color: 'var(--cyan)' }} />
                    <h3 className='font-semibold text-sm uppercase tracking-wider' style={{ color: 'var(--text-secondary)' }}>
                        Filters
                    </h3>
                </div>

                <div className='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4'>
                    {/* Search */}
                    <div className='lg:col-span-2'>
                        <label className='block text-xs font-semibold uppercase tracking-wider mb-2' style={{ color: 'var(--text-secondary)' }}>
                            Search
                        </label>
                        <div className='relative'>
                            <Search className='absolute left-3 top-1/2 -translate-y-1/2' size={16} style={{ color: 'var(--text-muted)' }} />
                            <input
                                type='text'
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                placeholder='Search by filename, ID, or type...'
                                className='w-full pl-10 pr-4 py-2 rounded-lg text-sm'
                                style={{
                                    background: 'var(--bg-hover)',
                                    border: '1px solid var(--border)',
                                    color: 'var(--text-primary)',
                                    outline: 'none'
                                }}
                            />
                        </div>
                    </div>

                    {/* Type Filter */}
                    <div>
                        <label className='block text-xs font-semibold uppercase tracking-wider mb-2' style={{ color: 'var(--text-secondary)' }}>
                            Type
                        </label>
                        <select
                            value={typeFilter}
                            onChange={(e) => { setTypeFilter(e.target.value); setPage(1); }}
                            className='w-full px-3 py-2 rounded-lg text-sm'
                            style={{
                                background: 'var(--bg-hover)',
                                border: '1px solid var(--border)',
                                color: 'var(--text-primary)',
                                outline: 'none'
                            }}
                        >
                            {DETECTION_TYPES.map(type => (
                                <option key={type} value={type}>{type}</option>
                            ))}
                        </select>
                    </div>

                    {/* Verdict Filter */}
                    <div>
                        <label className='block text-xs font-semibold uppercase tracking-wider mb-2' style={{ color: 'var(--text-secondary)' }}>
                            Verdict
                        </label>
                        <select
                            value={verdictFilter}
                            onChange={(e) => { setVerdictFilter(e.target.value); setPage(1); }}
                            className='w-full px-3 py-2 rounded-lg text-sm'
                            style={{
                                background: 'var(--bg-hover)',
                                border: '1px solid var(--border)',
                                color: 'var(--text-primary)',
                                outline: 'none'
                            }}
                        >
                            {VERDICTS.map(verdict => (
                                <option key={verdict} value={verdict}>{verdict}</option>
                            ))}
                        </select>
                    </div>

                    {/* Reset Button */}
                    <div className='flex items-end'>
                        <button
                            onClick={handleReset}
                            className='w-full px-4 py-2 rounded-lg text-sm font-semibold transition-all'
                            style={{ background: 'var(--bg-hover)', color: 'var(--text-primary)', border: '1px solid var(--border)' }}
                        >
                            Reset Filters
                        </button>
                    </div>
                </div>

                {/* Date Range */}
                <div className='grid grid-cols-1 md:grid-cols-2 gap-4 mt-4'>
                    <div>
                        <label className='block text-xs font-semibold uppercase tracking-wider mb-2' style={{ color: 'var(--text-secondary)' }}>
                            Date From
                        </label>
                        <div className='relative'>
                            <Calendar className='absolute left-3 top-1/2 -translate-y-1/2' size={16} style={{ color: 'var(--text-muted)' }} />
                            <input
                                type='date'
                                value={dateFrom}
                                onChange={(e) => { setDateFrom(e.target.value); setPage(1); }}
                                className='w-full pl-10 pr-4 py-2 rounded-lg text-sm'
                                style={{
                                    background: 'var(--bg-hover)',
                                    border: '1px solid var(--border)',
                                    color: 'var(--text-primary)',
                                    outline: 'none'
                                }}
                            />
                        </div>
                    </div>
                    <div>
                        <label className='block text-xs font-semibold uppercase tracking-wider mb-2' style={{ color: 'var(--text-secondary)' }}>
                            Date To
                        </label>
                        <div className='relative'>
                            <Calendar className='absolute left-3 top-1/2 -translate-y-1/2' size={16} style={{ color: 'var(--text-muted)' }} />
                            <input
                                type='date'
                                value={dateTo}
                                onChange={(e) => { setDateTo(e.target.value); setPage(1); }}
                                className='w-full pl-10 pr-4 py-2 rounded-lg text-sm'
                                style={{
                                    background: 'var(--bg-hover)',
                                    border: '1px solid var(--border)',
                                    color: 'var(--text-primary)',
                                    outline: 'none'
                                }}
                            />
                        </div>
                    </div>
                </div>
            </div>

            {/* History Table */}
            <div className='rounded-xl overflow-hidden' style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                <div className='px-6 py-4 flex items-center justify-between' style={{ borderBottom: '1px solid var(--border)' }}>
                    <h2 className='font-semibold text-sm uppercase tracking-wider' style={{ color: 'var(--text-secondary)' }}>
                        Results ({filteredHistory.length})
                    </h2>
                </div>

                {loading ? (
                    <div className='p-12 text-center'>
                        <RefreshCw size={48} className='mx-auto mb-4 animate-spin opacity-20' style={{ color: 'var(--text-muted)' }} />
                        <p className='text-sm' style={{ color: 'var(--text-muted)' }}>Loading history...</p>
                    </div>
                ) : filteredHistory.length === 0 ? (
                    <div className='p-12 text-center'>
                        <History size={48} className='mx-auto mb-4 opacity-20' style={{ color: 'var(--text-muted)' }} />
                        <p className='text-sm font-semibold mb-1' style={{ color: 'var(--text-secondary)' }}>
                            No detections found
                        </p>
                        <p className='text-xs' style={{ color: 'var(--text-muted)' }}>
                            Try adjusting your filters or run a new scan
                        </p>
                    </div>
                ) : (
                    <>
                        <div className='overflow-x-auto'>
                            <table className='w-full'>
                                <thead>
                                    <tr style={{ borderBottom: '1px solid var(--border)' }}>
                                        <th className='px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider' style={{ color: 'var(--text-secondary)' }}>
                                            ID
                                        </th>
                                        <th className='px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider' style={{ color: 'var(--text-secondary)' }}>
                                            Type
                                        </th>
                                        <th className='px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider' style={{ color: 'var(--text-secondary)' }}>
                                            Filename
                                        </th>
                                        <th className='px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider' style={{ color: 'var(--text-secondary)' }}>
                                            Verdict
                                        </th>
                                        <th className='px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider' style={{ color: 'var(--text-secondary)' }}>
                                            Confidence
                                        </th>
                                        <th className='px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider' style={{ color: 'var(--text-secondary)' }}>
                                            Date
                                        </th>
                                        <th className='px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider' style={{ color: 'var(--text-secondary)' }}>
                                            Actions
                                        </th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {filteredHistory.map((detection, idx) => (
                                        <motion.tr
                                            key={detection.id}
                                            initial={{ opacity: 0, y: 10 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ delay: idx * 0.03 }}
                                            className='cursor-pointer transition-colors hover:bg-opacity-50'
                                            style={{ borderBottom: '1px solid var(--border)' }}
                                            onClick={() => handleViewDetails(detection)}
                                        >
                                            <td className='px-6 py-4'>
                                                <span className='text-xs font-mono' style={{ color: 'var(--text-muted)' }}>
                                                    #{detection.id}
                                                </span>
                                            </td>
                                            <td className='px-6 py-4'>
                                                <span className='text-sm font-semibold capitalize' style={{ color: 'var(--text-primary)' }}>
                                                    {detection.detection_type?.replace('_', ' ') || 'Unknown'}
                                                </span>
                                            </td>
                                            <td className='px-6 py-4'>
                                                <div className='max-w-xs truncate text-sm' style={{ color: 'var(--text-primary)' }}>
                                                    {detection.filename || 'N/A'}
                                                </div>
                                            </td>
                                            <td className='px-6 py-4'>
                                                <VerdictBadge verdict={detection.verdict} />
                                            </td>
                                            <td className='px-6 py-4'>
                                                <div className='w-32'>
                                                    <ConfidenceBar confidence={detection.confidence || 0} />
                                                </div>
                                            </td>
                                            <td className='px-6 py-4'>
                                                <span className='text-xs' style={{ color: 'var(--text-muted)' }}>
                                                    {formatDate(detection.created_at)}
                                                </span>
                                            </td>
                                            <td className='px-6 py-4'>
                                                <button
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        handleViewDetails(detection);
                                                    }}
                                                    className='flex items-center gap-1 px-3 py-1.5 rounded-md text-xs font-semibold transition-all'
                                                    style={{ background: 'var(--bg-hover)', color: 'var(--text-primary)' }}
                                                >
                                                    <FileText size={14} />
                                                    View
                                                </button>
                                            </td>
                                        </motion.tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>

                        {/* Pagination */}
                        {totalPages > 1 && (
                            <div className='px-6 py-4 flex items-center justify-between' style={{ borderTop: '1px solid var(--border)' }}>
                                <button
                                    onClick={() => setPage(p => Math.max(1, p - 1))}
                                    disabled={page === 1}
                                    className='px-4 py-2 rounded-lg text-sm font-semibold transition-all disabled:opacity-50'
                                    style={{ background: 'var(--bg-hover)', color: 'var(--text-primary)' }}
                                >
                                    Previous
                                </button>
                                <span className='text-sm' style={{ color: 'var(--text-muted)' }}>
                                    Page {page} of {totalPages}
                                </span>
                                <button
                                    onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                                    disabled={page === totalPages}
                                    className='px-4 py-2 rounded-lg text-sm font-semibold transition-all disabled:opacity-50'
                                    style={{ background: 'var(--bg-hover)', color: 'var(--text-primary)' }}
                                >
                                    Next
                                </button>
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
}

// Made with Bob
