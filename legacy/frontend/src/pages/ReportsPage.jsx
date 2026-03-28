
import { useState, useEffect } from 'react';
import { FileText, Download, Search, Filter, Loader } from 'lucide-react';
import { detectionsAPI } from '../services/api';

const VERDICT_STYLE = {
    FAKE: 'bg-red-500/10 text-red-400 border-red-500/20',
    SUSPICIOUS: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
    REAL: 'bg-green-500/10 text-green-400 border-green-500/20',
    PENDING: 'bg-gray-500/10 text-gray-400 border-gray-500/20',
};

const ReportsPage = () => {
    const [searchTerm, setSearchTerm] = useState('');
    const [reports, setReports] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchHistory = async () => {
            try {
                const response = await detectionsAPI.getHistory({ limit: 100, offset: 0 });
                // Backend returns array of scan results
                const items = Array.isArray(response) ? response : [];
                const formattedReports = items.map(r => ({
                    id: String(r.id),
                    taskId: r.task_id || String(r.id),
                    date: r.timestamp ? new Date(r.timestamp).toLocaleString() : '—',
                    filename: r.filename || `Scan_${r.id}`,
                    verdict: r.verdict || 'PENDING',
                    confidence: r.confidence ? (r.confidence * 100).toFixed(1) : '0.0',
                    officer: 'System',
                }));
                setReports(formattedReports);
            } catch (err) {
                console.error('Failed to fetch history', err);
            } finally {
                setLoading(false);
            }
        };
        fetchHistory();
    }, []);

    const filteredReports = reports.filter(r =>
        r.filename.toLowerCase().includes(searchTerm.toLowerCase()) ||
        r.id.toLowerCase().includes(searchTerm.toLowerCase())
    );

    const handleDownload = (id) => {
        alert(`Downloading Forensic Report PDF for Scan #${id}...`);
    };

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-end">
                <div>
                    <h1 className="text-3xl font-bold text-white">Forensic Reports</h1>
                    <p className="text-gray-400 mt-1">Archive of all analyzed evidence and chain-of-custody logs.</p>
                </div>
                <div className="flex gap-3">
                    <button className="px-4 py-2 bg-cyber-gray border border-white/10 hover:border-white/30 rounded-lg text-white flex items-center gap-2 text-sm transition-colors">
                        <Filter size={16} /> Filter
                    </button>
                    <button className="px-4 py-2 bg-neon-blue text-black font-bold rounded-lg hover:bg-white transition-colors flex items-center gap-2 text-sm">
                        <Download size={16} /> Export CSV
                    </button>
                </div>
            </div>

            {/* Search Bar */}
            <div className="relative">
                <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400" size={20} />
                <input
                    type="text"
                    placeholder="Search by filename or scan ID..."
                    className="w-full bg-cyber-gray border border-white/10 rounded-xl py-3 pl-12 pr-4 text-white focus:outline-none focus:border-neon-blue transition-colors"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                />
            </div>

            {/* Reports Table */}
            <div className="bg-cyber-gray border border-white/10 rounded-xl overflow-hidden">
                <table className="w-full text-left">
                    <thead className="bg-white/5 border-b border-white/10 text-gray-400 text-sm uppercase tracking-wider">
                        <tr>
                            <th className="p-4 font-medium">Scan ID</th>
                            <th className="p-4 font-medium">Evidence File</th>
                            <th className="p-4 font-medium">Date Analyzed</th>
                            <th className="p-4 font-medium">Verdict</th>
                            <th className="p-4 font-medium">Confidence</th>
                            <th className="p-4 font-medium text-right">Actions</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                        {loading ? (
                            <tr>
                                <td colSpan={6} className="p-12 text-center text-gray-500">
                                    <Loader className="animate-spin mx-auto mb-3" size={28} />
                                    Loading reports...
                                </td>
                            </tr>
                        ) : filteredReports.length === 0 ? (
                            <tr>
                                <td colSpan={6} className="p-12 text-center text-gray-500">
                                    <FileText className="mx-auto mb-3 opacity-30" size={40} />
                                    {searchTerm ? 'No reports match your search.' : 'No scan reports yet. Upload a video or image to analyze.'}
                                </td>
                            </tr>
                        ) : (
                            filteredReports.map((report) => (
                                <tr key={report.id} className="hover:bg-white/5 transition-colors group">
                                    <td className="p-4 font-mono text-neon-blue text-sm">#{report.id}</td>
                                    <td className="p-4 text-white font-medium">
                                        <div className="flex items-center gap-2">
                                            <FileText size={16} className="text-gray-500 shrink-0" />
                                            <span className="truncate max-w-[200px]" title={report.filename}>{report.filename}</span>
                                        </div>
                                    </td>
                                    <td className="p-4 text-gray-400 text-sm">{report.date}</td>
                                    <td className="p-4">
                                        <span className={`px-2 py-1 rounded text-xs font-bold border ${VERDICT_STYLE[report.verdict] || VERDICT_STYLE.PENDING}`}>
                                            {report.verdict}
                                        </span>
                                    </td>
                                    <td className="p-4 text-gray-300 font-mono text-sm">{report.confidence}%</td>
                                    <td className="p-4 text-right">
                                        <button
                                            onClick={() => handleDownload(report.id)}
                                            className="p-2 hover:bg-white/10 rounded-lg text-gray-400 hover:text-white transition-colors opacity-0 group-hover:opacity-100"
                                            title="Download PDF"
                                        >
                                            <Download size={18} />
                                        </button>
                                    </td>
                                </tr>
                            ))
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default ReportsPage;
