
import { useState, useEffect } from 'react';
import { FileText, Download, Search, Filter, Eye, ChevronRight, Loader } from 'lucide-react';
import { Link } from 'react-router-dom';
import client from '../api/client';

const ReportsPage = () => {
    const [searchTerm, setSearchTerm] = useState('');
    const [reports, setReports] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchHistory = async () => {
            try {
                const response = await client.get('/scan/history');
                const formattedReports = response.data.map(r => ({
                    id: r.task_id,
                    date: new Date(r.created_at).toLocaleString(),
                    filename: r.filename,
                    verdict: r.verdict || 'PENDING',
                    confidence: r.final_score ? (r.final_score * 100).toFixed(1) : 0,
                    officer: 'You' // Since endpoint filters by current user
                }));
                setReports(formattedReports);
            } catch (err) {
                console.error("Failed to fetch history", err);
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
        alert(`Downloading Forensic Report PDF for ${id}...`);
    };

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-end">
                <div>
                    <h1 className="text-3xl font-bold text-white">Forensic Reports</h1>
                    <p className="text-gray-400 mt-1">Archive of all analyzed evidence and generated chain-of-custody logs.</p>
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
                    placeholder="Search by filename or ID..."
                    className="w-full bg-cyber-gray border border-white/10 rounded-xl py-3 pl-12 pr-4 text-white focus:outline-none focus:border-neon-blue transition-colors"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                />
            </div>

            {/* Reports List */}
            <div className="bg-cyber-gray border border-white/10 rounded-xl overflow-hidden">
                <table className="w-full text-left">
                    <thead className="bg-white/5 border-b border-white/10 text-gray-400 text-sm uppercase tracking-wider">
                        <tr>
                            <th className="p-4 font-medium">Scan ID</th>
                            <th className="p-4 font-medium">Evidence File</th>
                            <th className="p-4 font-medium">Date Analyzed</th>
                            <th className="p-4 font-medium">Verdict</th>
                            <th className="p-4 font-medium">Officer</th>
                            <th className="p-4 font-medium text-right">Actions</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                        {filteredReports.map((report) => (
                            <tr key={report.id} className="hover:bg-white/5 transition-colors group">
                                <td className="p-4 font-mono text-neon-blue text-sm">{report.id}</td>
                                <td className="p-4 text-white font-medium flex items-center gap-2">
                                    <FileText size={16} className="text-gray-500" />
                                    {report.filename}
                                </td>
                                <td className="p-4 text-gray-400 text-sm">{report.date}</td>
                                <td className="p-4">
                                    <span className={`px-2 py-1 rounded text-xs font-bold ${report.verdict === 'FAKE'
                                        ? 'bg-red-500/10 text-red-500 border border-red-500/20'
                                        : 'bg-green-500/10 text-green-500 border border-green-500/20'
                                        }`}>
                                        {report.verdict} ({report.confidence}%)
                                    </span>
                                </td>
                                <td className="p-4 text-gray-400 text-sm">{report.officer}</td>
                                <td className="p-4 text-right flex items-center justify-end gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                    <button
                                        onClick={() => handleDownload(report.id)}
                                        className="p-2 hover:bg-white/10 rounded-lg text-gray-400 hover:text-white transition-colors"
                                        title="Download PDF"
                                    >
                                        <Download size={18} />
                                    </button>
                                    <Link
                                        to={`/analysis/${report.id}`}
                                        className="p-2 hover:bg-white/10 rounded-lg text-neon-blue hover:bg-neon-blue/10 transition-colors"
                                        title="View Analysis"
                                    >
                                        <ChevronRight size={18} />
                                    </Link>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
                {filteredReports.length === 0 && (
                    <div className="p-8 text-center text-gray-500">
                        No reports found matching your search.
                    </div>
                )}
            </div>
        </div>
    );
};

export default ReportsPage;
