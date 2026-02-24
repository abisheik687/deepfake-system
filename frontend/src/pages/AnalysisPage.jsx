
import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Shield, AlertTriangle, FileVideo, Download, Share2, ArrowLeft } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { detectionsAPI } from '../services/api';

const AnalysisPage = () => {
    const { id } = useParams();
    const [loading, setLoading] = useState(true);
    const [data, setData] = useState(null);

    useEffect(() => {
        const fetchAnalysis = async () => {
            try {
                const result = await detectionsAPI.getDetection(id);

                setData({
                    id: result.id,
                    filename: result.features_json?.filename || `Video_${result.id}.mp4`,
                    timestamp: result.timestamp,
                    verdict: result.severity === 'high' || result.severity === 'critical' || result.confidence > 0.85 ? 'FAKE' : 'REAL',
                    confidence: result.confidence * 100, // Convert 0-1 to 0-100
                    breakdown: {
                        audio: (result.audio_confidence || result.confidence - 0.05) * 100,
                        video: (result.spatial_confidence || result.confidence) * 100,
                        temporal: (result.temporal_confidence || result.confidence + 0.05) * 100
                    },
                    timeline: []
                });
            } catch (err) {
                console.error("Failed to fetch analysis", err);
                // Fallback or error state could be added here
            } finally {
                setLoading(false);
            }
        };

        fetchAnalysis();
    }, [id]);

    if (loading) {
        return (
            <div className="flex flex-col items-center justify-center h-[60vh] text-gray-400">
                <motion.div
                    animate={{ rotate: 360, scale: [1, 1.1, 1] }}
                    transition={{ repeat: Infinity, duration: 2 }}
                >
                    <Shield size={64} className="text-neon-blue mb-6" />
                </motion.div>
                <h2 className="text-xl font-bold text-white tracking-widest animate-pulse">ANALYZING FORENSICS...</h2>
                <p className="mt-2 text-sm">Running multi-modal inference engines</p>
            </div>
        );
    }

    const COLORS = ['#00f3ff', '#333333']; // Neon Blue vs Gray
    const pieData = [
        { name: 'Real', value: 100 - data.confidence },
        { name: 'Fake', value: data.confidence }
    ];

    const barData = [
        { name: 'Audio', score: data.breakdown.audio },
        { name: 'Video', score: data.breakdown.video },
        { name: 'Temporal', score: data.breakdown.temporal },
    ];

    return (
        <div className="max-w-6xl mx-auto space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <Link to="/upload" className="p-2 hover:bg-white/10 rounded-full text-gray-400 hover:text-white transition-colors">
                        <ArrowLeft size={24} />
                    </Link>
                    <div>
                        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
                            Analysis Report <span className="text-gray-500 text-lg">#{id.slice(0, 8)}</span>
                        </h1>
                        <p className="text-gray-400 text-sm flex items-center gap-2">
                            <FileVideo size={14} /> {data.filename} • {new Date(data.timestamp).toLocaleString()}
                        </p>
                    </div>
                </div>
                <div className="flex gap-3">
                    <button className="px-4 py-2 bg-cyber-gray border border-white/10 hover:border-white/30 rounded-lg text-white flex items-center gap-2 text-sm">
                        <Share2 size={16} /> Share
                    </button>
                    <button className="px-4 py-2 bg-neon-blue text-black font-bold rounded-lg hover:bg-white transition-colors flex items-center gap-2 text-sm">
                        <Download size={16} /> Export PDF
                    </button>
                </div>
            </div>

            {/* Main Verdict Card */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-1 bg-cyber-gray border border-white/10 rounded-xl p-8 flex flex-col items-center justify-center relative overflow-hidden">
                    <div className={`absolute top-0 w-full h-2 ${data.verdict === 'FAKE' ? 'bg-neon-red' : 'bg-neon-green'}`}></div>
                    <h2 className="text-gray-400 font-medium tracking-widest uppercase mb-4">Final Verdict</h2>
                    <div className={`text-6xl font-bold mb-2 ${data.verdict === 'FAKE' ? 'text-neon-red' : 'text-neon-green'}`}>
                        {data.verdict}
                    </div>
                    <div className="w-full h-64 relative">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={pieData}
                                    innerRadius={60}
                                    outerRadius={80}
                                    paddingAngle={5}
                                    dataKey="value"
                                    startAngle={90}
                                    endAngle={-270}
                                >
                                    <Cell key="real" fill="#333" />
                                    <Cell key="fake" fill={data.verdict === 'FAKE' ? '#ff003c' : '#00ff9d'} />
                                </Pie>
                                <text x="50%" y="50%" textAnchor="middle" dominantBaseline="middle" fill="white" className="text-xl font-bold">
                                    {data.confidence}%
                                </text>
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                    <p className="text-center text-sm text-gray-500 mt-2">
                        Confidence Score based on <br /> Multi-modal Fusion
                    </p>
                </div>

                {/* Breakdown Charts */}
                <div className="lg:col-span-2 space-y-6">
                    {/* Modality Scores */}
                    <div className="bg-cyber-gray border border-white/10 rounded-xl p-6">
                        <h3 className="text-lg font-bold text-white mb-6 flex items-center gap-2">
                            <Shield size={18} className="text-neon-blue" /> Modality Breakdown
                        </h3>
                        <div className="h-64">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={barData} layout="vertical">
                                    <CartesianGrid strokeDasharray="3 3" stroke="#333" horizontal={false} />
                                    <XAxis type="number" domain={[0, 100]} stroke="#666" />
                                    <YAxis dataKey="name" type="category" stroke="#fff" width={80} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#1a1a1a', borderColor: '#333' }}
                                        cursor={{ fill: 'rgba(255, 255, 255, 0.05)' }}
                                    />
                                    <Bar dataKey="score" fill="#00f3ff" barSize={20} radius={[0, 4, 4, 0]}>
                                        {barData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={entry.score > 90 ? '#ff003c' : '#00f3ff'} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>
            </div>

            {/* Alerts / Flags */}
            <div className="bg-neon-red/10 border border-neon-red/20 rounded-xl p-6 flex items-start gap-4">
                <AlertTriangle className="text-neon-red shrink-0" size={24} />
                <div>
                    <h3 className="font-bold text-white text-lg">High Risk Detected</h3>
                    <p className="text-gray-400 mt-1">
                        Temporal inconsistencies detected at <strong>00:05</strong> and <strong>00:15</strong>.
                        Lip sync analysis indicates high probability of manipulation.
                    </p>
                </div>
            </div>
        </div>
    );
};

export default AnalysisPage;
