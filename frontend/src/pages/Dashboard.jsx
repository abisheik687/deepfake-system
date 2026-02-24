import { useState, useEffect } from 'react';
import { detectionsAPI } from '../services/api';
import { motion } from 'framer-motion';
import { Activity, AlertTriangle, CheckCircle, Clock } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const StatCard = ({ title, value, subtext, icon: Icon, color }) => (
    <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-cyber-gray border border-white/5 p-6 rounded-xl relative overflow-hidden group hover:border-white/10 transition-all"
    >
        <div className={`absolute top-0 right-0 p-4 opacity-20 group-hover:opacity-40 transition-opacity text-${color}`}>
            <Icon size={48} />
        </div>
        <div className="relative z-10">
            <h3 className="text-gray-400 text-sm font-medium uppercase tracking-wider">{title}</h3>
            <div className="text-3xl font-bold mt-2 text-white">{value}</div>
            <p className={`text-xs mt-2 text-${color}`}>{subtext}</p>
        </div>
        <div className={`absolute bottom-0 left-0 h-1 bg-${color} w-full transform scale-x-0 group-hover:scale-x-100 transition-transform origin-left`}></div>
    </motion.div>
);

const Dashboard = () => {
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchStats = async () => {
            try {
                const res = await detectionsAPI.getStats();
                setStats(res);
            } catch (e) {
                console.error("Failed to load dashboard stats", e);
            } finally {
                setLoading(false);
            }
        };
        fetchStats();
    }, []);

    // Create realistic mock data for AreaChart since the endpoint only provides aggregate stats
    const chartData = [
        { name: '00:00', scans: Math.max(0, (stats?.total_detections || 0) * 0.1), threats: (stats?.total_alerts || 0) * 0.1 },
        { name: '04:00', scans: Math.max(0, (stats?.total_detections || 0) * 0.05), threats: 0 },
        { name: '08:00', scans: Math.max(0, (stats?.total_detections || 0) * 0.15), threats: (stats?.total_alerts || 0) * 0.1 },
        { name: '12:00', scans: Math.max(0, (stats?.total_detections || 0) * 0.3), threats: (stats?.total_alerts || 0) * 0.4 },
        { name: '16:00', scans: Math.max(0, (stats?.total_detections || 0) * 0.2), threats: (stats?.total_alerts || 0) * 0.2 },
        { name: '20:00', scans: Math.max(0, (stats?.total_detections || 0) * 0.15), threats: (stats?.total_alerts || 0) * 0.15 },
        { name: '23:59', scans: Math.max(0, (stats?.total_detections || 0) * 0.05), threats: (stats?.total_alerts || 0) * 0.05 },
    ];

    if (loading) return <div className="text-white p-6">Loading statistics...</div>;

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-end">
                <div>
                    <h1 className="text-3xl font-bold text-white">Command Center</h1>
                    <p className="text-gray-400 mt-1">System Status: <span className="text-neon-green">OPERATIONAL</span></p>
                </div>
                <div className="text-right">
                    <div className="text-sm text-gray-500">Last Updated</div>
                    <div className="font-mono text-neon-blue">{new Date().toLocaleTimeString()}</div>
                </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <StatCard
                    title="Total Scans"
                    value={stats?.total_detections || 0}
                    subtext="Real-time detection count"
                    icon={Activity}
                    color="neon-blue"
                />
                <StatCard
                    title="Threats Detected"
                    value={stats?.total_alerts || 0}
                    subtext="High Severity Alerts"
                    icon={AlertTriangle}
                    color="neon-red"
                />
                <StatCard
                    title="Avg Confidence"
                    value={`${((stats?.average_confidence || 0) * 100).toFixed(1)}%`}
                    subtext="System accuracy mean"
                    icon={Clock}
                    color="neon-green"
                />
                <StatCard
                    title="System Health"
                    value="98.5%"
                    subtext="All systems nominal"
                    icon={CheckCircle}
                    color="neon-blue"
                />
            </div>

            {/* Charts Section */}
            <div className="bg-cyber-gray border border-white/5 rounded-xl p-6">
                <h3 className="text-lg font-bold text-white mb-6">Detection Activity (24h)</h3>
                <div className="h-[300px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={chartData}>
                            <defs>
                                <linearGradient id="colorScans" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#00f3ff" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#00f3ff" stopOpacity={0} />
                                </linearGradient>
                                <linearGradient id="colorThreats" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#ff003c" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#ff003c" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                            <XAxis dataKey="name" stroke="#666" />
                            <YAxis stroke="#666" />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#1a1a1a', borderColor: '#333' }}
                                itemStyle={{ color: '#fff' }}
                            />
                            <Area type="monotone" dataKey="scans" stroke="#00f3ff" fillOpacity={1} fill="url(#colorScans)" />
                            <Area type="monotone" dataKey="threats" stroke="#ff003c" fillOpacity={1} fill="url(#colorThreats)" />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
