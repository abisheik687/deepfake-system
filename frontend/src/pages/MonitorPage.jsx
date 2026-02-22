
import { useState, useEffect } from 'react';
import { Radio, AlertTriangle, Activity, Eye, Search } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const MonitorPage = () => {
    const [logs, setLogs] = useState([
        { id: 1, time: '10:42:01', source: 'Cam-04', type: 'INFO', message: 'Face detected: Confidence 98%' },
        { id: 2, time: '10:42:05', source: 'Cam-04', type: 'WARNING', message: 'Lip-sync anomaly detected' },
        { id: 3, time: '10:42:12', source: 'Cam-04', type: 'ALERT', message: 'DEEPFAKE DETECTED: Score 94%' },
    ]);

    // Simulate incoming logs
    useEffect(() => {
        const interval = setInterval(() => {
            const newLog = {
                id: Date.now(),
                time: new Date().toLocaleTimeString(),
                source: 'Cam-04',
                type: Math.random() > 0.7 ? (Math.random() > 0.8 ? 'ALERT' : 'WARNING') : 'INFO',
                message: Math.random() > 0.7 ? 'Temporal inconsistency found' : 'Scanning active stream...'
            };
            setLogs(prev => [newLog, ...prev].slice(0, 50));
        }, 3000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="h-[calc(100vh-100px)] grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left Column: Video Feed */}
            <div className="lg:col-span-2 flex flex-col gap-6">
                <div className="bg-cyber-gray border border-white/10 rounded-xl overflow-hidden relative group">
                    {/* Header Overlay */}
                    <div className="absolute top-0 left-0 w-full p-4 bg-gradient-to-b from-black/80 to-transparent z-10 flex justify-between items-start">
                        <div className="flex items-center gap-2 text-neon-red animate-pulse">
                            <Radio size={16} />
                            <span className="font-bold text-sm tracking-widest uppercase">Live Feed • REC</span>
                        </div>
                        <div className="bg-black/50 backdrop-blur px-3 py-1 rounded text-xs text-white border border-white/10">
                            CAM-04: MAIN_HALL_ENTRANCE
                        </div>
                    </div>

                    {/* Placeholder for Video Player */}
                    <div className="aspect-video bg-black relative flex items-center justify-center">
                        <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&w=1600&q=80')] bg-cover bg-center opacity-40"></div>
                        <div className="absolute inset-0 bg-[linear-gradient(rgba(0,0,0,0)_0%,rgba(0,243,255,0.05)_50%,rgba(0,0,0,0)_100%)] bg-[length:100%_4px] animate-scan"></div>

                        {/* Detection Bounding Box Mockup */}
                        <motion.div
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ duration: 0.5, repeat: Infinity, repeatType: "reverse", repeatDelay: 2 }}
                            className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 border-2 border-neon-red rounded-lg"
                        >
                            <div className="absolute -top-6 left-0 bg-neon-red text-black text-xs font-bold px-2 py-1">
                                FAKE DETECTED (94%)
                            </div>
                            {/* Corner Accents */}
                            <div className="absolute -top-1 -left-1 w-4 h-4 border-t-2 border-l-2 border-neon-red"></div>
                            <div className="absolute -top-1 -right-1 w-4 h-4 border-t-2 border-r-2 border-neon-red"></div>
                            <div className="absolute -bottom-1 -left-1 w-4 h-4 border-b-2 border-l-2 border-neon-red"></div>
                            <div className="absolute -bottom-1 -right-1 w-4 h-4 border-b-2 border-r-2 border-neon-red"></div>
                        </motion.div>
                    </div>

                    {/* Controls Overlay */}
                    <div className="absolute bottom-0 left-0 w-full p-4 bg-gradient-to-t from-black/80 to-transparent z-10 flex gap-4">
                        <button className="p-2 hover:bg-white/10 rounded-lg text-white transition-colors">
                            <Activity size={20} />
                        </button>
                        <button className="p-2 hover:bg-white/10 rounded-lg text-white transition-colors">
                            <Eye size={20} />
                        </button>
                        <div className="flex-1"></div>
                        <div className="flex items-center gap-2 text-xs text-neon-blue font-mono">
                            1920x1080 • 60FPS • 4.2Mbps
                        </div>
                    </div>
                </div>

                {/* Additional Cameras Grid Mockup */}
                <div className="grid grid-cols-3 gap-4 h-32">
                    {[1, 2, 3].map(i => (
                        <div key={i} className="bg-cyber-gray border border-white/10 rounded-lg relative overflow-hidden group cursor-pointer hover:border-neon-blue transition-colors">
                            <div className="absolute inset-0 bg-black/50 group-hover:bg-transparent transition-colors"></div>
                            <div className="absolute top-2 left-2 text-xs text-gray-400 font-mono">CAM-0{i}</div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Right Column: Real-time Logs */}
            <div className="bg-cyber-gray border border-white/10 rounded-xl flex flex-col h-full overflow-hidden">
                <div className="p-4 border-b border-white/10 flex justify-between items-center">
                    <h3 className="font-bold text-white flex items-center gap-2">
                        <Activity className="text-neon-blue" size={18} />
                        Event Stream
                    </h3>
                    <button className="p-1.5 hover:bg-white/5 rounded text-gray-400 hover:text-white">
                        <Search size={16} />
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto p-2 space-y-2 font-mono text-sm">
                    <AnimatePresence initial={false}>
                        {logs.map((log) => (
                            <motion.div
                                key={log.id}
                                initial={{ opacity: 0, x: 20 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0 }}
                                className={`p-3 rounded border-l-2 ${log.type === 'ALERT' ? 'bg-red-500/10 border-red-500 text-red-200' :
                                    log.type === 'WARNING' ? 'bg-yellow-500/10 border-yellow-500 text-yellow-200' :
                                        'bg-white/5 border-neon-blue text-gray-300'
                                    }`}
                            >
                                <div className="flex justify-between items-center text-xs opacity-70 mb-1">
                                    <span>[{log.time}]</span>
                                    <span>{log.source}</span>
                                </div>
                                <div>
                                    {log.type === 'ALERT' && <AlertTriangle size={14} className="inline mr-1" />}
                                    {log.message}
                                </div>
                            </motion.div>
                        ))}
                    </AnimatePresence>
                </div>
            </div>
        </div>
    );
};

export default MonitorPage;
