import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { agencyAPI } from '../services/api';

export default function AgencyPage() {
    const [status, setStatus] = useState(null);
    const [history, setHistory] = useState([]);
    const [logs, setLogs] = useState([]);
    const [loading, setLoading] = useState(true);

    const agentIcons = {
        'agent_fc_01': '🛡️',
        'agent_fa_02': '🔬',
        'agent_jr_03': '📝'
    };

    useEffect(() => {
        const fetchData = async () => {
            try {
                const statusData = await agencyAPI.getStatus();
                const historyData = await agencyAPI.getInvestigationHistory();
                const logsData = await agencyAPI.getLogs();
                
                setStatus(statusData);
                setHistory(historyData);
                setLogs(logsData);

            } catch (err) {
                console.error("Failed to fetch Agency data", err);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
        const interval = setInterval(fetchData, 4000);
        return () => clearInterval(interval);
    }, []);

    if (loading) return (
        <div className='h-screen flex items-center justify-center bg-[#050a14]'>
            <div className='text-center'>
                <div className='w-16 h-16 border-4 border-cyan-500/20 border-t-cyan-500 rounded-full animate-spin mb-4 mx-auto' />
                <p className='text-cyan-500 font-mono text-sm tracking-widest'>INITIATING MISSION CONTROL AGENCY...</p>
            </div>
        </div>
    );

    return (
        <div className='p-8 space-y-8 max-w-7xl mx-auto min-h-screen'>
            
            {/* ── Dashboard Header ── */ }
            <div className='flex flex-col md:flex-row justify-between items-start md:items-end gap-4'>
                <div>
                    <div className='flex items-center gap-3 mb-2'>
                        <span className='px-2 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/30 text-[10px] font-bold text-cyan-500 tracking-widest uppercase'>Active Operation v2.0</span>
                    </div>
                    <h1 className='text-4xl font-black italic tracking-tighter' style={{ color: 'var(--text-primary)' }}>
                        AGENCY <span style={{ color: 'var(--cyan)' }}>MISSION CONTROL</span>
                    </h1>
                    <p className='text-sm max-w-xl' style={{ color: 'var(--text-muted)' }}>
                        Decentralized autonomous reasoning hub for high-stakes forensic synthesis and threat intelligence.
                    </p>
                </div>
                <div className='flex gap-4'>
                    <div className='text-right'>
                        <p className='text-[10px] uppercase font-bold text-gray-500'>Forensic Uptime</p>
                        <p className='text-xl font-mono' style={{ color: 'var(--text-primary)' }}>99.98%</p>
                    </div>
                    <div className='w-px h-10 bg-white/10' />
                    <div className='text-right'>
                        <p className='text-[10px] uppercase font-bold text-gray-500'>Agent Cohesion</p>
                        <p className='text-xl font-mono text-cyan-400'>OPTIMAL</p>
                    </div>
                </div>
            </div>

            {/* ── Master Agent Grid ── */}
            <div className='grid grid-cols-1 lg:grid-cols-3 gap-6'>
                {status?.active_agents.map((agent, i) => (
                    <motion.div 
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: i * 0.1 }}
                        key={agent.id} 
                        className='relative group rounded-3xl p-8 border overflow-hidden'
                        style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}
                    >
                        {/* Background Decor */}
                        <div className='absolute -right-4 -top-4 text-8xl opacity-[0.03] grayscale pointer-events-none'>
                            {agentIcons[agent.id]}
                        </div>

                        <div className='relative z-10'>
                            <div className='flex justify-between items-center mb-6'>
                                <div className='w-12 h-12 rounded-2xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-2xl'>
                                    {agentIcons[agent.id]}
                                </div>
                                <div className='flex flex-col items-end'>
                                    <span className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[9px] font-black uppercase tracking-widest ${agent.status === 'PROCESSING' ? 'bg-yellow-500/10 text-yellow-500 border border-yellow-500/20' : 'bg-green-500/10 text-green-500 border border-green-500/20'}`}>
                                        <span className={`w-1.5 h-1.5 rounded-full ${agent.status === 'PROCESSING' ? 'bg-yellow-500 animate-pulse' : 'bg-green-500'}`} />
                                        {agent.status}
                                    </span>
                                </div>
                            </div>

                            <h3 className='text-xl font-bold mb-1' style={{ color: 'var(--text-primary)' }}>{agent.name}</h3>
                            <p className='text-[10px] font-mono text-cyan-500/50 mb-6 uppercase tracking-widest'>{agent.id}</p>
                            
                            <div className='space-y-4'>
                                <div className='bg-black/20 p-4 rounded-2xl border border-white/5'>
                                    <p className='text-[9px] uppercase font-bold text-gray-500 mb-2 tracking-widest'>Intelligence Profile</p>
                                    <div className='flex flex-wrap gap-2'>
                                        {agent.capabilities.map(cap => (
                                            <span key={cap} className='text-[10px] text-gray-400 flex items-center gap-1'>
                                                <span className='w-1 h-1 rounded-full bg-cyan-500/40' /> {cap}
                                            </span>
                                        ))}
                                    </div>
                                </div>

                                <div className='flex justify-between items-center px-2'>
                                    <span className='text-xs text-gray-500'>Investigations</span>
                                    <span className='text-lg font-mono font-bold' style={{ color: 'var(--text-primary)' }}>{agent.investigations_count}</span>
                                </div>
                            </div>
                        </div>
                    </motion.div>
                ))}
            </div>

            <div className='grid grid-cols-1 xl:grid-cols-2 gap-8'>
                
                {/* ── Active Investigation Stream ── */}
                <div className='rounded-3xl border overflow-hidden' style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                    <div className='px-8 py-5 border-b flex justify-between items-center' style={{ borderBottom: '1px solid var(--border)', background: 'rgba(255,255,255,0.02)' }}>
                        <div className='flex items-center gap-3'>
                            <div className='w-2 h-2 rounded-full bg-red-500 animate-ping' />
                            <h2 className='text-xs font-black uppercase tracking-[0.2em]' style={{ color: 'var(--text-primary)' }}>Live Neural Inquest Log</h2>
                        </div>
                        <span className='text-[10px] font-mono text-gray-500'>AGENT_COMM_OS_v2.0</span>
                    </div>
                    <div className='p-6 space-y-4 font-mono text-xs'>
                        <AnimatePresence mode='popLayout'>
                            {logs.map((log) => (
                                <motion.div 
                                    layout
                                    initial={{ opacity: 0, x: -10 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, scale: 0.95 }}
                                    key={log.id} 
                                    className='flex gap-4'
                                >
                                    <span className='text-gray-600 shrink-0'>[{log.time}]</span>
                                    <span style={{ color: 'var(--cyan)' }} className='shrink-0 uppercase font-bold'>{log.agent}:</span>
                                    <span style={{ color: 'var(--text-secondary)' }}>{log.message}</span>
                                </motion.div>
                            ))}
                        </AnimatePresence>
                    </div>
                </div>

                {/* ── Forensic Investigation History ── */}
                <div className='rounded-3xl border overflow-hidden flex flex-col' style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                    <div className='px-8 py-5 border-b flex justify-between items-center' style={{ borderBottom: '1px solid var(--border)', background: 'rgba(255,255,255,0.02)' }}>
                        <h2 className='text-xs font-black uppercase tracking-[0.2em]' style={{ color: 'var(--text-primary)' }}>Historical Syntheses</h2>
                    </div>
                    <div className='flex-1 overflow-y-auto max-h-[400px] divide-y divide-white/5'>
                        {history.map((inv) => (
                            <div key={inv.id} className='group px-8 py-5 flex items-center justify-between hover:bg-cyan-500/5 transition-all cursor-pointer'>
                                <div className='flex gap-6 items-center'>
                                    <div className={`w-12 h-12 rounded-2xl flex items-center justify-center font-mono font-black text-sm transition-all group-hover:scale-110
                                        ${inv.risk_score > 70 ? 'bg-red-500/10 text-red-500 border border-red-500/20' : 'bg-cyan-500/10 text-cyan-500 border border-cyan-500/20'}`}>
                                        {inv.risk_score}%
                                    </div>
                                    <div>
                                        <p className='text-sm font-bold tracking-tight mb-0.5' style={{ color: 'var(--text-primary)' }}>
                                            {inv.conclusion}
                                        </p>
                                        <div className='flex items-center gap-3 text-[10px] text-gray-500 uppercase font-bold tracking-widest'>
                                            <span>INCIDENT: {inv.alert_id}</span>
                                            <span className='w-1 h-1 rounded-full bg-white/10' />
                                            <span>TICKET: {inv.id}</span>
                                        </div>
                                    </div>
                                </div>
                                <div className='opacity-0 group-hover:opacity-100 transition-opacity text-cyan-500 text-lg'>
                                    →
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

            </div>

        </div>
    );
}
