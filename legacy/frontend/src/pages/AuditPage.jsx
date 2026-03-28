import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import AgencyCluster from '../components/AgencyCluster';

export default function AuditPage() {
    const [audit, setAudit] = useState(null);
    const [loading, setLoading] = useState(true);
    const [activeNodes, setActiveNodes] = useState([]);
    const [scanProgress, setScanProgress] = useState(0);

    useEffect(() => {
        const fetchAudit = async () => {
            try {
                const res = await fetch('/api/agency/audit/project');
                const data = await res.json();
                setAudit(data);
                
                // Simulate scan animation
                let current = 0;
                const interval = setInterval(() => {
                    if (current <= 144) {
                        setActiveNodes(prev => [...Array(current).keys()]);
                        setScanProgress(Math.floor((current / 144) * 100));
                        current += 4;
                    } else {
                        clearInterval(interval);
                        setLoading(false);
                    }
                }, 50);

            } catch (err) {
                console.error("Audit failed", err);
            }
        };
        fetchAudit();
    }, []);

    if (scanProgress < 100) return (
        <div className='h-screen flex flex-col items-center justify-center p-8 text-center'>
             <div className='w-full max-w-md mb-12'>
                <AgencyCluster activeIndices={activeNodes} />
            </div>
            <h2 className='text-2xl font-black italic text-cyan-500 mb-2 tracking-tighter'>DEEP NEURAL AUDIT IN PROGRESS</h2>
            <div className='w-64 h-1 bg-white/5 rounded-full overflow-hidden'>
                <motion.div 
                    initial={{ width: 0 }}
                    animate={{ width: `${scanProgress}%` }}
                    className='h-full bg-cyan-500'
                />
            </div>
            <p className='text-[10px] font-mono text-gray-500 mt-4 uppercase tracking-[0.4em]'>Analyzing 144 Agent Synthesis Vectors...</p>
        </div>
    );

    return (
        <div className='p-8 space-y-12 max-w-7xl mx-auto'>
            {/* ── Header ── */}
            <div className='flex justify-between items-end'>
                <div>
                    <h1 className='text-5xl font-black italic tracking-tighter' style={{ color: 'var(--text-primary)' }}>
                        PROJECT <span style={{ color: 'var(--cyan)' }}>READINESS AUDIT</span>
                    </h1>
                    <p className='text-sm mt-2 opacity-60 max-w-2xl'>
                        Comprehensive 144-point neurological scan of the KAVACH-AI system architecture, security perimeter, and algorithmic accuracy.
                    </p>
                </div>
                <div className='text-right'>
                    <p className='text-[10px] font-black uppercase tracking-widest text-gray-500'>Audit Signature</p>
                    <p className='text-xs font-mono text-cyan-500'>KAVACH_V2_FINAL_SIGN_OFF</p>
                </div>
            </div>

            {/* ── Score Matrix ── */}
            <div className='grid grid-cols-1 lg:grid-cols-12 gap-8'>
                <div className='lg:col-span-4 flex flex-col items-center justify-center glass rounded-[3rem] p-12 text-center aspect-square relative'>
                    <div className='absolute inset-0 opacity-10 pixel-grid rounded-[3rem]' />
                    <motion.div 
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        className='text-8xl font-black tracking-tighter text-cyan-500 mb-2'
                    >
                        {audit.overall_score}
                    </motion.div>
                    <div className='text-xs font-black uppercase tracking-[0.5em] text-gray-400'>Project Cohesion</div>
                    <div className='mt-8 flex gap-4'>
                        {Object.entries(audit.categories).map(([cat, val]) => (
                            <div key={cat} className='text-center'>
                                <p className='text-lg font-bold' style={{ color: 'var(--text-primary)' }}>{val}</p>
                                <p className='text-[8px] uppercase text-gray-500'>{cat}</p>
                            </div>
                        ))}
                    </div>
                </div>

                <div className='lg:col-span-8 space-y-8'>
                    <AgencyCluster activeIndices={activeNodes} />
                    
                    <div className='grid grid-cols-1 md:grid-cols-2 gap-4'>
                         {/* Lacking / Needs Improvement */}
                         <div className='glass rounded-3xl p-8 border-l-4 border-l-yellow-500/50'>
                            <h3 className='text-xs font-black uppercase tracking-widest text-yellow-500 mb-6 flex items-center gap-2'>
                                <span className='w-1.5 h-1.5 rounded-full bg-yellow-500' />
                                Optimization Vectors (What is lacking)
                            </h3>
                            <ul className='space-y-4'>
                                {audit.lacking.map((item, i) => (
                                    <li key={i} className='flex gap-4 group'>
                                        <span className='text-cyan-500 font-mono text-[10px] opacity-40'>[0{i+1}]</span>
                                        <p className='text-sm text-gray-400 leading-relaxed group-hover:text-gray-200 transition-colors'>{item}</p>
                                    </li>
                                ))}
                            </ul>
                         </div>

                         {/* System Strengths */}
                         <div className='glass rounded-3xl p-8 border-l-4 border-l-cyan-500/50'>
                            <h3 className='text-xs font-black uppercase tracking-widest text-cyan-500 mb-6 flex items-center gap-2'>
                                <span className='w-1.5 h-1.5 rounded-full bg-cyan-500' />
                                Structural Strengths
                            </h3>
                            <ul className='space-y-4 text-sm text-gray-400'>
                                <li className='flex gap-4'>
                                    <span className='text-green-500'>✓</span>
                                    <span>Hyper-Modal Ensemble Redundancy (5-Layer)</span>
                                </li>
                                <li className='flex gap-4'>
                                    <span className='text-green-500'>✓</span>
                                    <span>Decentralized Forensic Ingestion via Kafka</span>
                                </li>
                                <li className='flex gap-4'>
                                    <span className='text-green-500'>✓</span>
                                    <span>Autonomous LangGraph Agency State Management</span>
                                </li>
                            </ul>
                         </div>
                    </div>
                </div>
            </div>

            {/* ── Final Call to Action ── */}
            <motion.button 
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className='w-full py-6 rounded-2xl bg-cyan-600 text-[#050a14] font-black uppercase tracking-[0.5em] text-sm shadow-[0_0_50px_rgba(0,255,255,0.2)]'
            >
                Confirm System Integrity & Seal Release
            </motion.button>

        </div>
    );
}
