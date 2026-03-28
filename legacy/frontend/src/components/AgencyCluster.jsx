import React from 'react';
import { motion } from 'framer-motion';

export default function AgencyCluster({ activeIndices = [], labels = {} }) {
    // Generate 144 nodes (12x12 grid)
    const nodes = Array.from({ length: 144 }, (_, i) => i);

    return (
        <div className='relative p-6 glass rounded-3xl overflow-hidden'>
            <div className='absolute inset-0 pixel-grid opacity-20 pointer-events-none' />
            
            <div className='relative z-10'>
                <div className='flex justify-between items-center mb-6'>
                    <div>
                        <h3 className='text-xs font-black uppercase tracking-[0.3em] text-cyan-500'>Agency Neural Matrix</h3>
                        <p className='text-[10px] text-gray-500 font-mono'>144 ACTIVE HYPER-NODES</p>
                    </div>
                    <div className='flex gap-2'>
                        <div className='flex items-center gap-1'>
                            <div className='w-1.5 h-1.5 rounded-full bg-cyan-500' />
                            <span className='text-[8px] uppercase text-gray-500'>Active</span>
                        </div>
                        <div className='flex items-center gap-1'>
                            <div className='w-1.5 h-1.5 rounded-full bg-white/10' />
                            <span className='text-[8px] uppercase text-gray-500'>Standby</span>
                        </div>
                    </div>
                </div>

                <div className='grid grid-cols-12 gap-1.5'>
                    {nodes.map((node) => {
                        const isActive = activeIndices.includes(node);
                        return (
                            <motion.div
                                key={node}
                                initial={{ opacity: 0, scale: 0 }}
                                animate={{ 
                                    opacity: 1, 
                                    scale: 1,
                                    backgroundColor: isActive ? '#00ffff' : 'rgba(255,255,255,0.05)',
                                    boxShadow: isActive ? '0 0 10px #00ffff' : 'none'
                                }}
                                transition={{ 
                                    delay: (node % 12 + Math.floor(node / 12)) * 0.01,
                                    duration: 0.2
                                }}
                                className='aspect-square rounded-[2px] cursor-help transition-all duration-300'
                                title={labels[node] || `Agent Node ${node + 1}`}
                            />
                        );
                    })}
                </div>

                <div className='mt-6 pt-6 border-t border-white/5 flex justify-between items-center'>
                    <div className='text-[10px] font-mono text-gray-400'>
                        CLUSTER_COHESION: <span className='text-cyan-400'>0.9984_SIGMA</span>
                    </div>
                    <div className='text-[10px] font-mono text-gray-500 uppercase italic'>
                        Neural Inquest Core v2.0
                    </div>
                </div>
            </div>
        </div>
    );
}
