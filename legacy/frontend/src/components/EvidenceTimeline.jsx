import React from 'react';
import { motion } from 'framer-motion';

export function EvidenceTimeline({ events }) {
    if (!events || events.length === 0) return null;
    
    return (
        <div className="relative border-l border-white/10 ml-3 space-y-6 pb-4">
            {events.map((event, index) => (
                <motion.div 
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.15 }}
                    key={index} 
                    className="relative pl-6"
                >
                    {/* Timeline dot */}
                    <div className="absolute -left-[5px] top-1.5 w-2.5 h-2.5 rounded-full bg-cyan-500 border-2 border-[#13161d] shadow-[0_0_8px_rgba(0,212,170,0.5)]" />
                    
                    <div className="flex flex-col">
                        <span className="text-[10px] font-black uppercase tracking-widest text-cyan-500 mb-1">
                            {event.timestamp || new Date().toISOString()}
                        </span>
                        <h4 className="text-sm font-bold text-white mb-1">{event.action}</h4>
                        
                        {event.hash && (
                            <div className="bg-black/30 px-2 py-1.5 rounded border border-white/5 font-mono text-[10px] text-gray-400 break-all mt-1">
                                <span className="text-gray-600 select-none mr-2">SHA-256</span>
                                {event.hash}
                            </div>
                        )}
                        
                        {event.details && (
                            <p className="text-xs text-gray-500 mt-1">{event.details}</p>
                        )}
                    </div>
                </motion.div>
            ))}
        </div>
    );
}
