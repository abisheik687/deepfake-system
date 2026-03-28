import React from 'react';
import { motion } from 'framer-motion';

export function ModelBreakdownBars({ scores }) {
    if (!scores || Object.keys(scores).length === 0) return (
        <p className="text-gray-500 text-sm italic p-4">No model telemetry available.</p>
    );

    return (
        <div className="space-y-5 p-4">
            {Object.entries(scores).map(([modelName, score], idx) => {
                const pct = (score * 100).toFixed(1);
                // Color scaling based on score threshold
                const color = score >= 0.8 ? '#EF4444' : score >= 0.5 ? '#F59E0B' : '#14B8A6';
                
                return (
                    <div key={modelName} className="space-y-1.5">
                        <div className="flex justify-between items-center text-xs">
                            <span className="font-mono text-gray-300 tracking-wider uppercase">{modelName}</span>
                            <span className="font-black font-mono" style={{ color }}>{pct}%</span>
                        </div>
                        <div className="h-2 w-full bg-black/40 rounded-full overflow-hidden border border-white/5 relative">
                            {/* Animated fill bar */}
                            <motion.div 
                                initial={{ width: 0 }}
                                animate={{ width: `${pct}%` }}
                                transition={{ duration: 1, delay: idx * 0.1, ease: 'easeOut' }}
                                className="h-full rounded-full relative"
                                style={{ backgroundColor: color }}
                            >
                                {/* Shine effect */}
                                <div className="absolute top-0 right-0 bottom-0 left-0 bg-gradient-to-r from-transparent via-white/20 to-transparent" />
                            </motion.div>
                        </div>
                    </div>
                );
            })}
        </div>
    );
}
