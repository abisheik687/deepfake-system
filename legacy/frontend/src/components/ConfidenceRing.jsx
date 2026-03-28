import React from 'react';
import { motion } from 'framer-motion';

export function ConfidenceRing({ percentage, size = 120, strokeWidth = 8, verdict }) {
    const radius = (size - strokeWidth) / 2;
    const circumference = radius * 2 * Math.PI;
    const strokeDashoffset = circumference - (percentage / 100) * circumference;
    
    // Choose color based on verdict or score
    let strokeColor = 'var(--cyan)'; // Default cyan
    if (verdict === 'FAKE' || verdict === 'DEEPFAKE' || percentage > 75) strokeColor = '#EF4444'; // red-500
    else if (verdict === 'SUSPICIOUS' || (percentage > 40 && percentage <= 75)) strokeColor = '#F59E0B'; // amber-500
    else if (verdict === 'REAL' || verdict === 'AUTHENTIC') strokeColor = '#14B8A6'; // teal-500

    return (
        <div className="relative flex items-center justify-center font-mono" style={{ width: size, height: size }}>
            {/* Background ring */}
            <svg width={size} height={size} className="transform -rotate-90">
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    stroke="currentColor"
                    strokeWidth={strokeWidth}
                    fill="transparent"
                    className="text-gray-800"
                />
                {/* Foreground animated ring */}
                <motion.circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    stroke={strokeColor}
                    strokeWidth={strokeWidth}
                    fill="transparent"
                    strokeDasharray={circumference}
                    initial={{ strokeDashoffset: circumference }}
                    animate={{ strokeDashoffset }}
                    transition={{ duration: 1.5, ease: "easeOut" }}
                    strokeLinecap="round"
                    className="drop-shadow-md"
                />
            </svg>
            <div className="absolute flex flex-col items-center justify-center">
                <span className="text-2xl font-black" style={{ color: 'var(--text-primary)' }}>
                    {percentage.toFixed(1)}%
                </span>
                <span className="text-[9px] uppercase tracking-widest text-gray-500 font-bold">
                    Confidence
                </span>
            </div>
        </div>
    );
}
