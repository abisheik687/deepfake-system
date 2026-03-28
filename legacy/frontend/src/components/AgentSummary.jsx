import React from 'react';

export default function AgentSummary({ findings, summary, reportPath }) {
    if (!findings && !summary && !reportPath) return null;

    return (
        <div className='rounded-xl overflow-hidden mt-6'
            style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
            <div className='px-4 py-3' style={{ borderBottom: '1px solid var(--border)', background: 'rgba(0, 255, 255, 0.05)' }}>
                <h2 className='text-sm font-semibold uppercase tracking-wide flex items-center gap-2'
                    style={{ color: 'var(--cyan)' }}>
                    <span>🛡️</span> AI Agency Mission Control Findings
                </h2>
            </div>
            
            <div className='p-6 space-y-6'>
                {/* Public Summary (Journalist) */}
                {summary && (
                    <div className='bg-black/20 p-4 rounded-lg border border-white/5'>
                        <h3 className='text-xs font-bold uppercase mb-2 text-gray-500'>Public Communication (Journalist Agent)</h3>
                        <p className='text-sm italic' style={{ color: 'var(--text-primary)' }}>
                            "{summary}"
                        </p>
                    </div>
                )}

                {/* Technical Findings (Fact-Checker / Forensic) */}
                {findings && findings.length > 0 && (
                    <div className='space-y-2'>
                        <h3 className='text-xs font-bold uppercase text-gray-500'>Forensic Evidence Chain</h3>
                        <ul className='space-y-2'>
                            {findings.map((finding, i) => (
                                <li key={i} className='text-sm flex gap-2 items-start'>
                                    <span className='text-cyan-500 font-bold'>↳</span>
                                    <span style={{ color: 'var(--text-secondary)' }}>{finding}</span>
                                </li>
                            ))}
                        </ul>
                    </div>
                )}

                {/* Forensic PDF Link */}
                {reportPath && (
                    <div className='pt-2'>
                        <a 
                            href={`/api/v1/reports/download/${reportPath.split('/').pop()}`}
                            target='_blank'
                            rel='noopener noreferrer'
                            className='inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-bold transition-all
                                     hover:bg-cyan-500 hover:text-black'
                            style={{ border: '1px solid var(--cyan)', color: 'var(--cyan)' }}
                        >
                            📄 Download Full Forensic PDF Report
                        </a>
                    </div>
                )}
            </div>
        </div>
    );
}
