export function ConfidenceMeter({ value, size = 120 }) {
    // value: 0–100
    const r = (size - 16) / 2;
    const circ = 2 * Math.PI * r;
    const offset = circ - (value / 100) * circ;
    const color = value >= 80 ? '#E63946' : value >= 50 ? '#F4A261' : '#2DC653';

    return (
        <div className='relative inline-flex items-center justify-center'
            style={{ width: size, height: size }}>
            <svg width={size} height={size} className='-rotate-90'>
                <circle cx={size / 2} cy={size / 2} r={r}
                    fill='none' stroke='#1a2f4a' strokeWidth={8} />
                <circle cx={size / 2} cy={size / 2} r={r}
                    fill='none' stroke={color} strokeWidth={8}
                    strokeDasharray={circ} strokeDashoffset={offset}
                    strokeLinecap='round'
                    style={{ transition: 'stroke-dashoffset 0.6s ease' }} />
            </svg>
            <div className='absolute flex flex-col items-center'>
                <span className='font-bold text-white' style={{ fontSize: size * 0.22 }}>
                    {value.toFixed(1)}%
                </span>
                <span className='text-xs' style={{ color: 'var(--text-muted)' }}>
                    confidence
                </span>
            </div>
        </div>
    );
}
