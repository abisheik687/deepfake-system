export function VerdictBadge({ verdict, size = 'md', pulse = true }) {
    const v = verdict?.toUpperCase() || 'UNCERTAIN';
    
    const sizes = {
        sm: 'px-2 py-0.5 text-[10px]', 
        md: 'px-3 py-1 text-xs',
        lg: 'px-4 py-1.5 text-sm', 
        xl: 'px-6 py-2 text-base'
    };
    
    // Core styling based on sprint 10 UI requirements
    let styleClasses = "inline-flex items-center font-black tracking-[0.15em] rounded-md border ";
    
    if (v === 'FAKE' || v === 'DEEPFAKE') {
        styleClasses += `bg-red-500/10 text-red-500 border-red-500/50 ${pulse ? 'animate-pulse [box-shadow:0_0_15px_rgba(239,68,68,0.3)]' : ''}`;
    } else if (v === 'SUSPICIOUS' || v === 'UNCERTAIN') {
        styleClasses += `bg-amber-500/10 text-amber-500 border-amber-500/50 ${pulse ? 'animate-[pulse_3s_ease-in-out_infinite]' : ''}`;
    } else if (v === 'REAL' || v === 'AUTHENTIC') {
        styleClasses += `bg-teal-500/10 text-teal-400 border-teal-500/30`;
    } else if (v === 'PROCESSING') {
        styleClasses += `bg-cyan-500/10 text-cyan-400 border-cyan-500/30 animate-pulse`;
    } else {
        styleClasses += `bg-gray-500/10 text-gray-400 border-gray-500/30`;
    }

    return (
        <span className={`${styleClasses} ${sizes[size]}`}>
            {v}
        </span>
    );
}
