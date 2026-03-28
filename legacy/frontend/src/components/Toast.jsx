import { useEffect } from 'react';

const ICONS = { success: '✅', error: '❌', info: 'ℹ️', warning: '⚠️' };

export function Toast({ message, type = 'info', duration = 3500, onClose }) {
    useEffect(() => {
        const t = setTimeout(onClose, duration);
        return () => clearTimeout(t);
    }, [duration, onClose]);

    const colors = {
        success: 'border-green-500 bg-green-900/40',
        error: 'border-red-500   bg-red-900/40',
        info: 'border-cyan-500  bg-cyan-900/40',
        warning: 'border-orange-400 bg-orange-900/40',
    };

    return (
        <div className={`fixed bottom-6 right-6 z-50 flex items-center gap-3
                     px-4 py-3 rounded-lg border backdrop-blur-sm
                     shadow-2xl animate-slide-up ${colors[type]}`}>
            <span>{ICONS[type]}</span>
            <p className='text-sm text-white font-medium'>{message}</p>
            <button onClick={onClose}
                className='ml-2 text-gray-400 hover:text-white transition-colors'>✕</button>
        </div>
    );
}
