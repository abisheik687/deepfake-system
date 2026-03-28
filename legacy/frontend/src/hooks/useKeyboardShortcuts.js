import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

export function useKeyboardShortcuts() {
    const navigate = useNavigate();
    useEffect(() => {
        const handler = (e) => {
            // Only fire if no input/textarea is focused
            if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;
            if (e.key === 'd') navigate('/dashboard');
            if (e.key === 's') navigate('/scan');
            if (e.key === 'a') navigate('/alerts');
            if (e.key === 'l') navigate('/live');
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, [navigate]);
}
