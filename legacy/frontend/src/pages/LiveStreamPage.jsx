import { useState, useEffect, useRef } from 'react';
import { VerdictBadge } from '../components/VerdictBadge';

const MAX_LOG = 80;

function WSIndicator({ status }) {
    const colors = { connected: '#2DC653', reconnecting: '#F4A261', disconnected: '#6B7E94' };
    return (
        <div className='flex items-center gap-1.5'>
            <div className={`w-2 h-2 rounded-full
                      ${status === 'connected' ? 'animate-pulse' : ''}`}
                style={{ background: colors[status] ?? colors.disconnected }} />
            <span className='text-xs' style={{ color: 'var(--text-muted)' }}>{status}</span>
        </div>
    );
}

export default function LiveStreamPage() {
    const [url, setUrl] = useState('');
    const [streaming, setStreaming] = useState(false);
    const [wsStatus, setWsStatus] = useState('disconnected');
    const [log, setLog] = useState([]);
    const [soundEnabled, setSoundEnabled] = useState(false);
    const wsRef = useRef(null);
    const logRef = useRef(null);
    const audioCtxRef = useRef(null);

    const playAlert = () => {
        if (!soundEnabled) return;
        const ctx = audioCtxRef.current ?? new AudioContext();
        audioCtxRef.current = ctx;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain); gain.connect(ctx.destination);
        osc.frequency.value = 880;
        gain.gain.setValueAtTime(0.3, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4);
        osc.start(); osc.stop(ctx.currentTime + 0.4);
    };

    const startStream = () => {
        if (!url) return;
        setStreaming(true); setWsStatus('reconnecting'); setLog([]);

        const ws = new WebSocket(
            `ws://localhost:8000/ws/live?url=${encodeURIComponent(url)}`
        );
        wsRef.current = ws;

        ws.onopen = () => setWsStatus('connected');
        ws.onclose = () => { setWsStatus('disconnected'); setStreaming(false); };
        ws.onerror = () => setWsStatus('reconnecting');

        ws.onmessage = (e) => {
            const msg = JSON.parse(e.data);
            if (msg.verdict === 'DEEPFAKE') playAlert();
            setLog(prev => [
                { ...msg, id: Date.now() },
                ...prev.slice(0, MAX_LOG - 1),
            ]);
        };
    };

    const stopStream = () => {
        wsRef.current?.close();
        setStreaming(false); setWsStatus('disconnected');
    };

    // Auto-scroll log to top
    useEffect(() => {
        logRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
    }, [log.length]);

    return (
        <div className='p-6 max-w-7xl mx-auto'>
            <div className='flex items-center justify-between mb-6'>
                <h1 className='text-2xl font-bold' style={{ color: 'var(--text-primary)' }}>
                    Live Stream Detection
                </h1>
                <WSIndicator status={wsStatus} />
            </div>

            {/* ── URL input + controls ── */}
            <div className='flex gap-3 mb-6'>
                <input
                    type='url' value={url} onChange={e => setUrl(e.target.value)}
                    disabled={streaming}
                    placeholder='https://youtube.com/watch?v=…  or  rtsp://192.168.1.1:554/stream'
                    className='flex-1 px-4 py-2.5 rounded-lg text-sm font-mono
                     disabled:opacity-50'
                    style={{
                        background: 'var(--bg-card)', border: '1px solid var(--border)',
                        color: 'var(--text-primary)', outline: 'none'
                    }}
                    aria-label='Stream URL'
                />
                {!streaming
                    ? <button onClick={startStream} disabled={!url}
                        className='px-5 py-2.5 rounded-lg text-sm font-bold
                               disabled:opacity-40 transition-all'
                        style={{ background: 'var(--cyan)', color: '#0A1628' }}>
                        ◉ Start
                    </button>
                    : <button onClick={stopStream}
                        className='px-5 py-2.5 rounded-lg text-sm font-bold transition-all'
                        style={{ background: 'var(--verdict-fake)', color: '#fff' }}>
                        ⬛ Stop
                    </button>
                }
                <button onClick={() => setSoundEnabled(s => !s)}
                    className='px-4 py-2.5 rounded-lg text-sm transition-all'
                    style={{
                        background: soundEnabled ? '#1a2f4a' : 'var(--bg-card)',
                        border: '1px solid var(--border)',
                        color: soundEnabled ? 'var(--cyan)' : 'var(--text-muted)',
                    }}
                    aria-label='Toggle alert sound'
                    title='Toggle deepfake alert sound'>
                    {soundEnabled ? '🔔' : '🔕'}
                </button>
            </div>

            <p className='text-xs mb-4' style={{ color: 'var(--text-muted)' }}>
                Supported: YouTube • RTSP • Direct MP4 URLs
            </p>

            <div className='grid grid-cols-1 lg:grid-cols-3 gap-6'>

                {/* ── Preview pane ── */}
                <div className='lg:col-span-2 rounded-xl overflow-hidden flex items-center
                         justify-center aspect-video'
                    style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                    {streaming
                        ? <iframe
                            src={url.includes('youtube') ? url.replace('watch?v=', 'embed/') : url}
                            title='Live stream' allow='autoplay'
                            className='w-full h-full border-0'
                        />
                        : <div className='flex flex-col items-center gap-3'>
                            <span className='text-5xl'>📡</span>
                            <p className='text-sm' style={{ color: 'var(--text-muted)' }}>
                                Enter a stream URL and press Start
                            </p>
                        </div>
                    }
                </div>

                {/* ── Detection log ── */}
                <div className='rounded-xl overflow-hidden flex flex-col'
                    style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', maxHeight: 480 }}>
                    <div className='px-4 py-3 flex items-center justify-between'
                        style={{ borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
                        <h2 className='text-sm font-semibold uppercase tracking-wide'
                            style={{ color: 'var(--text-secondary)' }}>
                            Detection Log
                        </h2>
                        <span className='text-xs' style={{ color: 'var(--text-muted)' }}>
                            {log.length} events
                        </span>
                    </div>
                    <div ref={logRef} className='flex-1 overflow-y-auto p-3 space-y-2'>
                        {log.length === 0
                            ? <p className='text-center text-sm py-8'
                                style={{ color: 'var(--text-muted)' }}>
                                Waiting for events...
                            </p>
                            : log.map(entry => (
                                <div key={entry.id}
                                    className='flex items-center gap-2 text-xs py-1.5 px-2 rounded'
                                    style={{ background: 'var(--bg-primary)' }}>
                                    <VerdictBadge verdict={entry.verdict} size='sm' />
                                    <span className='flex-1 font-mono truncate'
                                        style={{ color: 'var(--text-muted)' }}>
                                        {(entry.confidence * 100).toFixed(1)}%
                                    </span>
                                    <span style={{ color: 'var(--text-disabled)' }}>
                                        {new Date(entry.timestamp).toLocaleTimeString()}
                                    </span>
                                </div>
                            ))
                        }
                    </div>
                </div>
            </div>
        </div>
    );
}
