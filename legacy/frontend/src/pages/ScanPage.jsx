import { useState, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useToast } from '../hooks/useToast';
import { Toast } from '../components/Toast';
import { unifiedAPI } from '../services/api';

// ── Pipeline Step Labels ────────────────────────────────────────────
const STEPS = ['Face Detection', 'Model Inference', 'Ensemble', 'Report'];

function ProgressPipeline({ stage }) {
    return (
        <div className='flex items-center gap-0 w-full my-6'>
            {STEPS.map((step, i) => {
                const idx = i + 1;
                const done = stage > idx;
                const active = stage === idx;
                return (
                    <div key={step} className='flex-1 flex items-center'>
                        <div className='flex flex-col items-center flex-1'>
                            <div className={`w-8 h-8 rounded-full flex items-center justify-center
                              text-xs font-bold transition-all duration-300
                              ${done ? 'bg-green-500 text-white'
                                    : active ? 'bg-cyan-500 text-black animate-pulse'
                                        : 'bg-gray-700 text-gray-500'}`}>
                                {done ? '✓' : idx}
                            </div>
                            <span className={`text-xs mt-1 text-center transition-colors
                               ${active ? 'text-cyan-400 font-semibold'
                                    : done ? 'text-green-400'
                                        : 'text-gray-600'}`}>
                                {step}
                            </span>
                        </div>
                        {i < STEPS.length - 1 && (
                            <div className='h-0.5 flex-1 mb-5 transition-all duration-500'
                                style={{ background: done ? '#2DC653' : '#1a2f4a' }} />
                        )}
                    </div>
                );
            })}
        </div>
    );
}

// ── Mode definitions ────────────────────────────────────────────────
const MODES = [
    { id: 'image', label: '🖼 Image', accept: 'image/jpeg,image/png,image/webp', maxMB: 20 },
    { id: 'video', label: '🎬 Video', accept: 'video/mp4,video/webm,video/avi,video/mov', maxMB: 500 },
    { id: 'url',   label: '🔗 URL / Stream', accept: null, maxMB: null },
];

const TIERS = [
    { id: 'fast',          label: 'Fast',          desc: '2 models · ~1s' },
    { id: 'balanced',      label: 'Balanced',       desc: '3 models · ~3s' },
    { id: 'comprehensive', label: 'Comprehensive',  desc: '7 models · ~10s' },
];

export default function ScanPage() {
    const navigate   = useNavigate();
    const { toasts, toast, remove } = useToast();
    const fileRef    = useRef();

    const [mode,      setMode]      = useState('image');
    const [tier,      setTier]      = useState('balanced');
    const [file,      setFile]      = useState(null);
    const [preview,   setPreview]   = useState(null); // object URL for image/video preview
    const [streamUrl, setStreamUrl] = useState('');
    const [dragOver,  setDragOver]  = useState(false);
    const [stage,     setStage]     = useState(0);
    const [scanning,  setScanning]  = useState(false);

    // Switch mode → clear previous selection
    const switchMode = (m) => {
        setMode(m);
        setFile(null);
        setPreview(null);
        setStreamUrl('');
        setStage(0);
        if (fileRef.current) fileRef.current.value = '';
    };

    const handleFile = useCallback((f) => {
        if (!f) return;
        const currentMode = MODES.find(m => m.id === mode) || MODES[0];
        const maxMB = currentMode.maxMB;
        if (maxMB && f.size > maxMB * 1024 * 1024) {
            toast(`File too large. Max ${maxMB}MB for ${mode}.`, 'error');
            return;
        }
        setFile(f);
        setPreview(URL.createObjectURL(f));
        setStage(0);
    }, [mode, toast]);

    const onDrop = useCallback((e) => {
        e.preventDefault();
        setDragOver(false);
        handleFile(e.dataTransfer.files[0]);
    }, [handleFile]);

    const clearFile = () => {
        if (preview) URL.revokeObjectURL(preview);
        setFile(null);
        setPreview(null);
        setStage(0);
        if (fileRef.current) fileRef.current.value = '';
    };

    const runScan = async () => {
        if (mode !== 'url' && !file) {
            toast('Please select a file', 'error');
            return;
        }
        if (mode === 'url' && !streamUrl.trim()) {
            toast('Please enter a URL', 'error');
            return;
        }

        setScanning(true);
        setStage(1);

        try {
            let data;

            if (mode === 'url') {
                setStage(2);
                data = await unifiedAPI.extensionScan({ url: streamUrl.trim(), tier });
            } else if (mode === 'video') {
                setStage(2);
                const formData = new FormData();
                formData.append('file', file);
                formData.append('tier', tier);
                formData.append('sample_fps', '1.0');
                formData.append('max_frames', '30');
                data = await unifiedAPI.analyzeVideo(formData);
            } else {
                // image
                setStage(2);
                data = await unifiedAPI.analyzeUnifiedFile(file, {
                    tier,
                    return_heatmap: true,
                    detect_faces: true,
                });
            }

            setStage(5);
            toast('Scan complete!', 'success');
            const scanId = data.detection_id ?? data.task_id ?? data.id;
            navigate(`/alerts/${scanId}`, { state: { result: data } });

        } catch (err) {
            console.error('[ScanPage] scan error:', err);
            toast(err?.response?.data?.detail || err?.message || 'Scan failed', 'error');
            setStage(0);
        } finally {
            setScanning(false);
        }
    };

    const currentMode = MODES.find(m => m.id === mode) || MODES[0];
    const canRun = !scanning && (mode === 'url' ? !!streamUrl.trim() : !!file);

    return (
        <div className='p-6 max-w-3xl mx-auto'>
            <h1 className='text-2xl font-bold mb-1' style={{ color: 'var(--text-primary)' }}>
                New Scan
            </h1>
            <p className='mb-6 text-sm' style={{ color: 'var(--text-muted)' }}>
                Upload an image or video, or provide a URL / stream for live detection.
            </p>

            {/* ── Mode tabs ── */}
            <div className='flex gap-2 mb-5'>
                {MODES.map(m => (
                    <button key={m.id} onClick={() => switchMode(m.id)}
                        className='px-4 py-2 rounded-md text-sm font-medium transition-all'
                        style={{
                            background: mode === m.id ? 'var(--cyan)' : 'var(--bg-card)',
                            color: mode === m.id ? '#0A1628' : 'var(--text-secondary)',
                            border: '1px solid var(--border)',
                        }}>
                        {m.label}
                    </button>
                ))}
            </div>

            {/* ── Tier selector ── */}
            <div className='flex gap-2 mb-5'>
                {TIERS.map(t => (
                    <button key={t.id} onClick={() => setTier(t.id)}
                        className='flex-1 py-2 rounded-md text-xs font-medium transition-all'
                        style={{
                            background: tier === t.id ? 'rgba(0,212,170,0.15)' : 'var(--bg-card)',
                            color: tier === t.id ? 'var(--cyan)' : 'var(--text-muted)',
                            border: `1px solid ${tier === t.id ? 'var(--cyan)' : 'var(--border)'}`,
                        }}>
                        <div className='font-bold'>{t.label}</div>
                        <div style={{ opacity: 0.7 }}>{t.desc}</div>
                    </button>
                ))}
            </div>

            {/* ── Input zone ── */}
            {mode === 'url' ? (
                <div className='space-y-3 mb-6'>
                    <input
                        type='url'
                        value={streamUrl}
                        onChange={e => setStreamUrl(e.target.value)}
                        placeholder='https://example.com/image.jpg  or  rtsp://192.168.1.1:554/stream'
                        className='w-full px-4 py-3 rounded-lg text-sm font-mono'
                        style={{
                            background: 'var(--bg-card)', border: '1px solid var(--border)',
                            color: 'var(--text-primary)', outline: 'none',
                        }}
                        aria-label='Media URL'
                        onKeyDown={e => e.key === 'Enter' && canRun && runScan()}
                    />
                    <p className='text-xs' style={{ color: 'var(--text-muted)' }}>
                        Supports: direct image URLs, YouTube links, RTSP streams, MP4 URLs
                    </p>
                </div>
            ) : (
                <>
                    {/* File drop zone */}
                    {!file ? (
                        <div
                            onDrop={onDrop}
                            onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                            onDragLeave={() => setDragOver(false)}
                            onClick={() => fileRef.current?.click()}
                            className='rounded-xl border-2 border-dashed cursor-pointer
                             flex flex-col items-center justify-center py-14 px-6
                             transition-all duration-200 mb-6'
                            style={{
                                borderColor: dragOver ? 'var(--cyan)' : 'var(--border)',
                                background: dragOver ? 'rgba(0,212,170,0.05)' : 'var(--bg-card)',
                            }}
                        >
                            <span className='text-4xl mb-3'>⬆</span>
                            <p className='font-medium text-sm' style={{ color: 'var(--text-primary)' }}>
                                Drop file here or click to browse
                            </p>
                            <p className='text-xs mt-1' style={{ color: 'var(--text-muted)' }}>
                                {mode === 'image'
                                    ? 'JPG, PNG, WebP — max 20MB'
                                    : 'MP4, WebM, AVI, MOV — max 500MB'}
                            </p>
                            <input
                                ref={fileRef}
                                type='file'
                                accept={currentMode.accept}
                                className='hidden'
                                onChange={e => handleFile(e.target.files[0])}
                            />
                        </div>
                    ) : (
                        <div className='rounded-xl overflow-hidden mb-6'
                            style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                            {/* Preview */}
                            {mode === 'image' && preview && (
                                <img src={preview} alt='Preview'
                                    className='w-full max-h-72 object-contain bg-black' />
                            )}
                            {mode === 'video' && preview && (
                                <video src={preview} controls
                                    className='w-full max-h-72 bg-black' />
                            )}
                            <div className='px-4 py-3 flex items-center justify-between gap-4'>
                                <div className='text-sm font-mono truncate'
                                    style={{ color: 'var(--text-secondary)' }}>
                                    {file.name} — {(file.size / 1024 / 1024).toFixed(2)} MB
                                </div>
                                <button onClick={clearFile}
                                    className='text-xs px-3 py-1 rounded'
                                    style={{ color: 'var(--text-muted)', border: '1px solid var(--border)' }}>
                                    Clear
                                </button>
                            </div>
                        </div>
                    )}
                </>
            )}

            {/* Pipeline progress */}
            {stage > 0 && <ProgressPipeline stage={stage} />}

            {/* Scan button */}
            <button
                onClick={runScan}
                disabled={!canRun}
                className='mt-2 w-full py-3 rounded-lg font-bold text-sm tracking-wide
                   transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed'
                style={{
                    background: scanning ? 'var(--cyan-dim, #007a5a)' : 'var(--cyan)',
                    color: '#0A1628',
                }}>
                {scanning ? 'Analysing…' : '⊕  Run Deepfake Scan'}
            </button>

            {toasts.map(t => (
                <Toast key={t.id} message={t.message} type={t.type} onClose={() => remove(t.id)} />
            ))}
        </div>
    );
}
