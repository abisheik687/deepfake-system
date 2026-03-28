import { useState, useEffect, useRef, useCallback } from 'react';
import { Radio, AlertTriangle, Activity, Eye, EyeOff, Camera, CameraOff, ShieldCheck, ShieldAlert, Wifi } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// ─── Constants ────────────────────────────────────────────────────────────────
const WS_URL = 'ws://localhost:8000/ws';
const FRAME_INTERVAL_MS = 900; // Send one frame per ~900ms to avoid overloading

// ─── Helpers ─────────────────────────────────────────────────────────────────
function captureFrame(videoEl, canvasEl) {
    if (!videoEl || !canvasEl) return null;
    const { videoWidth: vw, videoHeight: vh } = videoEl;
    if (!vw || !vh) return null;
    canvasEl.width = vw;
    canvasEl.height = vh;
    const ctx = canvasEl.getContext('2d');
    // Mirror the frame to match the mirrored video display
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(videoEl, -vw, 0, vw, vh);
    ctx.restore();
    return canvasEl.toDataURL('image/jpeg', 0.6);
}

const VERDICT_CONFIG = {
    REAL: { color: '#00ff9d', bg: 'rgba(0,255,157,0.12)', border: '#00ff9d', icon: ShieldCheck, label: 'REAL' },
    FAKE: { color: '#ff003c', bg: 'rgba(255,0,60,0.12)', border: '#ff003c', icon: ShieldAlert, label: 'FAKE DETECTED' },
    NO_FACE: { color: '#94a3b8', bg: 'rgba(148,163,184,0.08)', border: '#475569', icon: Eye, label: 'NO FACE' },
    ERROR: { color: '#f59e0b', bg: 'rgba(245,158,11,0.10)', border: '#f59e0b', icon: AlertTriangle, label: 'ERROR' },
};

// ─── Component ────────────────────────────────────────────────────────────────
const MonitorPage = () => {
    // Camera / DOM refs
    const videoRef = useRef(null);
    const captureCanvas = useRef(null);    // offscreen: frame extraction
    const overlayCanvas = useRef(null);    // onscreen: bounding boxes
    const streamRef = useRef(null);
    const wsRef = useRef(null);
    const timerRef = useRef(null);

    // State
    const [camStatus, setCamStatus] = useState('idle'); // idle | requesting | active | denied | error
    const [scanning, setScanning] = useState(false);
    const [wsStatus, setWsStatus] = useState('disconnected'); // connected | disconnected
    const [verdict, setVerdict] = useState(null); // REAL | FAKE | NO_FACE | ERROR
    const [confidence, setConfidence] = useState(0);
    const [logs, setLogs] = useState([]);

    const addLog = useCallback((type, source, message) => {
        setLogs(prev => [{
            id: Date.now() + Math.random(),
            time: new Date().toLocaleTimeString(),
            type, source, message
        }, ...prev].slice(0, 60));
    }, []);

    // ── WebSocket setup ──────────────────────────────────────────────────────
    useEffect(() => {
        let ws;
        const connect = () => {
            ws = new WebSocket(WS_URL);
            wsRef.current = ws;

            ws.onopen = () => {
                setWsStatus('connected');
                addLog('INFO', 'SYS', 'WebSocket secure connection established');
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);

                    if (data.type === 'frame_result') {
                        const { verdict: v, confidence: c, faces, message } = data;
                        setVerdict(v);
                        setConfidence(c);

                        // Draw bounding boxes on overlay canvas
                        drawOverlay(faces);

                        const isAlert = v === 'FAKE';
                        addLog(isAlert ? 'ALERT' : 'INFO', 'CAM-01', message);

                    } else {
                        // System / alert messages
                        const isAlert = data.type === 'alert' || data.data?.severity === 'high';
                        addLog(isAlert ? 'ALERT' : 'INFO', 'SYS', data.message || `System: ${data.type}`);
                    }
                } catch (e) {
                    console.error('WS parse error', e);
                }
            };

            ws.onclose = () => {
                setWsStatus('disconnected');
                addLog('WARNING', 'SYS', 'WebSocket connection lost — retrying in 3s');
                setTimeout(connect, 3000);
            };

            ws.onerror = () => {
                setWsStatus('disconnected');
            };
        };

        connect();
        return () => {
            ws?.close();
            clearInterval(timerRef.current);
        };
    }, [addLog]);

    // ── Overlay drawing ──────────────────────────────────────────────────────
    const drawOverlay = useCallback((faces) => {
        const canvas = overlayCanvas.current;
        const video = videoRef.current;
        if (!canvas || !video) return;

        const { clientWidth: dw, clientHeight: dh } = video;
        canvas.width = dw;
        canvas.height = dh;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, dw, dh);

        if (!faces || faces.length === 0) return;

        faces.forEach(face => {
            // xn/yn/wn/hn are normalised (0–1) from the backend
            const x = face.xn * dw;
            const y = face.yn * dh;
            const w = face.wn * dw;
            const h = face.hn * dh;

            // Flip x since the video is mirrored
            const flippedX = dw - x - w;

            const color = verdict === 'FAKE' ? '#ff003c' : '#00ff9d';

            // Main border
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(flippedX, y, w, h);

            // Corner accents
            const cornerLen = Math.min(w, h) * 0.18;
            ctx.lineWidth = 3;
            [[flippedX, y], [flippedX + w, y], [flippedX, y + h], [flippedX + w, y + h]].forEach(([cx, cy], i) => {
                const xDir = i % 2 === 0 ? 1 : -1;
                const yDir = i < 2 ? 1 : -1;
                ctx.beginPath();
                ctx.moveTo(cx + xDir * cornerLen, cy);
                ctx.lineTo(cx, cy);
                ctx.lineTo(cx, cy + yDir * cornerLen);
                ctx.stroke();
            });

            // Label badge
            const label = verdict === 'FAKE' ? `FAKE ${Math.round(confidence * 100)}%` : `REAL ${Math.round(confidence * 100)}%`;
            const badgePad = 6;
            ctx.font = 'bold 13px monospace';
            const tw = ctx.measureText(label).width;
            ctx.fillStyle = color;
            ctx.fillRect(flippedX - 1, y - 24, tw + badgePad * 2, 22);
            ctx.fillStyle = verdict === 'FAKE' ? '#000' : '#000';
            ctx.fillText(label, flippedX + badgePad - 1, y - 8);
        });
    }, [verdict, confidence]);

    // ── Camera start ─────────────────────────────────────────────────────────
    const startCamera = useCallback(async () => {
        setCamStatus('requesting');
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
                audio: false
            });
            streamRef.current = stream;
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                await videoRef.current.play();
            }
            setCamStatus('active');
            addLog('INFO', 'CAM-01', 'Camera stream started — ready for analysis');
        } catch (err) {
            const denied = err.name === 'NotAllowedError';
            setCamStatus(denied ? 'denied' : 'error');
            addLog('ALERT', 'CAM-01', denied ? 'Camera permission denied by user' : `Camera error: ${err.message}`);
        }
    }, [addLog]);

    // ── Camera stop ──────────────────────────────────────────────────────────
    const stopCamera = useCallback(() => {
        clearInterval(timerRef.current);
        setScanning(false);
        streamRef.current?.getTracks().forEach(t => t.stop());
        streamRef.current = null;
        if (videoRef.current) videoRef.current.srcObject = null;
        const canvas = overlayCanvas.current;
        if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
        setVerdict(null);
        setCamStatus('idle');
        addLog('INFO', 'CAM-01', 'Camera stream stopped');
    }, [addLog]);

    // ── Scanning toggle ──────────────────────────────────────────────────────
    const toggleScanning = useCallback(() => {
        if (scanning) {
            clearInterval(timerRef.current);
            setScanning(false);
            addLog('INFO', 'SYS', 'Deep-scan analysis paused');
        } else {
            setScanning(true);
            addLog('INFO', 'SYS', 'Deep-scan analysis started — streaming frames to AI engine');
            timerRef.current = setInterval(() => {
                if (wsRef.current?.readyState !== WebSocket.OPEN) return;
                const frame = captureFrame(videoRef.current, captureCanvas.current);
                if (!frame) return;
                wsRef.current.send(JSON.stringify({ type: 'frame', data: frame }));
            }, FRAME_INTERVAL_MS);
        }
    }, [scanning, addLog]);

    // Auto-start scanning when camera becomes active
    useEffect(() => {
        if (camStatus === 'active' && !scanning) {
            toggleScanning();
        }
    }, [camStatus]); // eslint-disable-line react-hooks/exhaustive-deps

    const vc = verdict ? VERDICT_CONFIG[verdict] : null;
    const VerdictIcon = vc?.icon;

    return (
        <div className="h-[calc(100vh-100px)] grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* ─── Left: Camera Feed ─────────────────────────────────────────── */}
            <div className="lg:col-span-2 flex flex-col gap-4">

                {/* Camera viewport */}
                <div className="bg-cyber-gray border border-white/10 rounded-xl overflow-hidden relative flex-1 flex flex-col">

                    {/* Top bar */}
                    <div className="absolute top-0 left-0 w-full p-3 bg-gradient-to-b from-black/90 to-transparent z-20 flex justify-between items-center">
                        <div className="flex items-center gap-2">
                            {scanning && camStatus === 'active' ? (
                                <div className="flex items-center gap-2 text-neon-red">
                                    <Radio size={14} className="animate-pulse" />
                                    <span className="font-bold text-xs tracking-widest uppercase">Live · Scanning</span>
                                </div>
                            ) : (
                                <div className="flex items-center gap-2 text-gray-500">
                                    <Radio size={14} />
                                    <span className="font-bold text-xs tracking-widest uppercase">Standby</span>
                                </div>
                            )}
                        </div>
                        <div className={`flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full border ${wsStatus === 'connected'
                                ? 'border-green-500/40 bg-green-500/10 text-green-400'
                                : 'border-red-500/40 bg-red-500/10 text-red-400'
                            }`}>
                            <Wifi size={11} />
                            {wsStatus === 'connected' ? 'AI Engine Online' : 'Reconnecting…'}
                        </div>
                    </div>

                    {/* Video + overlay */}
                    <div className="relative flex-1 flex items-center justify-center bg-black min-h-0" style={{ aspectRatio: '16/9' }}>
                        {/* Actual camera feed */}
                        <video
                            ref={videoRef}
                            className="w-full h-full object-cover"
                            style={{ transform: 'scaleX(-1)', display: camStatus === 'active' ? 'block' : 'none' }}
                            muted
                            playsInline
                        />

                        {/* Hidden capture canvas */}
                        <canvas ref={captureCanvas} style={{ display: 'none' }} />

                        {/* Visible overlay canvas (bounding boxes) */}
                        <canvas
                            ref={overlayCanvas}
                            className="absolute inset-0 w-full h-full pointer-events-none"
                            style={{ display: camStatus === 'active' ? 'block' : 'none' }}
                        />

                        {/* Scan line animation when active */}
                        {scanning && camStatus === 'active' && (
                            <div
                                className="absolute inset-0 pointer-events-none"
                                style={{
                                    background: 'linear-gradient(transparent 0%, rgba(0,243,255,0.04) 50%, transparent 100%)',
                                    backgroundSize: '100% 8px',
                                    animation: 'scan 2s linear infinite',
                                }}
                            />
                        )}

                        {/* Idle / Permission States */}
                        {camStatus !== 'active' && (
                            <div className="flex flex-col items-center justify-center gap-5 p-8 text-center">
                                {camStatus === 'idle' && (
                                    <>
                                        <div className="w-20 h-20 rounded-full bg-white/5 border border-white/10 flex items-center justify-center">
                                            <Camera size={36} className="text-gray-500" />
                                        </div>
                                        <div>
                                            <p className="text-white font-bold text-lg mb-1">Real-Time Deepfake Scanner</p>
                                            <p className="text-gray-500 text-sm">Grant camera access to begin live AI analysis</p>
                                        </div>
                                        <motion.button
                                            whileHover={{ scale: 1.04 }}
                                            whileTap={{ scale: 0.97 }}
                                            onClick={startCamera}
                                            className="px-6 py-2.5 bg-neon-blue text-black font-bold rounded-lg text-sm flex items-center gap-2 hover:bg-white transition-colors"
                                        >
                                            <Camera size={16} />
                                            Start Camera
                                        </motion.button>
                                    </>
                                )}

                                {camStatus === 'requesting' && (
                                    <>
                                        <motion.div
                                            animate={{ scale: [1, 1.15, 1] }}
                                            transition={{ repeat: Infinity, duration: 1.4 }}
                                            className="w-20 h-20 rounded-full bg-neon-blue/10 border border-neon-blue/30 flex items-center justify-center"
                                        >
                                            <Camera size={36} className="text-neon-blue" />
                                        </motion.div>
                                        <p className="text-gray-400 text-sm">Requesting camera permission…</p>
                                    </>
                                )}

                                {camStatus === 'denied' && (
                                    <>
                                        <div className="w-20 h-20 rounded-full bg-red-500/10 border border-red-500/30 flex items-center justify-center">
                                            <CameraOff size={36} className="text-red-400" />
                                        </div>
                                        <div>
                                            <p className="text-red-400 font-bold mb-1">Camera Permission Denied</p>
                                            <p className="text-gray-500 text-xs">Please allow camera access in your browser settings, then try again.</p>
                                        </div>
                                        <button
                                            onClick={startCamera}
                                            className="px-5 py-2 border border-red-500/40 text-red-400 rounded-lg text-sm hover:bg-red-500/10 transition-colors"
                                        >
                                            Try Again
                                        </button>
                                    </>
                                )}

                                {camStatus === 'error' && (
                                    <>
                                        <div className="w-20 h-20 rounded-full bg-yellow-500/10 border border-yellow-500/30 flex items-center justify-center">
                                            <AlertTriangle size={36} className="text-yellow-400" />
                                        </div>
                                        <div>
                                            <p className="text-yellow-400 font-bold mb-1">Camera Error</p>
                                            <p className="text-gray-500 text-xs">Could not access camera. Check if another app is using it.</p>
                                        </div>
                                        <button
                                            onClick={startCamera}
                                            className="px-5 py-2 border border-yellow-500/40 text-yellow-400 rounded-lg text-sm hover:bg-yellow-500/10 transition-colors"
                                        >
                                            Retry
                                        </button>
                                    </>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Bottom controls */}
                    <div className="p-3 bg-gradient-to-t from-black/80 to-transparent flex gap-3 items-center z-10">
                        {camStatus === 'active' ? (
                            <>
                                <motion.button
                                    whileTap={{ scale: 0.95 }}
                                    onClick={toggleScanning}
                                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-bold transition-all ${scanning
                                            ? 'bg-red-500/20 border border-red-500/40 text-red-400 hover:bg-red-500/30'
                                            : 'bg-neon-blue/20 border border-neon-blue/40 text-neon-blue hover:bg-neon-blue/30'
                                        }`}
                                >
                                    {scanning ? <><Eye size={14} /> Pause Scan</> : <><Eye size={14} /> Resume Scan</>}
                                </motion.button>

                                <motion.button
                                    whileTap={{ scale: 0.95 }}
                                    onClick={stopCamera}
                                    className="flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-bold bg-white/5 border border-white/10 text-gray-400 hover:text-white hover:bg-white/10 transition-all"
                                >
                                    <CameraOff size={14} />
                                    Stop Camera
                                </motion.button>

                                <div className="flex-1" />
                                {scanning && (
                                    <motion.div
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        className="text-xs text-gray-500 font-mono"
                                    >
                                        Sending frame every ~{FRAME_INTERVAL_MS}ms
                                    </motion.div>
                                )}
                            </>
                        ) : (
                            <div className="text-xs text-gray-600 font-mono">Camera offline</div>
                        )}
                    </div>
                </div>

                {/* ── Verdict Signal Card ────────────────────────────────────────── */}
                <AnimatePresence mode="wait">
                    {vc && camStatus === 'active' ? (
                        <motion.div
                            key={verdict}
                            initial={{ opacity: 0, y: 10, scale: 0.97 }}
                            animate={{ opacity: 1, y: 0, scale: 1 }}
                            exit={{ opacity: 0, y: -8, scale: 0.97 }}
                            transition={{ duration: 0.25 }}
                            style={{ background: vc.bg, borderColor: vc.border }}
                            className="rounded-xl border p-4 flex items-center gap-4"
                        >
                            <div
                                style={{ background: `${vc.color}22`, borderColor: `${vc.color}55` }}
                                className="w-12 h-12 rounded-full border flex items-center justify-center shrink-0"
                            >
                                {VerdictIcon && <VerdictIcon size={22} style={{ color: vc.color }} />}
                            </div>
                            <div className="flex-1">
                                <div className="flex items-baseline gap-3">
                                    <span className="font-black text-xl tracking-wide" style={{ color: vc.color }}>
                                        {vc.label}
                                    </span>
                                    {verdict !== 'NO_FACE' && verdict !== 'ERROR' && (
                                        <span className="text-sm font-mono text-gray-400">
                                            {Math.round(confidence * 100)}% confidence
                                        </span>
                                    )}
                                </div>
                                <div className="mt-1 h-1.5 bg-black/30 rounded-full overflow-hidden">
                                    <motion.div
                                        className="h-full rounded-full"
                                        style={{ background: vc.color }}
                                        animate={{ width: `${verdict !== 'NO_FACE' && verdict !== 'ERROR' ? Math.round(confidence * 100) : 0}%` }}
                                        transition={{ duration: 0.4 }}
                                    />
                                </div>
                            </div>
                            {verdict === 'FAKE' && (
                                <motion.div
                                    animate={{ scale: [1, 1.2, 1] }}
                                    transition={{ repeat: Infinity, duration: 0.8 }}
                                    className="w-3 h-3 rounded-full bg-neon-red shrink-0"
                                />
                            )}
                        </motion.div>
                    ) : camStatus === 'active' && scanning ? (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="rounded-xl border border-white/10 bg-white/5 p-4 flex items-center gap-3 text-gray-500 text-sm"
                        >
                            <motion.div
                                animate={{ rotate: 360 }}
                                transition={{ repeat: Infinity, duration: 1.5, ease: 'linear' }}
                                className="w-5 h-5 rounded-full border-2 border-neon-blue border-t-transparent"
                            />
                            Analysing frame…
                        </motion.div>
                    ) : null}
                </AnimatePresence>
            </div>

            {/* ─── Right: Event Log ────────────────────────────────────────────── */}
            <div className="bg-cyber-gray border border-white/10 rounded-xl flex flex-col h-full overflow-hidden">
                <div className="p-4 border-b border-white/10 flex justify-between items-center shrink-0">
                    <h3 className="font-bold text-white flex items-center gap-2 text-sm">
                        <Activity className="text-neon-blue" size={16} />
                        Live Event Stream
                    </h3>
                    <span className="text-xs text-gray-600 font-mono">{logs.length} events</span>
                </div>

                <div className="flex-1 overflow-y-auto p-2 space-y-1.5 font-mono text-xs">
                    <AnimatePresence initial={false}>
                        {logs.length === 0 ? (
                            <div className="text-center text-gray-600 pt-12 text-xs">
                                Waiting for events…
                            </div>
                        ) : (
                            logs.map((log) => (
                                <motion.div
                                    key={log.id}
                                    initial={{ opacity: 0, x: 16 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0 }}
                                    className={`p-2.5 rounded border-l-2 ${log.type === 'ALERT'
                                            ? 'bg-red-500/10 border-red-500 text-red-200'
                                            : log.type === 'WARNING'
                                                ? 'bg-yellow-500/10 border-yellow-500 text-yellow-200'
                                                : 'bg-white/5 border-neon-blue/60 text-gray-300'
                                        }`}
                                >
                                    <div className="flex justify-between items-center opacity-60 mb-0.5">
                                        <span>[{log.time}]</span>
                                        <span className="text-[10px]">{log.source}</span>
                                    </div>
                                    <div className="leading-snug">
                                        {log.type === 'ALERT' && <AlertTriangle size={11} className="inline mr-1 shrink-0" />}
                                        {log.message}
                                    </div>
                                </motion.div>
                            ))
                        )}
                    </AnimatePresence>
                </div>
            </div>
        </div>
    );
};

export default MonitorPage;
