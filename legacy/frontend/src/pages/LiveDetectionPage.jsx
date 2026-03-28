import { useState, useEffect, useRef } from 'react';
import { Video, Mic, Users, Square, Circle, Download, AlertTriangle } from 'lucide-react';
import { motion } from 'framer-motion';
import { liveVideoAPI, liveAudioAPI, interviewAPI } from '../services/api';

const MODES = {
    VIDEO: { id: 'video', label: 'Video Call', icon: Video, wsPath: '/api/live-video/ws/video' },
    AUDIO: { id: 'audio', label: 'Voice Call', icon: Mic, wsPath: '/api/live-audio/ws/audio' },
    INTERVIEW: { id: 'interview', label: 'Interview Proctoring', icon: Users, wsPath: '/api/interview/ws/interview' }
};

function WSIndicator({ status }) {
    const colors = { connected: '#2DC653', reconnecting: '#F4A261', disconnected: '#6B7E94' };
    return (
        <div className='flex items-center gap-1.5'>
            <div className={`w-2 h-2 rounded-full ${status === 'connected' ? 'animate-pulse' : ''}`}
                style={{ background: colors[status] ?? colors.disconnected }} />
            <span className='text-xs uppercase tracking-wider' style={{ color: 'var(--text-muted)' }}>
                {status}
            </span>
        </div>
    );
}

function ConfidenceMeter({ score, label }) {
    const isReal = score < 0.5;
    const percentage = Math.round(score * 100);
    const color = isReal ? '#2DC653' : '#E63946';
    
    return (
        <div className='space-y-2'>
            <div className='flex items-center justify-between'>
                <span className='text-xs font-semibold uppercase tracking-wider' style={{ color: 'var(--text-secondary)' }}>
                    {label}
                </span>
                <span className='text-lg font-bold' style={{ color }}>
                    {percentage}%
                </span>
            </div>
            <div className='h-2 rounded-full overflow-hidden' style={{ background: 'var(--bg-card)' }}>
                <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${percentage}%` }}
                    transition={{ duration: 0.3 }}
                    className='h-full'
                    style={{ background: color }}
                />
            </div>
        </div>
    );
}

export default function LiveDetectionPage() {
    const [mode, setMode] = useState('VIDEO');
    const [isRecording, setIsRecording] = useState(false);
    const [wsStatus, setWsStatus] = useState('disconnected');
    const [sessionId, setSessionId] = useState(null);
    const [confidence, setConfidence] = useState({ video: 0, audio: 0, overall: 0 });
    const [alerts, setAlerts] = useState([]);
    const [transcript, setTranscript] = useState([]);
    
    const wsRef = useRef(null);
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const mediaStreamRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const frameIntervalRef = useRef(null);
    const audioChunkIntervalRef = useRef(null);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            stopRecording();
        };
    }, []);

    const startRecording = async () => {
        try {
            const constraints = mode === 'VIDEO' || mode === 'INTERVIEW'
                ? { video: { width: 640, height: 480 }, audio: mode === 'INTERVIEW' }
                : { audio: true };

            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            mediaStreamRef.current = stream;

            if (mode === 'VIDEO' || mode === 'INTERVIEW') {
                videoRef.current.srcObject = stream;
                await videoRef.current.play();
            }

            // Connect WebSocket
            const wsUrl = `ws://localhost:8000${MODES[mode].wsPath}`;
            const ws = new WebSocket(wsUrl);
            wsRef.current = ws;

            ws.onopen = () => {
                setWsStatus('connected');
                setIsRecording(true);
                
                if (mode === 'VIDEO' || mode === 'INTERVIEW') {
                    startFrameCapture();
                }
                
                if (mode === 'AUDIO' || mode === 'INTERVIEW') {
                    startAudioCapture();
                }
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'session_started') {
                    setSessionId(data.session_id);
                } else if (data.type === 'frame_result' || data.type === 'chunk_result') {
                    updateConfidence(data);
                } else if (data.type === 'alert') {
                    addAlert(data);
                } else if (data.type === 'transcript') {
                    addTranscript(data);
                } else if (data.type === 'integrity_update') {
                    setConfidence(prev => ({
                        ...prev,
                        video: data.video_score || prev.video,
                        audio: data.audio_score || prev.audio,
                        overall: data.integrity_score || prev.overall
                    }));
                }
            };

            ws.onerror = () => setWsStatus('reconnecting');
            ws.onclose = () => {
                setWsStatus('disconnected');
                setIsRecording(false);
            };

        } catch (error) {
            console.error('Failed to start recording:', error);
            alert('Failed to access camera/microphone. Please grant permissions.');
        }
    };

    const stopRecording = () => {
        // Stop frame capture
        if (frameIntervalRef.current) {
            clearInterval(frameIntervalRef.current);
            frameIntervalRef.current = null;
        }

        // Stop audio capture
        if (audioChunkIntervalRef.current) {
            clearInterval(audioChunkIntervalRef.current);
            audioChunkIntervalRef.current = null;
        }

        // Stop media recorder
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop();
        }

        // Stop media stream
        if (mediaStreamRef.current) {
            mediaStreamRef.current.getTracks().forEach(track => track.stop());
            mediaStreamRef.current = null;
        }

        // Close WebSocket
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }

        setIsRecording(false);
        setWsStatus('disconnected');
    };

    const startFrameCapture = () => {
        frameIntervalRef.current = setInterval(() => {
            if (!videoRef.current || !canvasRef.current || !wsRef.current) return;

            const canvas = canvasRef.current;
            const video = videoRef.current;
            const ctx = canvas.getContext('2d');

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0);

            canvas.toBlob((blob) => {
                if (blob && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                    const reader = new FileReader();
                    reader.onloadend = () => {
                        const base64 = reader.result.split(',')[1];
                        wsRef.current.send(JSON.stringify({
                            type: 'frame',
                            data: base64,
                            timestamp: Date.now()
                        }));
                    };
                    reader.readAsDataURL(blob);
                }
            }, 'image/jpeg', 0.8);
        }, 2000); // Every 2 seconds
    };

    const startAudioCapture = () => {
        const audioStream = mediaStreamRef.current;
        if (!audioStream) return;

        const mediaRecorder = new MediaRecorder(audioStream, {
            mimeType: 'audio/webm;codecs=opus'
        });
        mediaRecorderRef.current = mediaRecorder;

        const audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = () => {
            if (audioChunks.length > 0 && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                const reader = new FileReader();
                reader.onloadend = () => {
                    const base64 = reader.result.split(',')[1];
                    wsRef.current.send(JSON.stringify({
                        type: 'audio_chunk',
                        data: base64,
                        timestamp: Date.now()
                    }));
                };
                reader.readAsDataURL(audioBlob);
                audioChunks.length = 0;
            }
        };

        // Record in 2-second chunks
        audioChunkIntervalRef.current = setInterval(() => {
            if (mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
            }
            if (mediaRecorder.state === 'inactive') {
                mediaRecorder.start();
            }
        }, 2000);

        mediaRecorder.start();
    };

    const updateConfidence = (data) => {
        if (mode === 'VIDEO') {
            setConfidence(prev => ({ ...prev, video: data.confidence || 0 }));
        } else if (mode === 'AUDIO') {
            setConfidence(prev => ({ ...prev, audio: data.confidence || 0 }));
        }
    };

    const addAlert = (alert) => {
        setAlerts(prev => [{ ...alert, id: Date.now() }, ...prev.slice(0, 19)]);
    };

    const addTranscript = (item) => {
        setTranscript(prev => [...prev, { ...item, id: Date.now() }]);
    };

    const downloadReport = async () => {
        if (!sessionId) return;
        
        try {
            let response;
            if (mode === 'VIDEO') {
                response = await liveVideoAPI.exportSession(sessionId);
            } else if (mode === 'AUDIO') {
                response = await liveAudioAPI.exportSession(sessionId);
            } else if (mode === 'INTERVIEW') {
                response = await interviewAPI.generateReport(sessionId);
            }
            
            if (response.report_path) {
                window.open(`http://localhost:8000${response.report_path}`, '_blank');
            }
        } catch (error) {
            console.error('Failed to download report:', error);
        }
    };

    const borderColor = isRecording 
        ? (confidence.overall > 0.5 || confidence.video > 0.5 || confidence.audio > 0.5 ? '#E63946' : '#2DC653')
        : 'var(--border)';

    return (
        <div className='p-6 max-w-7xl mx-auto'>
            {/* Header */}
            <div className='flex items-center justify-between mb-6'>
                <h1 className='text-2xl font-bold' style={{ color: 'var(--text-primary)' }}>
                    Live Detection
                </h1>
                <WSIndicator status={wsStatus} />
            </div>

            {/* Mode Selector */}
            <div className='flex gap-3 mb-6'>
                {Object.entries(MODES).map(([key, m]) => {
                    const Icon = m.icon;
                    const isActive = mode === key;
                    return (
                        <button
                            key={key}
                            onClick={() => !isRecording && setMode(key)}
                            disabled={isRecording}
                            className='flex items-center gap-2 px-4 py-2.5 rounded-lg font-semibold text-sm transition-all disabled:opacity-50'
                            style={{
                                background: isActive ? 'var(--cyan)' : 'var(--bg-card)',
                                color: isActive ? '#0A1628' : 'var(--text-secondary)',
                                border: `1px solid ${isActive ? 'var(--cyan)' : 'var(--border)'}`
                            }}
                        >
                            <Icon size={18} />
                            {m.label}
                        </button>
                    );
                })}
            </div>

            <div className='grid grid-cols-1 lg:grid-cols-3 gap-6'>
                {/* Video/Audio Feed */}
                <div className='lg:col-span-2 space-y-4'>
                    <div className='rounded-xl overflow-hidden relative' 
                        style={{ background: 'var(--bg-card)', border: `2px solid ${borderColor}` }}>
                        
                        {(mode === 'VIDEO' || mode === 'INTERVIEW') && (
                            <video
                                ref={videoRef}
                                className='w-full aspect-video bg-black'
                                muted
                                playsInline
                            />
                        )}
                        
                        {mode === 'AUDIO' && (
                            <div className='w-full aspect-video bg-black flex items-center justify-center'>
                                <div className='text-center'>
                                    <Mic size={64} className='mx-auto mb-4' style={{ color: 'var(--cyan)' }} />
                                    <p className='text-lg font-semibold' style={{ color: 'var(--text-primary)' }}>
                                        Audio Stream Active
                                    </p>
                                    {isRecording && (
                                        <div className='mt-4 flex items-center justify-center gap-2'>
                                            <div className='w-3 h-3 bg-red-500 rounded-full animate-pulse' />
                                            <span className='text-sm' style={{ color: 'var(--text-muted)' }}>Recording</span>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}

                        {/* Recording indicator overlay */}
                        {isRecording && (mode === 'VIDEO' || mode === 'INTERVIEW') && (
                            <div className='absolute top-4 left-4 flex items-center gap-2 px-3 py-1.5 rounded-lg bg-black/70 backdrop-blur-sm'>
                                <Circle size={12} className='text-red-500 fill-red-500 animate-pulse' />
                                <span className='text-xs font-semibold text-white'>LIVE</span>
                            </div>
                        )}

                        {/* Confidence overlay */}
                        {isRecording && (
                            <div className='absolute bottom-4 left-4 right-4 p-4 rounded-lg bg-black/70 backdrop-blur-sm'>
                                {mode === 'INTERVIEW' ? (
                                    <>
                                        <ConfidenceMeter score={confidence.video} label='Video Authenticity' />
                                        <div className='my-2' />
                                        <ConfidenceMeter score={confidence.audio} label='Audio Authenticity' />
                                        <div className='my-2' />
                                        <ConfidenceMeter score={confidence.overall} label='Overall Integrity' />
                                    </>
                                ) : mode === 'VIDEO' ? (
                                    <ConfidenceMeter score={confidence.video} label='Authenticity Score' />
                                ) : (
                                    <ConfidenceMeter score={confidence.audio} label='Authenticity Score' />
                                )}
                            </div>
                        )}
                    </div>

                    {/* Controls */}
                    <div className='flex gap-3'>
                        {!isRecording ? (
                            <button
                                onClick={startRecording}
                                className='flex-1 flex items-center justify-center gap-2 px-6 py-3 rounded-lg font-bold text-sm transition-all'
                                style={{ background: 'var(--cyan)', color: '#0A1628' }}
                            >
                                <Circle size={18} />
                                Start Recording
                            </button>
                        ) : (
                            <button
                                onClick={stopRecording}
                                className='flex-1 flex items-center justify-center gap-2 px-6 py-3 rounded-lg font-bold text-sm transition-all'
                                style={{ background: '#E63946', color: 'white' }}
                            >
                                <Square size={18} />
                                Stop Recording
                            </button>
                        )}
                        
                        <button
                            onClick={downloadReport}
                            disabled={!sessionId}
                            className='flex items-center gap-2 px-6 py-3 rounded-lg font-bold text-sm transition-all disabled:opacity-50'
                            style={{ background: 'var(--bg-card)', color: 'var(--text-primary)', border: '1px solid var(--border)' }}
                        >
                            <Download size={18} />
                            Export Report
                        </button>
                    </div>

                    {/* Hidden canvas for frame capture */}
                    <canvas ref={canvasRef} className='hidden' />
                </div>

                {/* Sidebar: Alerts & Transcript */}
                <div className='space-y-4'>
                    {/* Alerts */}
                    <div className='rounded-xl overflow-hidden' style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                        <div className='px-4 py-3 flex items-center gap-2' style={{ borderBottom: '1px solid var(--border)' }}>
                            <AlertTriangle size={16} style={{ color: 'var(--cyan)' }} />
                            <h3 className='font-semibold text-sm uppercase tracking-wider' style={{ color: 'var(--text-secondary)' }}>
                                Alerts
                            </h3>
                        </div>
                        <div className='p-4 space-y-2 max-h-64 overflow-y-auto'>
                            {alerts.length === 0 ? (
                                <p className='text-xs text-center py-4' style={{ color: 'var(--text-muted)' }}>
                                    No alerts yet
                                </p>
                            ) : (
                                alerts.map(alert => (
                                    <div key={alert.id} className='p-3 rounded-lg' style={{ background: 'var(--bg-hover)' }}>
                                        <div className='flex items-start justify-between gap-2'>
                                            <span className='text-xs font-semibold' style={{ color: '#E63946' }}>
                                                {alert.message || 'Suspicious activity detected'}
                                            </span>
                                            <span className='text-xs shrink-0' style={{ color: 'var(--text-muted)' }}>
                                                {new Date(alert.timestamp).toLocaleTimeString()}
                                            </span>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>

                    {/* Transcript (for audio/interview modes) */}
                    {(mode === 'AUDIO' || mode === 'INTERVIEW') && (
                        <div className='rounded-xl overflow-hidden' style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                            <div className='px-4 py-3' style={{ borderBottom: '1px solid var(--border)' }}>
                                <h3 className='font-semibold text-sm uppercase tracking-wider' style={{ color: 'var(--text-secondary)' }}>
                                    Transcript
                                </h3>
                            </div>
                            <div className='p-4 space-y-2 max-h-64 overflow-y-auto'>
                                {transcript.length === 0 ? (
                                    <p className='text-xs text-center py-4' style={{ color: 'var(--text-muted)' }}>
                                        No transcript yet
                                    </p>
                                ) : (
                                    transcript.map(item => (
                                        <div key={item.id} className='p-3 rounded-lg' style={{ background: 'var(--bg-hover)' }}>
                                            <div className='flex items-start justify-between gap-2 mb-1'>
                                                <span className='text-xs font-semibold' style={{ 
                                                    color: item.is_fake ? '#E63946' : '#2DC653' 
                                                }}>
                                                    {item.is_fake ? 'FAKE' : 'REAL'}
                                                </span>
                                                <span className='text-xs' style={{ color: 'var(--text-muted)' }}>
                                                    {new Date(item.timestamp).toLocaleTimeString()}
                                                </span>
                                            </div>
                                            <p className='text-xs' style={{ color: 'var(--text-primary)' }}>
                                                {item.text || '[Audio chunk processed]'}
                                            </p>
                                        </div>
                                    ))
                                )}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

// Made with Bob
