import { useState, useRef, useEffect, useCallback } from 'react';
import { Camera, Shield, AlertTriangle } from 'lucide-react';
import { unifiedAPI } from '../services/api';

const LiveScan = () => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [isStreaming, setIsStreaming] = useState(false);
    const [scanResult, setScanResult] = useState(null);
    const [error, setError] = useState(null);

    const startCamera = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                videoRef.current.play();
                setIsStreaming(true);
                setError(null);
            }
        } catch (err) {
            setError("Camera access denied or unavailable.");
            console.error(err);
        }
    };

    const stopCamera = () => {
        if (videoRef.current && videoRef.current.srcObject) {
            const tracks = videoRef.current.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            setIsStreaming(false);
            setScanResult(null);
        }
    };

    const captureFrame = useCallback(async () => {
        if (!isStreaming || !videoRef.current || !canvasRef.current) return;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert to base64
        const base64Frame = canvas.toDataURL('image/jpeg', 0.8);

        try {
            const result = await unifiedAPI.analyzeLiveFrame({
                frame: base64Frame,
                tier: 'fast',
                source: 'webcam'
            });
            setScanResult(result);
        } catch (err) {
            console.error("Live analysis failed", err);
        }
    }, [isStreaming]);

    useEffect(() => {
        let interval;
        if (isStreaming) {
            // Sample at 2 FPS using the optimized fast tier
            interval = setInterval(captureFrame, 500);
        }
        return () => clearInterval(interval);
    }, [isStreaming, captureFrame]);

    // Cleanup on unmount
    useEffect(() => {
        return () => stopCamera();
    }, []);

    return (
        <div className="space-y-6 max-w-5xl mx-auto">
            <div>
                <h1 className="text-3xl font-bold text-white flex items-center gap-3">
                    <Camera className="text-neon-blue" /> Live Enforcement
                </h1>
                <p className="text-gray-400 mt-2">
                    Real-time deepfake analysis using optimized high-speed frequency models.
                    <span className="text-neon-blue ml-2 text-sm uppercase tracking-wider font-bold truncate">Latency &lt; 100ms</span>
                </p>
            </div>

            <div className="grid md:grid-cols-3 gap-6">
                <div className="md:col-span-2">
                    <div className="bg-cyber-gray border border-white/5 rounded-xl p-4">
                        <div className="relative rounded-lg overflow-hidden bg-black/50 aspect-video flex items-center justify-center">
                            {!isStreaming && (
                                <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                                    <Camera size={48} className="text-white/20 mb-4" />
                                    <p className="text-gray-400">Camera inactive</p>
                                </div>
                            )}

                            <video
                                ref={videoRef}
                                className={`w-full h-full object-cover ${!isStreaming && 'hidden'}`}
                                muted
                                playsInline
                            />
                            {/* Hidden canvas for extraction */}
                            <canvas ref={canvasRef} className="hidden" />

                            {/* Live HUD Overlay */}
                            {isStreaming && scanResult && (
                                <div className="absolute top-4 right-4">
                                    <div className={`px-4 py-2 rounded border backdrop-blur-md flex items-center gap-2 font-bold ${scanResult.verdict === 'FAKE' ? 'bg-red-500/20 text-red-400 border-red-500/50' :
                                            scanResult.verdict === 'SUSPICIOUS' ? 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50' :
                                                'bg-green-500/20 text-green-400 border-green-500/50'
                                        }`}>
                                        {scanResult.verdict === 'FAKE' ? <AlertTriangle size={18} /> : <Shield size={18} />}
                                        {scanResult.verdict}
                                    </div>
                                </div>
                            )}
                        </div>

                        <div className="mt-4 flex gap-4">
                            {!isStreaming ? (
                                <button
                                    onClick={startCamera}
                                    className="px-6 py-3 bg-neon-blue hover:bg-neon-blue/90 text-black font-bold rounded-lg flex-1 transition-all"
                                >
                                    Initialize Secure Stream
                                </button>
                            ) : (
                                <button
                                    onClick={stopCamera}
                                    className="px-6 py-3 bg-red-500/20 hover:bg-red-500/30 text-red-500 border border-red-500/50 font-bold rounded-lg flex-1 transition-all"
                                >
                                    Terminate Connection
                                </button>
                            )}
                        </div>
                        {error && <p className="text-red-400 mt-2 text-sm">{error}</p>}
                    </div>
                </div>

                <div className="md:col-span-1">
                    <div className="bg-cyber-gray border border-white/5 rounded-xl p-6 h-full flex flex-col">
                        <h3 className="text-xl font-bold mb-6 text-white border-b border-white/10 pb-4">Telemetry Stream</h3>

                        {scanResult ? (
                            <div className="space-y-6 flex-1">
                                <div>
                                    <div className="text-sm text-gray-400 mb-1">Live Risk Score</div>
                                    <div className={`text-5xl font-black ${scanResult.risk_score >= 65 ? 'text-red-400' :
                                            scanResult.risk_score >= 30 ? 'text-yellow-400' :
                                                'text-green-400'
                                        }`}>
                                        {scanResult.risk_score}
                                    </div>
                                </div>
                                <div className="space-y-3">
                                    <div className="flex justify-between border-b border-white/5 pb-2">
                                        <span className="text-gray-400">Confidence</span>
                                        <span className="text-white font-mono">{(scanResult.confidence * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="flex justify-between border-b border-white/5 pb-2">
                                        <span className="text-gray-400">Model Used</span>
                                        <span className="text-white font-mono">Frequency (Fast)</span>
                                    </div>
                                    <div className="flex justify-between border-b border-white/5 pb-2">
                                        <span className="text-gray-400">Latency</span>
                                        <span className="text-neon-blue font-mono">{scanResult.latency_ms}ms</span>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="flex-1 flex items-center justify-center text-gray-500 font-mono text-sm">
                                AWAITING TELEMETRY...
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default LiveScan;
