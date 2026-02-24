import { useState } from 'react';
import { motion } from 'framer-motion';
import { Video, Upload, AlertCircle, PlayCircle } from 'lucide-react';
import { unifiedAPI } from '../services/api';

const VideoScan = () => {
    const [file, setFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState('');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);

    const handleFileChange = (e) => {
        const selected = e.target.files[0];
        if (selected && selected.type.startsWith('video/')) {
            setFile(selected);
            setPreviewUrl(URL.createObjectURL(selected));
            setResult(null);
        }
    };

    const handleScan = async () => {
        if (!file) return;
        setLoading(true);

        // Create form data for video upload
        const formData = new FormData();
        formData.append('file', file);
        formData.append('tier', 'balanced');
        formData.append('sample_fps', 2.0);
        formData.append('max_frames', 20);

        try {
            // Unified orchestrator call for video
            const res = await unifiedAPI.analyzeVideo(formData);
            setResult(res);
        } catch (error) {
            console.error('Scan failed:', error);
            setResult({ error: 'Failed to process video through orchestrator.' });
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="space-y-6 max-w-4xl mx-auto">
            <div>
                <h1 className="text-3xl font-bold text-white flex items-center gap-3">
                    <Video className="text-neon-purple" /> Video Orchestration
                </h1>
                <p className="text-gray-400 mt-2">
                    Temporal frame aggregation analyzes video streams across multiple models to find fleeting anomalies.
                </p>
            </div>

            <div className="bg-cyber-gray border border-white/5 rounded-xl p-6">
                {!previewUrl ? (
                    <div className="border-2 border-dashed border-white/10 rounded-xl p-12 text-center hover:border-neon-purple/50 transition-colors cursor-pointer">
                        <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                        <h3 className="text-xl font-bold text-white mb-2">Upload Video to Scan</h3>
                        <p className="text-sm text-gray-400 mb-6">Supports MP4, AVI, MOV (Max 50MB)</p>
                        <input
                            type="file"
                            id="video-upload"
                            className="hidden"
                            accept="video/*"
                            onChange={handleFileChange}
                        />
                        <label
                            htmlFor="video-upload"
                            className="px-6 py-3 bg-white/5 hover:bg-white/10 text-white rounded-lg cursor-pointer transition-colors"
                        >
                            Browse Files
                        </label>
                    </div>
                ) : (
                    <div className="space-y-6">
                        <div className="relative rounded-xl overflow-hidden border border-white/10 bg-black/50 aspect-video">
                            <video
                                src={previewUrl}
                                className="w-full h-full object-contain"
                                controls
                            />
                        </div>

                        <div className="flex gap-4">
                            <button
                                onClick={() => { setFile(null); setPreviewUrl(''); setResult(null); }}
                                className="px-6 py-3 bg-transparent border border-white/10 hover:bg-white/5 text-white rounded-lg transition-colors flex-1"
                            >
                                Clear
                            </button>
                            <button
                                onClick={handleScan}
                                disabled={loading}
                                className="px-6 py-3 bg-neon-purple hover:bg-neon-purple/90 text-white font-bold rounded-lg transition-colors flex-1 disabled:opacity-50 flex items-center justify-center gap-2"
                            >
                                {loading ? (
                                    <>
                                        <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                        Processing Temporal Frames...
                                    </>
                                ) : (
                                    <>
                                        <PlayCircle /> Run Orchestration
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                )}
            </div>

            {/* Results Section */}
            {result && !result.error && (
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`border rounded-xl p-6 ${result.verdict === 'FAKE' ? 'bg-red-500/10 border-red-500/30' :
                            result.verdict === 'SUSPICIOUS' ? 'bg-yellow-500/10 border-yellow-500/30' :
                                'bg-green-500/10 border-green-500/30'
                        }`}
                >
                    <div className="flex items-start justify-between mb-6">
                        <div>
                            <h3 className="text-xl font-bold text-white mb-1">
                                Temporal Verdict: <span className={
                                    result.verdict === 'FAKE' ? 'text-red-400' :
                                        result.verdict === 'SUSPICIOUS' ? 'text-yellow-400' :
                                            'text-green-400'
                                }>{result.verdict}</span>
                            </h3>
                            <p className="text-gray-300">Overall Confidence: {(result.confidence * 100).toFixed(1)}%</p>
                        </div>
                        <div className="text-right">
                            <div className="text-sm text-gray-400 mb-1">Aggregated Risk Score</div>
                            <div className="text-4xl font-black text-white">{result.risk_score}</div>
                        </div>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6 border-t border-white/10 pt-6">
                        <div>
                            <div className="text-gray-400 text-sm mb-1">Frames Analyzed</div>
                            <div className="text-xl font-mono text-white">{result.frames_analyzed || result.models_used}</div>
                        </div>
                        <div>
                            <div className="text-gray-400 text-sm mb-1">Agreement</div>
                            <div className="text-xl font-mono text-white">{(result.agreement_score * 100).toFixed(1)}%</div>
                        </div>
                        <div>
                            <div className="text-gray-400 text-sm mb-1">Total Latency</div>
                            <div className="text-xl font-mono text-neon-purple">{result.latency_ms}ms</div>
                        </div>
                        <div>
                            <div className="text-gray-400 text-sm mb-1">Source</div>
                            <div className="text-xl font-mono text-white break-all">{result.filename || "Video upload"}</div>
                        </div>
                    </div>
                </motion.div>
            )}

            {result && result.error && (
                <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 flex items-center gap-3 text-red-400">
                    <AlertCircle />
                    <span>{result.error}</span>
                </div>
            )}
        </div>
    );
};

export default VideoScan;
