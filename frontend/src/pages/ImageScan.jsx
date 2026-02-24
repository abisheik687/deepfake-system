import { useState } from 'react';
import { motion } from 'framer-motion';
import { Image as ImageIcon, Upload, AlertCircle } from 'lucide-react';
import { unifiedAPI } from '../services/api';

const ImageScan = () => {
    const [file, setFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState('');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);

    const handleFileChange = (e) => {
        const selected = e.target.files[0];
        if (selected && selected.type.startsWith('image/')) {
            setFile(selected);
            setPreviewUrl(URL.createObjectURL(selected));
            setResult(null);
        }
    };

    const handleScan = async () => {
        if (!file) return;
        setLoading(true);
        try {
            // Convert file to base64
            const reader = new FileReader();
            reader.onloadend = async () => {
                const base64data = reader.result;
                try {
                    // Unified orchestrator call
                    const res = await unifiedAPI.analyzeImage({
                        data: base64data,
                        source: 'web'
                    });
                    setResult(res);
                } catch (error) {
                    console.error('Scan failed:', error);
                    setResult({ error: 'Failed to process image through orchestrator.' });
                } finally {
                    setLoading(false);
                }
            };
            reader.readAsDataURL(file);
        } catch (err) {
            console.error(err);
            setLoading(false);
        }
    };

    return (
        <div className="space-y-6 max-w-4xl mx-auto">
            <div>
                <h1 className="text-3xl font-bold text-white flex items-center gap-3">
                    <ImageIcon className="text-neon-blue" /> Image Analysis
                </h1>
                <p className="text-gray-400 mt-2">
                    Scan images through the unified model orchestrator to detect manipulation.
                </p>
            </div>

            <div className="bg-cyber-gray border border-white/5 rounded-xl p-6">
                {!previewUrl ? (
                    <div className="border-2 border-dashed border-white/10 rounded-xl p-12 text-center hover:border-neon-blue/50 transition-colors">
                        <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                        <h3 className="text-xl font-bold text-white mb-2">Upload Image to Scan</h3>
                        <p className="text-sm text-gray-400 mb-6">Supports JPG, PNG (Max 10MB)</p>
                        <input
                            type="file"
                            id="image-upload"
                            className="hidden"
                            accept="image/*"
                            onChange={handleFileChange}
                        />
                        <label
                            htmlFor="image-upload"
                            className="px-6 py-3 bg-white/5 hover:bg-white/10 text-white rounded-lg cursor-pointer transition-colors"
                        >
                            Browse Files
                        </label>
                    </div>
                ) : (
                    <div className="space-y-6">
                        <div className="relative rounded-xl overflow-hidden border border-white/10 bg-black/50 aspect-video flex-center flex items-center justify-center h-64">
                            <img src={previewUrl} alt="Preview" className="max-h-full object-contain" />
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
                                className="px-6 py-3 bg-neon-blue hover:bg-neon-blue/90 text-black font-bold rounded-lg transition-colors flex-1 disabled:opacity-50"
                            >
                                {loading ? 'Orchestrating Models...' : 'Run Analysis'}
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
                    <div className="flex items-start justify-between">
                        <div>
                            <h3 className="text-xl font-bold text-white mb-1">
                                System Verdict: <span className={
                                    result.verdict === 'FAKE' ? 'text-red-400' :
                                        result.verdict === 'SUSPICIOUS' ? 'text-yellow-400' :
                                            'text-green-400'
                                }>{result.verdict}</span>
                            </h3>
                            <p className="text-gray-300">Confidence: {(result.confidence * 100).toFixed(1)}%</p>
                            <p className="text-sm text-gray-500 mt-2">
                                Models Used: {result.models_used} | Latency: {result.latency_ms}ms | Agreement: {(result.agreement_score * 100).toFixed(1)}%
                            </p>
                        </div>
                        <div className="text-right">
                            <div className="text-sm text-gray-400 mb-1">Risk Score</div>
                            <div className="text-4xl font-black text-white">{result.risk_score}</div>
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

export default ImageScan;
