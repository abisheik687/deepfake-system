
import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileVideo, AlertCircle, CheckCircle2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useChunkedUpload } from '../hooks/useChunkedUpload';

const UploadPage = () => {
    const { uploadFile, progress, isUploading, error, result } = useChunkedUpload();

    const onDrop = useCallback((acceptedFiles) => {
        if (acceptedFiles?.length > 0) {
            uploadFile(acceptedFiles[0]);
        }
    }, [uploadFile]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'video/*': ['.mp4', '.avi', '.mov', '.mkv']
        },
        maxFiles: 1,
        multiple: false
    });

    return (
        <div className="max-w-4xl mx-auto space-y-8">
            <div>
                <h1 className="text-3xl font-bold text-white mb-2">New Forensic Scan</h1>
                <p className="text-gray-400">Upload video evidence for deepfake analysis. Supports MP4, AVI, MOV.</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Upload Zone */}
                <div className="lg:col-span-2">
                    {!isUploading && !result ? (
                        <div
                            {...getRootProps()}
                            className={`border-2 border-dashed rounded-2xl h-[400px] flex flex-col items-center justify-center cursor-pointer transition-all duration-300 ${isDragActive
                                ? 'border-neon-blue bg-neon-blue/5'
                                : 'border-white/10 hover:border-white/30 hover:bg-white/5'
                                }`}
                        >
                            <input {...getInputProps()} />
                            <div className="w-20 h-20 bg-cyber-black rounded-full flex items-center justify-center mb-6 shadow-lg border border-white/10">
                                <Upload className={isDragActive ? 'text-neon-blue' : 'text-gray-400'} size={32} />
                            </div>
                            <h3 className="text-xl font-bold text-white mb-2">
                                {isDragActive ? 'Drop video here' : 'Drag & Drop Video Evidence'}
                            </h3>
                            <p className="text-sm text-gray-500 max-w-xs text-center">
                                or click to browse files (Max 2GB)
                            </p>
                        </div>
                    ) : (
                        <div className="border border-white/10 bg-cyber-gray rounded-2xl h-[400px] flex flex-col items-center justify-center p-8">
                            {isUploading ? (
                                <div className="w-full max-w-md text-center">
                                    <div className="mb-4 flex justify-center">
                                        <motion.div
                                            animate={{ rotate: 360 }}
                                            transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                                        >
                                            <Upload className="text-neon-blue" size={48} />
                                        </motion.div>
                                    </div>
                                    <h3 className="text-xl font-bold text-white mb-2">Uploading Evidence...</h3>
                                    <p className="text-gray-400 mb-6">{progress}% Complete</p>
                                    <div className="h-2 bg-black rounded-full overflow-hidden">
                                        <motion.div
                                            className="h-full bg-neon-blue"
                                            initial={{ width: 0 }}
                                            animate={{ width: `${progress}%` }}
                                        />
                                    </div>
                                </div>
                            ) : result ? (
                                <div className="text-center">
                                    <motion.div
                                        initial={{ scale: 0 }}
                                        animate={{ scale: 1 }}
                                        className="w-20 h-20 bg-green-500/10 rounded-full flex items-center justify-center mx-auto mb-6 text-green-500"
                                    >
                                        <CheckCircle2 size={40} />
                                    </motion.div>
                                    <h3 className="text-2xl font-bold text-white mb-2">Upload Successful</h3>
                                    <p className="text-gray-400 mb-8">Evidence ID: #{result.detectionId}</p>
                                    <button
                                        onClick={() => window.location.href = `/analysis/${result.detectionId}`}
                                        className="px-6 py-2 bg-neon-blue text-black font-bold rounded-lg hover:bg-white transition-colors"
                                    >
                                        View Full Analysis
                                    </button>
                                </div>
                            ) : null}
                        </div>
                    )}
                </div>

                {/* Info Sidebar */}
                <div className="space-y-6">
                    <div className="bg-cyber-gray border border-white/5 rounded-xl p-6">
                        <h3 className="font-bold text-white mb-4 flex items-center gap-2">
                            <FileVideo size={18} className="text-neon-blue" />
                            Supported Formats
                        </h3>
                        <ul className="space-y-3 text-sm text-gray-400">
                            <li className="flex items-center gap-2">
                                <div className="w-1.5 h-1.5 rounded-full bg-neon-blue"></div>
                                MP4 (H.264/HEVC)
                            </li>
                            <li className="flex items-center gap-2">
                                <div className="w-1.5 h-1.5 rounded-full bg-neon-blue"></div>
                                AVI (Uncompressed)
                            </li>
                            <li className="flex items-center gap-2">
                                <div className="w-1.5 h-1.5 rounded-full bg-neon-blue"></div>
                                MOV (QuickTime)
                            </li>
                        </ul>
                    </div>

                    {error && (
                        <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 flex items-start gap-3">
                            <AlertCircle className="text-red-500 shrink-0" size={20} />
                            <div>
                                <h4 className="font-bold text-red-500 text-sm">Upload Error</h4>
                                <p className="text-xs text-red-400 mt-1">{error}</p>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default UploadPage;
