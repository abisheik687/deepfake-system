
import { useState } from 'react';
import client from '../api/client';
import { v4 as uuidv4 } from 'uuid';

const CHUNK_SIZE = 5 * 1024 * 1024; // 5MB

export const useChunkedUpload = () => {
    const [progress, setProgress] = useState(0);
    const [isUploading, setIsUploading] = useState(false);
    const [error, setError] = useState(null);
    const [result, setResult] = useState(null); // result will contain { status: 'completed', detectionId: '...' } on success

    const uploadFile = async (file) => {
        setIsUploading(true);
        setProgress(0);
        setError(null);
        setResult(null);

        const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
        const fileId = uuidv4(); // Unique ID for this upload session

        try {
            for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
                const start = chunkIndex * CHUNK_SIZE;
                const end = Math.min(start + CHUNK_SIZE, file.size);
                const chunk = file.slice(start, end);

                const formData = new FormData();
                formData.append('file', chunk);

                // Headers for chunk metadata
                const headers = {
                    // 'Content-Type': 'multipart/form-data' is usually set automatically by axios for FormData
                    'file-id': fileId, // Using header as per Day 10 API
                };

                // Query params for chunk index
                const response = await client.post('/api/upload/chunked', formData, {
                    headers,
                    params: {
                        chunk_index: chunkIndex,
                        total_chunks: totalChunks
                    }
                });

                const percentCompleted = Math.round(((chunkIndex + 1) / totalChunks) * 100);
                setProgress(percentCompleted);

                // If this is the last chunk, it will return the completed detection ID
                if (chunkIndex === totalChunks - 1 && response.data.status === 'completed') {
                    // On successful completion of the last chunk, set the result with detection ID
                    setResult({ status: 'completed', detectionId: response.data.detection_id.toString() });
                }
            }
        } catch (err) {
            console.error("Upload failed:", err);
            setError(err.response?.data?.detail || "Upload failed. Please try again.");
            setResult({ status: 'failed' }); // Indicate failure in result as well
        } finally {
            setIsUploading(false);
        }
    };

    return { uploadFile, progress, isUploading, error, result };
};
