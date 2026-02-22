
import { useState } from 'react';
import client from '../api/client';
import { v4 as uuidv4 } from 'uuid';

const CHUNK_SIZE = 5 * 1024 * 1024; // 5MB

export const useChunkedUpload = () => {
    const [progress, setProgress] = useState(0);
    const [isUploading, setIsUploading] = useState(false);
    const [error, setError] = useState(null);
    const [result, setResult] = useState(null);

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
                    'Content-Type': 'multipart/form-data',
                    'file-id': fileId, // Using header as per Day 10 API
                };

                // Query params for chunk index
                await client.post('/scan/upload/chunked', formData, {
                    headers,
                    params: {
                        chunk_index: chunkIndex,
                        total_chunks: totalChunks
                    }
                });

                const percentCompleted = Math.round(((chunkIndex + 1) / totalChunks) * 100);
                setProgress(percentCompleted);
            }

            // Upload Complete - Get Task ID or Verdict
            // In a real flow, the last chunk return might have the result or we poll
            // For Day 17, we'll confirm success
            setResult({ status: 'uploaded', fileId });

        } catch (err) {
            console.error("Upload failed:", err);
            setError(err.response?.data?.detail || "Upload failed. Please try again.");
        } finally {
            setIsUploading(false);
        }
    };

    return { uploadFile, progress, isUploading, error, result };
};
