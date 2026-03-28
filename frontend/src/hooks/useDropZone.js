/**
 * Internal trace:
 * - Wrong before: file selection logic lived inside pages, lacked inline validation, and did not build stable previews for image, video, and audio uploads.
 * - Fixed now: drag-and-drop, browse, validation, preview generation, and clear/reset behavior are centralized in one hook.
 */

import { useEffect, useState } from 'react';

const SUPPORTED = {
  'image/jpeg': { kind: 'image', maxBytes: 20 * 1024 * 1024 },
  'image/png': { kind: 'image', maxBytes: 20 * 1024 * 1024 },
  'image/webp': { kind: 'image', maxBytes: 20 * 1024 * 1024 },
  'video/mp4': { kind: 'video', maxBytes: 100 * 1024 * 1024 },
  'video/webm': { kind: 'video', maxBytes: 100 * 1024 * 1024 },
  'audio/wav': { kind: 'audio', maxBytes: 20 * 1024 * 1024 },
  'audio/x-wav': { kind: 'audio', maxBytes: 20 * 1024 * 1024 },
  'audio/mpeg': { kind: 'audio', maxBytes: 20 * 1024 * 1024 },
  'audio/ogg': { kind: 'audio', maxBytes: 20 * 1024 * 1024 },
};

async function buildWaveform(file) {
  try {
    const AudioContextImpl = window.AudioContext || window.webkitAudioContext;
    const context = new AudioContextImpl();
    const buffer = await context.decodeAudioData(await file.arrayBuffer());
    const data = buffer.getChannelData(0);
    const chunks = Math.min(96, Math.max(24, Math.floor(data.length / 4000)));
    const samples = [];
    const step = Math.floor(data.length / chunks);
    for (let index = 0; index < chunks; index += 1) {
      const slice = data.slice(index * step, (index + 1) * step);
      const average = slice.reduce((sum, value) => sum + Math.abs(value), 0) / Math.max(slice.length, 1);
      samples.push(Number(average.toFixed(4)));
    }
    await context.close();
    return samples;
  } catch {
    return [];
  }
}

function buildVideoThumbnail(file, url) {
  return new Promise((resolve) => {
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.src = url;
    video.muted = true;
    video.playsInline = true;
    video.currentTime = 0.1;
    video.onloadeddata = () => {
      const canvas = document.createElement('canvas');
      canvas.width = 320;
      canvas.height = 180;
      const context = canvas.getContext('2d');
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      resolve(canvas.toDataURL('image/jpeg', 0.82));
    };
    video.onerror = () => resolve('');
  });
}

export function useDropZone() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState('');
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => () => {
    if (preview?.url) URL.revokeObjectURL(preview.url);
  }, [preview]);

  const clear = () => {
    if (preview?.url) URL.revokeObjectURL(preview.url);
    setFile(null);
    setPreview(null);
    setError('');
    setIsDragging(false);
  };

  const selectFile = async (nextFile) => {
    if (!nextFile) return;

    const config = SUPPORTED[nextFile.type];
    if (!config) {
      setError('Unsupported file type. Use JPEG, PNG, WEBP, MP4, WEBM, WAV, MP3, or OGG.');
      return;
    }

    if (nextFile.size > config.maxBytes) {
      setError(`File is too large for ${config.kind} analysis.`);
      return;
    }

    if (preview?.url) URL.revokeObjectURL(preview.url);

    const url = URL.createObjectURL(nextFile);
    const nextPreview = { kind: config.kind, url, thumbnailUrl: '', waveform: [] };
    if (config.kind === 'video') {
      nextPreview.thumbnailUrl = await buildVideoThumbnail(nextFile, url);
    }
    if (config.kind === 'audio') {
      nextPreview.waveform = await buildWaveform(nextFile);
    }

    setFile(nextFile);
    setPreview(nextPreview);
    setError('');
  };

  const onDragOver = (event) => {
    event.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = (event) => {
    event.preventDefault();
    setIsDragging(false);
  };

  const onDrop = async (event) => {
    event.preventDefault();
    setIsDragging(false);
    const dropped = event.dataTransfer.files?.[0];
    await selectFile(dropped);
  };

  return {
    file,
    preview,
    error,
    isDragging,
    clear,
    onDragLeave,
    onDragOver,
    onDrop,
    selectFile,
    setError,
  };
}
