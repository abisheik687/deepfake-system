/**
 * Internal trace:
 * - Wrong before: the drop zone was dependable but still felt utilitarian and did not fully capitalize on the web app's new visual hierarchy.
 * - Fixed now: the component has a more premium empty state, clearer file chips, stronger preview presentation, and better mobile-to-desktop rhythm.
 */

import { motion } from 'framer-motion';
import { AudioLines, Film, Image as ImageIcon, RefreshCcw, Sparkles, UploadCloud, X } from 'lucide-react';

function getIcon(kind) {
  if (kind === 'video') return Film;
  if (kind === 'audio') return AudioLines;
  return ImageIcon;
}

function formatBytes(bytes) {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** exponent;
  return `${value.toFixed(exponent === 0 ? 0 : 1)} ${units[exponent]}`;
}

/**
 * @param {{
 *   dropzone: ReturnType<import('../hooks/useDropZone').useDropZone>,
 *   disabled: boolean,
 *   onBrowse: () => void,
 * }} props
 */
function DropZone({ dropzone, disabled, onBrowse }) {
  const Icon = getIcon(dropzone.preview?.kind);
  const empty = !dropzone.file;

  return (
    <div className="space-y-4">
      <motion.div
        onDragOver={disabled ? undefined : dropzone.onDragOver}
        onDragLeave={disabled ? undefined : dropzone.onDragLeave}
        onDrop={disabled ? undefined : dropzone.onDrop}
        whileHover={disabled ? undefined : { scale: 1.008 }}
        animate={
          dropzone.file
            ? { scale: [1, 1.012, 1], borderColor: ['rgba(98,244,255,0.18)', '#62f4ff', 'rgba(98,244,255,0.18)'] }
            : dropzone.isDragging
              ? { scale: 1.015, borderColor: '#62f4ff' }
              : { scale: 1, borderColor: 'rgba(98,244,255,0.16)' }
        }
        transition={{ duration: 0.45 }}
        className="panel relative overflow-hidden rounded-[32px] border border-dashed px-5 py-6 sm:px-6 sm:py-7"
      >
        {dropzone.file ? (
          <div className="drop-burst" aria-hidden="true">
            <span />
            <span />
            <span />
            <span />
            <span />
          </div>
        ) : null}

        <div className="relative z-10 flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
          <div className="flex flex-col gap-5 sm:flex-row sm:items-start">
            <div className="flex h-16 w-16 shrink-0 items-center justify-center rounded-[22px] border border-cyan-300/20 bg-cyan-300/10 text-cyan-100 shadow-[0_0_18px_rgba(98,244,255,0.16)]">
              <Icon size={28} />
            </div>

            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <span className="hero-chip label-font inline-flex items-center gap-2 rounded-full px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-cyan-50">
                  <Sparkles size={12} />
                  {empty ? 'Ready for upload' : 'Evidence staged'}
                </span>
                <span className="media-badge label-font rounded-full px-3 py-2 text-[11px] uppercase tracking-[0.16em] text-slate-300">Drag and drop or browse</span>
              </div>

              <div>
                <p className="heading-font text-2xl uppercase tracking-[0.18em] text-white sm:text-[1.8rem]">
                  {empty ? 'Upload forensic evidence' : 'Evidence locked in'}
                </p>
                <p className="label-font mt-3 max-w-2xl text-sm leading-7 text-slate-300">
                  Submit one media file and inspect the result in a web-native review flow built for large screens and mobile uploads alike.
                </p>
              </div>

              {dropzone.file ? (
                <div className="flex flex-wrap gap-2 text-sm text-slate-200">
                  <span className="media-badge label-font rounded-full px-3 py-2">{dropzone.file.name}</span>
                  <span className="media-badge data-font rounded-full px-3 py-2">{formatBytes(dropzone.file.size)}</span>
                  <span className="media-badge label-font rounded-full px-3 py-2 uppercase">{dropzone.preview?.kind}</span>
                </div>
              ) : (
                <div className="flex flex-wrap gap-2 text-xs text-slate-300">
                  <span className="media-badge label-font rounded-full px-3 py-2 uppercase tracking-[0.16em]">JPEG / PNG / WEBP</span>
                  <span className="media-badge label-font rounded-full px-3 py-2 uppercase tracking-[0.16em]">MP4 / WEBM</span>
                  <span className="media-badge label-font rounded-full px-3 py-2 uppercase tracking-[0.16em]">WAV / MP3 / OGG</span>
                </div>
              )}
            </div>
          </div>

          <div className="flex flex-col gap-3 sm:flex-row lg:flex-col lg:items-stretch">
            <button
              type="button"
              disabled={disabled}
              onClick={onBrowse}
              className="action-primary label-font inline-flex items-center justify-center gap-2 rounded-full px-5 py-3 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-45"
            >
              <UploadCloud size={16} />
              Browse file
            </button>
            {dropzone.file ? (
              <>
                <button
                  type="button"
                  disabled={disabled}
                  onClick={() => dropzone.clear()}
                  className="action-secondary label-font inline-flex items-center justify-center gap-2 rounded-full px-5 py-3 text-sm font-semibold transition disabled:opacity-45"
                >
                  <X size={16} />
                  Clear
                </button>
                <button
                  type="button"
                  disabled={disabled}
                  onClick={onBrowse}
                  className="action-secondary label-font inline-flex items-center justify-center gap-2 rounded-full px-5 py-3 text-sm font-semibold transition disabled:opacity-45"
                >
                  <RefreshCcw size={16} />
                  Replace
                </button>
              </>
            ) : null}
          </div>
        </div>
      </motion.div>

      {dropzone.error ? <div className="inline-error rounded-[24px] px-4 py-3 text-sm leading-7">{dropzone.error}</div> : null}

      {dropzone.preview?.kind === 'image' ? (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="panel overflow-hidden rounded-[32px] p-3 sm:p-4">
          <img src={dropzone.preview.url} alt="Selected upload preview" className="max-h-[420px] w-full rounded-[24px] object-contain bg-black/20" />
        </motion.div>
      ) : null}

      {dropzone.preview?.kind === 'video' ? (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="panel overflow-hidden rounded-[32px] p-4 sm:p-5">
          {dropzone.preview.thumbnailUrl ? (
            <img src={dropzone.preview.thumbnailUrl} alt="Video thumbnail" className="aspect-video w-full rounded-[24px] object-cover" />
          ) : (
            <video src={dropzone.preview.url} controls className="aspect-video w-full rounded-[24px] object-cover" />
          )}
        </motion.div>
      ) : null}

      {dropzone.preview?.kind === 'audio' ? (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="panel rounded-[32px] p-4 sm:p-5">
          <div className="audio-bars mb-4 h-28 rounded-[24px] bg-black/15 p-4 sm:h-32">
            {dropzone.preview.waveform.map((value, index) => (
              <span key={`${index}-${value}`} className="bg-cyan-300/75" style={{ height: `${Math.max(12, value * 180)}px` }} />
            ))}
          </div>
          <audio controls src={dropzone.preview.url} className="w-full" />
        </motion.div>
      ) : null}
    </div>
  );
}

export default DropZone;
