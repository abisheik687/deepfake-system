/**
 * Internal trace:
 * - Wrong before: the analyse page was functional but visually flat and did not fully use the wider web layout to guide users through the upload flow.
 * - Fixed now: the page combines a stronger responsive composition, a more premium upload workspace, and clearer sidebar guidance for the web-first product.
 */

import { useRef } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, AudioLines, Film, Image as ImageIcon, ShieldCheck, Upload, Workflow } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import DropZone from '../components/DropZone.jsx';
import ProgressBar from '../components/ProgressBar.jsx';
import { useDropZone } from '../hooks/useDropZone.js';

const surfaceCards = [
  { title: 'Images', detail: 'PNG, JPEG, WEBP', icon: ImageIcon },
  { title: 'Videos', detail: 'MP4, WEBM up to 100 MB', icon: Film },
  { title: 'Audio', detail: 'WAV, MP3, OGG', icon: AudioLines },
];

const checkpoints = [
  'Client-side validation catches unsupported files before upload.',
  'Server-side validation enforces size, mime type, and processing path.',
  'Results stay in one web session, so the workflow works cleanly on any screen size.',
];

/**
 * @param {{ analysis: ReturnType<import('../hooks/useAnalysis').useAnalysis> }} props
 */
function Analyse({ analysis }) {
  const navigate = useNavigate();
  const dropzone = useDropZone();
  const inputRef = useRef(null);

  const onBrowse = () => inputRef.current?.click();

  const onFileChange = async (event) => {
    const nextFile = event.target.files?.[0];
    await dropzone.selectFile(nextFile);
    event.target.value = '';
  };

  const onSubmit = async () => {
    if (!dropzone.file) {
      dropzone.setError('Choose a supported file before starting analysis.');
      return;
    }
    try {
      await analysis.analyseFile(dropzone.file, dropzone.preview);
      navigate('/results');
    } catch {
      // Error state is already handled by the hook.
    }
  };

  const busy = analysis.status === 'uploading' || analysis.status === 'analysing';

  return (
    <div className="scan-shell px-4 py-5 sm:px-6 lg:px-10 lg:py-8">
      <div className="mx-auto max-w-7xl space-y-8">
        <div className="flex flex-col gap-5 rounded-[30px] border border-white/6 bg-black/10 px-5 py-5 backdrop-blur sm:px-6 lg:flex-row lg:items-start lg:justify-between">
          <div className="space-y-3">
            <Link to="/" className="label-font inline-flex items-center gap-2 text-sm text-slate-400 transition hover:text-cyan-100">
              <ArrowLeft size={14} />
              Back to home
            </Link>
            <div className="space-y-3">
              <p className="section-kicker label-font text-xs font-semibold">Web-first analysis console</p>
              <h1 className="heading-font text-4xl uppercase tracking-[0.16em] text-white sm:text-5xl">Analyse upload</h1>
              <p className="max-w-3xl text-sm leading-7 text-slate-300 sm:text-base">
                The extension path is gone and the full workflow now lives here. Upload from desktop or mobile, track progress in one place,
                and land directly in a readable forensic results view.
              </p>
            </div>
          </div>
          <div className="grid gap-3 sm:grid-cols-3 lg:w-[28rem] lg:grid-cols-1 xl:w-[30rem] xl:grid-cols-3">
            {surfaceCards.map(({ title, detail, icon: Icon }) => (
              <div key={title} className="signal-card rounded-[24px] p-4">
                <div className="mb-3 flex h-11 w-11 items-center justify-center rounded-2xl border border-cyan-300/20 bg-cyan-300/10 text-cyan-100">
                  <Icon size={18} />
                </div>
                <p className="heading-font text-sm uppercase tracking-[0.14em] text-white">{title}</p>
                <p className="label-font mt-2 text-sm text-slate-400">{detail}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="grid gap-6 xl:grid-cols-[minmax(0,1.1fr)_22rem]">
          <div className="space-y-6">
            <DropZone dropzone={dropzone} disabled={busy} onBrowse={onBrowse} />
            <input ref={inputRef} type="file" className="hidden" onChange={onFileChange} />

            <ProgressBar status={analysis.status === 'idle' ? 'idle' : analysis.status} progress={analysis.progress} />

            {analysis.error ? (
              <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="inline-error rounded-[28px] px-5 py-5 sm:px-6">
                <p className="heading-font text-lg uppercase tracking-[0.16em] text-white">{analysis.error.title}</p>
                <p className="label-font mt-2 text-sm leading-7 text-rose-50">{analysis.error.message}</p>
                <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center">
                  <button type="button" onClick={onSubmit} className="action-secondary label-font rounded-full px-4 py-3 text-sm font-semibold transition">
                    {analysis.error.retryLabel}
                  </button>
                  <span className="data-font text-xs uppercase tracking-[0.18em] text-rose-100/80">{analysis.error.code}</span>
                </div>
              </motion.div>
            ) : null}

            <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap">
              <button
                type="button"
                disabled={busy}
                onClick={onSubmit}
                className="action-primary heading-font inline-flex items-center justify-center gap-2 rounded-full px-6 py-4 text-sm uppercase tracking-[0.16em] transition disabled:cursor-not-allowed disabled:opacity-45"
              >
                <Upload size={16} />
                {busy ? 'Analysing' : 'Analyse file'}
              </button>
              <button
                type="button"
                onClick={() => {
                  analysis.reset();
                  dropzone.clear();
                }}
                className="action-secondary label-font inline-flex items-center justify-center gap-2 rounded-full px-5 py-4 text-sm font-semibold transition"
              >
                <ShieldCheck size={16} />
                Reset session
              </button>
            </div>
          </div>

          <aside className="space-y-5 xl:sticky xl:top-6 xl:self-start">
            <div className="panel rounded-[28px] p-5 sm:p-6">
              <div className="mb-4 flex items-center gap-3">
                <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-cyan-300/18 bg-cyan-300/10 text-cyan-100">
                  <Workflow size={18} />
                </div>
                <div>
                  <p className="section-kicker label-font text-[11px] font-semibold">Workflow guide</p>
                  <p className="heading-font text-base uppercase tracking-[0.14em] text-white">Why this page works better</p>
                </div>
              </div>
              <div className="space-y-3">
                {checkpoints.map((point, index) => (
                  <div key={point} className="rounded-2xl border border-white/6 bg-white/3 px-4 py-3">
                    <div className="mb-2 flex items-center gap-2">
                      <span className="heading-font text-xs uppercase tracking-[0.18em] text-cyan-100">0{index + 1}</span>
                    </div>
                    <p className="label-font text-sm leading-7 text-slate-300">{point}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="signal-card rounded-[28px] p-5 sm:p-6">
              <p className="section-kicker label-font text-[11px] font-semibold">Limits</p>
              <div className="mt-3 grid gap-3 sm:grid-cols-3 xl:grid-cols-1">
                <div className="metric-tile rounded-2xl p-4">
                  <p className="label-font text-xs uppercase tracking-[0.16em] text-slate-400">Images</p>
                  <p className="heading-font mt-3 text-2xl text-cyan-100">20 MB</p>
                </div>
                <div className="metric-tile rounded-2xl p-4">
                  <p className="label-font text-xs uppercase tracking-[0.16em] text-slate-400">Audio</p>
                  <p className="heading-font mt-3 text-2xl text-emerald-200">20 MB</p>
                </div>
                <div className="metric-tile rounded-2xl p-4">
                  <p className="label-font text-xs uppercase tracking-[0.16em] text-slate-400">Video</p>
                  <p className="heading-font mt-3 text-2xl text-orange-200">100 MB</p>
                </div>
              </div>
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}

export default Analyse;
