/**
 * Internal trace:
 * - Wrong before: the results page was correct but still too report-like, with weaker visual hierarchy and not enough responsive composition for a polished web product.
 * - Fixed now: the results experience has a stronger hero, media preview, clearer KPI tiles, and more fluid layouts across screen sizes while staying fully driven by live API data.
 */

import { motion } from 'framer-motion';
import { ArrowLeft, AudioLines, Clock3, FileVideo2, Image as ImageIcon, RotateCcw, Shield, Waves } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import ResultCard from '../components/ResultCard.jsx';
import VerdictBanner from '../components/VerdictBanner.jsx';
import VideoFrameGrid from '../components/VideoFrameGrid.jsx';
import WaveformPlayer from '../components/WaveformPlayer.jsx';

function ProbabilityRing({ probability }) {
  const radius = 72;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - circumference * probability;

  return (
    <div className="panel orbital-ring relative flex w-full max-w-md flex-col items-center rounded-[32px] p-5 sm:p-6">
      <svg viewBox="0 0 180 180" className="h-52 w-52 sm:h-60 sm:w-60">
        <circle cx="90" cy="90" r={radius} stroke="rgba(255,255,255,0.07)" strokeWidth="12" fill="none" />
        <motion.circle
          cx="90"
          cy="90"
          r={radius}
          stroke="url(#ringGradient)"
          strokeWidth="12"
          fill="none"
          strokeLinecap="round"
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 0.9, ease: 'easeOut' }}
          strokeDasharray={circumference}
          transform="rotate(-90 90 90)"
        />
        <defs>
          <linearGradient id="ringGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#7af8ff" />
            <stop offset="55%" stopColor="#3aa6ff" />
            <stop offset="100%" stopColor="#3df6a3" />
          </linearGradient>
        </defs>
        <text x="90" y="80" textAnchor="middle" className="heading-font fill-white text-[14px] uppercase tracking-[0.22em]">Fake probability</text>
        <text x="90" y="110" textAnchor="middle" className="heading-font fill-cyan-100 text-[28px]">{(probability * 100).toFixed(1)}%</text>
      </svg>
      <p className="label-font text-center text-sm leading-6 text-slate-400">Weighted ensemble signal across the analysed upload.</p>
    </div>
  );
}

function AssetPreviewCard({ asset }) {
  if (!asset) return null;

  return (
    <div className="panel rounded-[32px] p-5 sm:p-6">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <p className="section-kicker label-font text-[11px] font-semibold">Uploaded media</p>
          <p className="heading-font mt-2 text-lg uppercase tracking-[0.14em] text-white">{asset.name}</p>
        </div>
        <div className="media-badge label-font rounded-full px-4 py-2 text-xs uppercase tracking-[0.16em] text-slate-300">{asset.fileType}</div>
      </div>

      {asset.fileType === 'image' ? <img src={asset.previewUrl} alt={asset.name} className="max-h-[320px] w-full rounded-[24px] object-contain bg-black/20" /> : null}
      {asset.fileType === 'video' ? (
        <img src={asset.thumbnailUrl || asset.previewUrl} alt={asset.name} className="aspect-video w-full rounded-[24px] object-cover" />
      ) : null}
      {asset.fileType === 'audio' ? (
        <div className="space-y-4">
          <div className="audio-bars h-32 rounded-[24px] bg-black/15 p-4">
            {(asset.waveform || []).map((value, index) => (
              <span key={`${index}-${value}`} className="bg-cyan-300/70" style={{ height: `${Math.max(12, value * 180)}px` }} />
            ))}
          </div>
          <audio controls src={asset.previewUrl} className="w-full" />
        </div>
      ) : null}
    </div>
  );
}

/**
 * @param {{ analysis: ReturnType<import('../hooks/useAnalysis').useAnalysis> }} props
 */
function Results({ analysis }) {
  const navigate = useNavigate();
  const { result, asset } = analysis;

  if (!result) {
    return (
      <div className="scan-shell flex items-center justify-center px-4 py-6 sm:px-6 lg:px-10">
        <div className="panel max-w-xl rounded-[32px] p-8 text-center">
          <p className="heading-font text-2xl uppercase tracking-[0.18em] text-white">No result loaded</p>
          <p className="label-font mt-3 text-sm leading-7 text-slate-300">Start a new upload to generate a deepfake analysis result.</p>
          <Link to="/analyse" className="action-primary heading-font mt-6 inline-flex rounded-full px-5 py-3 text-sm uppercase tracking-[0.16em] transition">
            Go to analyse
          </Link>
        </div>
      </div>
    );
  }

  const metaCards = [
    { label: 'File type', value: result.file_type, icon: result.file_type === 'video' ? FileVideo2 : result.file_type === 'audio' ? AudioLines : ImageIcon },
    { label: 'Confidence', value: `${(result.overall_confidence * 100).toFixed(1)}%`, icon: Shield },
    { label: 'Processing time', value: `${(result.processing_time_ms / 1000).toFixed(2)}s`, icon: Clock3 },
  ];

  return (
    <div className="scan-shell px-4 py-5 sm:px-6 lg:px-10 lg:py-8">
      <div className="mx-auto max-w-7xl space-y-8">
        <div className="flex flex-col gap-4 rounded-[30px] border border-white/6 bg-black/10 px-5 py-5 backdrop-blur sm:flex-row sm:items-center sm:justify-between sm:px-6">
          <div>
            <button type="button" onClick={() => navigate(-1)} className="label-font mb-3 inline-flex items-center gap-2 text-sm text-slate-400 transition hover:text-cyan-100">
              <ArrowLeft size={14} />
              Back
            </button>
            <p className="section-kicker label-font text-[11px] font-semibold">Forensic output</p>
            <h1 className="heading-font mt-2 text-4xl uppercase tracking-[0.16em] text-white sm:text-5xl">Detection results</h1>
          </div>
          <button
            type="button"
            onClick={() => {
              analysis.reset();
              navigate('/analyse');
            }}
            className="action-secondary heading-font inline-flex items-center justify-center gap-2 rounded-full px-5 py-4 text-sm uppercase tracking-[0.16em] transition"
          >
            <RotateCcw size={16} />
            Analyse another
          </button>
        </div>

        <VerdictBanner verdict={result.verdict} fakeProbability={result.fake_probability} />

        <div className="grid gap-6 xl:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
          <div className="space-y-6">
            <ProbabilityRing probability={result.fake_probability} />
            <AssetPreviewCard asset={asset} />
          </div>

          <div className="space-y-6">
            <div className="grid gap-4 sm:grid-cols-3">
              {metaCards.map(({ label, value, icon: Icon }) => (
                <div key={label} className="signal-card rounded-[28px] p-5">
                  <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-2xl border border-cyan-300/18 bg-cyan-300/10 text-cyan-100">
                    <Icon size={18} />
                  </div>
                  <p className="label-font text-xs uppercase tracking-[0.16em] text-slate-400">{label}</p>
                  <p className="heading-font mt-3 text-2xl uppercase text-white">{value}</p>
                </div>
              ))}
            </div>

            {result.warnings?.length ? (
              <div className="panel rounded-[32px] p-5 sm:p-6">
                <p className="heading-font text-lg uppercase tracking-[0.16em] text-amber-200">Warnings</p>
                <div className="mt-4 space-y-3">
                  {result.warnings.map((warning) => (
                    <div key={warning} className="rounded-2xl border border-amber-400/16 bg-amber-400/10 px-4 py-3">
                      <p className="label-font text-sm leading-7 text-slate-100">{warning}</p>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}

            <div className="panel rounded-[32px] p-5 sm:p-6">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="section-kicker label-font text-[11px] font-semibold">Model breakdown</p>
                  <p className="heading-font mt-2 text-2xl uppercase tracking-[0.14em] text-white">Confidence by model</p>
                </div>
                <div className="media-badge data-font rounded-full px-4 py-2 text-xs uppercase tracking-[0.14em] text-slate-300">
                  {result.model_scores.length} active scores
                </div>
              </div>
              <div className="mt-5 grid gap-4 lg:grid-cols-2">
                {result.model_scores.map((score, index) => (
                  <motion.div key={`${score.model}-${score.mode}`} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.08 }}>
                    <ResultCard score={score} />
                  </motion.div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {result.video_frame_previews?.length ? (
          <section className="space-y-4">
            <div className="flex items-center gap-3">
              <FileVideo2 className="text-cyan-200" size={20} />
              <h2 className="heading-font text-2xl uppercase tracking-[0.16em] text-white">Sampled video frames</h2>
            </div>
            <VideoFrameGrid frames={result.video_frame_previews} />
          </section>
        ) : null}

        {result.audio_result ? (
          <section className="space-y-4">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center gap-3">
                <Waves className="text-cyan-200" size={20} />
                <h2 className="heading-font text-2xl uppercase tracking-[0.16em] text-white">Audio result</h2>
              </div>
              <span className="media-badge label-font rounded-full px-4 py-2 text-sm text-slate-200">
                {result.audio_result.verdict} · {(result.audio_result.fake_probability * 100).toFixed(1)}%
              </span>
            </div>
            <WaveformPlayer waveform={result.audio_result.waveform || asset?.waveform || []} url={asset?.previewUrl || ''} verdict={result.audio_result.verdict} />
          </section>
        ) : null}
      </div>
    </div>
  );
}

export default Results;
