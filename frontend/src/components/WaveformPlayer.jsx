/**
 * Internal trace:
 * - Wrong before: the waveform player showed the data, but it did not present audio evidence with enough polish for the upgraded web app.
 * - Fixed now: the waveform card uses stronger spacing, verdict-aware color, and a cleaner playback section.
 */

import { motion } from 'framer-motion';

/**
 * @param {{ waveform: number[], url: string, verdict: string }} props
 */
function WaveformPlayer({ waveform, url, verdict }) {
  const color = verdict === 'FAKE' ? 'bg-rose-400/80' : verdict === 'REAL' ? 'bg-emerald-300/80' : 'bg-amber-300/80';

  return (
    <div className="panel rounded-[32px] p-5 sm:p-6">
      <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="section-kicker label-font text-[11px] font-semibold">Audio evidence</p>
          <p className="heading-font mt-2 text-xl uppercase tracking-[0.14em] text-white">Waveform review</p>
        </div>
        <div className="media-badge label-font rounded-full px-4 py-2 text-xs uppercase tracking-[0.16em] text-slate-200">{verdict}</div>
      </div>

      <div className="mb-5 flex h-32 items-end gap-1 overflow-hidden rounded-[24px] bg-black/15 p-4 sm:h-36">
        {waveform.map((value, index) => (
          <motion.div
            key={`${index}-${value}`}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: `${Math.max(12, value * 200)}px`, opacity: 1 }}
            transition={{ delay: index * 0.01, duration: 0.25 }}
            className={`flex-1 rounded-full ${color}`}
          />
        ))}
      </div>
      {url ? <audio controls src={url} className="w-full" /> : null}
    </div>
  );
}

export default WaveformPlayer;
