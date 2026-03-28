/**
 * Internal trace:
 * - Wrong before: progress feedback showed the percentage but still lacked the stronger stage framing expected from the redesigned web workflow.
 * - Fixed now: the progress bar includes clearer stage labels, a richer visual treatment, and better responsive readability.
 */

import { motion } from 'framer-motion';

const LABELS = {
  idle: 'Awaiting evidence',
  uploading: 'Uploading file securely',
  analysing: 'Running ensemble analysis',
  done: 'Analysis complete',
  error: 'Analysis halted',
};

const STAGES = ['idle', 'uploading', 'analysing', 'done'];

/**
 * @param {{ status: string, progress: number }} props
 */
function ProgressBar({ status, progress }) {
  return (
    <div className="panel relative overflow-hidden rounded-[32px] p-5 sm:p-6">
      {status === 'analysing' ? (
        <motion.div
          className="scanline"
          initial={{ x: '-100%' }}
          animate={{ x: ['-100%', '100%'] }}
          transition={{ repeat: Number.POSITIVE_INFINITY, duration: 1.25, ease: 'linear' }}
        />
      ) : null}

      <div className="relative z-10 space-y-5">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="section-kicker label-font text-[11px] font-semibold">Processing state</p>
            <p className="heading-font mt-2 text-xl uppercase tracking-[0.14em] text-white">{LABELS[status]}</p>
          </div>
          <div className="media-badge data-font inline-flex w-fit rounded-full px-4 py-2 text-sm text-slate-200">{progress}% complete</div>
        </div>

        <div className="grid gap-2 sm:grid-cols-4">
          {STAGES.map((stage) => {
            const active = stage === status || (stage === 'done' && status === 'done');
            const reached = STAGES.indexOf(stage) <= STAGES.indexOf(status === 'error' ? 'analysing' : status);
            return (
              <div
                key={stage}
                className={`rounded-2xl border px-3 py-3 text-center ${active ? 'border-cyan-300/30 bg-cyan-300/10' : reached ? 'border-white/8 bg-white/4' : 'border-white/6 bg-black/10'}`}
              >
                <p className={`label-font text-[11px] uppercase tracking-[0.18em] ${active ? 'text-cyan-100' : 'text-slate-400'}`}>{stage}</p>
              </div>
            );
          })}
        </div>

        <div className="h-3 rounded-full bg-white/6">
          <motion.div
            className="h-full rounded-full bg-gradient-to-r from-cyan-300 via-sky-300 to-emerald-300"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.45, ease: 'easeOut' }}
          />
        </div>
      </div>
    </div>
  );
}

export default ProgressBar;
