/**
 * Internal trace:
 * - Wrong before: the verdict banner worked functionally but did not feel cinematic enough for the upgraded web presentation.
 * - Fixed now: the banner has stronger copy hierarchy, richer layout, and more distinctive motion while staying responsive.
 */

import { motion } from 'framer-motion';
import { AlertTriangle, ShieldAlert, ShieldCheck } from 'lucide-react';

const MAP = {
  FAKE: {
    icon: ShieldAlert,
    border: 'rgba(255,92,122,0.4)',
    background: 'linear-gradient(135deg, rgba(255,92,122,0.2), rgba(18,27,45,0.92))',
    glow: '0 0 30px rgba(255,92,122,0.2)',
    label: 'High forensic concern',
    description: 'The ensemble leaned toward synthetic manipulation and highlighted suspicious artifact patterns.',
    animate: { x: [0, -5, 5, -2, 0], opacity: [0.7, 1, 1, 1, 1] },
  },
  REAL: {
    icon: ShieldCheck,
    border: 'rgba(61,246,163,0.34)',
    background: 'linear-gradient(135deg, rgba(61,246,163,0.16), rgba(18,27,45,0.92))',
    glow: '0 0 28px rgba(61,246,163,0.16)',
    label: 'Authenticity signal holds',
    description: 'The weighted models converged toward a real verdict with comparatively stable evidence patterns.',
    animate: { opacity: [0.35, 1], scale: [0.985, 1.01, 1] },
  },
  UNCERTAIN: {
    icon: AlertTriangle,
    border: 'rgba(255,159,82,0.36)',
    background: 'linear-gradient(135deg, rgba(255,159,82,0.18), rgba(18,27,45,0.92))',
    glow: '0 0 28px rgba(255,159,82,0.16)',
    label: 'Model disagreement detected',
    description: 'The verdict is cautious because the contributing models diverged beyond the confidence spread threshold.',
    animate: { opacity: [0.45, 1, 0.78, 1] },
  },
};

/**
 * @param {{ verdict: 'REAL'|'FAKE'|'UNCERTAIN', fakeProbability: number }} props
 */
function VerdictBanner({ verdict, fakeProbability }) {
  const config = MAP[verdict] || MAP.UNCERTAIN;
  const Icon = config.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      animate={config.animate}
      transition={{ duration: 0.8 }}
      className="rounded-[32px] border p-5 sm:p-6"
      style={{ background: config.background, borderColor: config.border, boxShadow: config.glow }}
    >
      <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex items-start gap-4">
          <div className="flex h-16 w-16 shrink-0 items-center justify-center rounded-[24px] border border-white/10 bg-white/6 text-white sm:h-18 sm:w-18">
            <Icon size={28} />
          </div>
          <div className="space-y-3">
            <p className="section-kicker label-font text-[11px] font-semibold">Verdict</p>
            <div>
              <h2 className="heading-font text-4xl uppercase tracking-[0.18em] text-white sm:text-5xl">{verdict}</h2>
              <p className="label-font mt-2 text-sm font-semibold uppercase tracking-[0.12em] text-slate-200">{config.label}</p>
            </div>
            <p className="label-font max-w-2xl text-sm leading-7 text-slate-200/92">{config.description}</p>
          </div>
        </div>

        <div className="grid gap-3 sm:grid-cols-2 lg:w-[20rem]">
          <div className="metric-tile rounded-[24px] p-4">
            <p className="label-font text-xs uppercase tracking-[0.16em] text-slate-300">Fake probability</p>
            <p className="heading-font mt-3 text-3xl text-white">{(fakeProbability * 100).toFixed(1)}%</p>
          </div>
          <div className="metric-tile rounded-[24px] p-4">
            <p className="label-font text-xs uppercase tracking-[0.16em] text-slate-300">Review mode</p>
            <p className="heading-font mt-3 text-xl uppercase text-white">Web forensic</p>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

export default VerdictBanner;
