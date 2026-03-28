/**
 * Internal trace:
 * - Wrong before: the score cards were accurate but visually repetitive and did not carry enough hierarchy into the redesigned results view.
 * - Fixed now: each model card has stronger labeling, richer confidence framing, and responsive detail badges.
 */

import { animate, motion, useMotionValue } from 'framer-motion';
import { useEffect, useState } from 'react';

function getBand(probability) {
  if (probability >= 0.8) return 'High signal';
  if (probability >= 0.55) return 'Moderate signal';
  return 'Low signal';
}

/**
 * @param {{ score: { model: string, fake_prob: number, weight: number, mode: string } }} props
 */
function ResultCard({ score }) {
  const count = useMotionValue(0);
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    const unsubscribe = count.on('change', (latest) => setDisplay(latest));
    const controls = animate(count, score.fake_prob * 100, { duration: 0.8, ease: 'easeOut' });
    return () => {
      controls.stop();
      unsubscribe();
    };
  }, [count, score.fake_prob]);

  return (
    <motion.div initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} className="signal-card rounded-[28px] p-5">
      <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <p className="heading-font text-lg uppercase tracking-[0.14em] text-slate-100">{score.model}</p>
          <p className="label-font mt-2 text-xs uppercase tracking-[0.16em] text-slate-400">{score.mode}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <span className="media-badge label-font rounded-full px-3 py-2 text-xs font-semibold text-cyan-100">weight {score.weight.toFixed(2)}</span>
          <span className="media-badge label-font rounded-full px-3 py-2 text-xs text-slate-200">{getBand(score.fake_prob)}</span>
        </div>
      </div>

      <div className="mb-3 flex items-end justify-between gap-3">
        <span className="label-font text-sm text-slate-400">Fake probability</span>
        <span className="heading-font text-3xl text-cyan-100">{display.toFixed(1)}%</span>
      </div>

      <div className="h-3 rounded-full bg-white/5">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${score.fake_prob * 100}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          className="h-full rounded-full bg-gradient-to-r from-cyan-300 via-sky-300 to-white"
        />
      </div>
    </motion.div>
  );
}

export default ResultCard;
