/**
 * Internal trace:
 * - Wrong before: the video frame grid worked, but the cards could better emphasize the frame identity and forensic score in the refreshed UI.
 * - Fixed now: frame cards use a richer overlay treatment and respond more cleanly across breakpoints.
 */

import { motion } from 'framer-motion';

/**
 * @param {{ frames: { index: number, fake_probability: number, image_base64: string }[] }} props
 */
function VideoFrameGrid({ frames }) {
  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
      {frames.map((frame, index) => (
        <motion.div
          key={`${frame.index}-${index}`}
          initial={{ opacity: 0, filter: 'blur(12px)', scale: 0.985 }}
          animate={{ opacity: 1, filter: 'blur(0px)', scale: 1 }}
          transition={{ delay: index * 0.08, duration: 0.45 }}
          className="panel overflow-hidden rounded-[28px]"
        >
          <div className="relative">
            <img src={frame.image_base64} alt={`Analysed frame ${frame.index}`} className="aspect-video w-full object-cover" />
            <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/65 to-transparent px-4 py-4">
              <div className="flex items-end justify-between gap-3">
                <span className="media-badge label-font rounded-full px-3 py-2 text-[11px] uppercase tracking-[0.16em] text-slate-100">Frame {frame.index}</span>
                <span className="heading-font text-lg text-cyan-100">{(frame.fake_probability * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  );
}

export default VideoFrameGrid;
