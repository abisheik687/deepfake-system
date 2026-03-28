/**
 * Internal trace:
 * - Wrong before: the landing page sold the upload workflow, but the composition still felt more like a sparse dashboard shell than a premium web-first product.
 * - Fixed now: the home page has a stronger visual narrative, richer content hierarchy, responsive sections, and a more memorable forensic web-app presentation.
 */

import { motion } from 'framer-motion';
import {
  ArrowRight,
  AudioLines,
  Film,
  Image as ImageIcon,
  Radar,
  Shield,
  Sparkles,
  Workflow,
} from 'lucide-react';
import { Link } from 'react-router-dom';

const mediaCards = [
  { title: 'Image Forensics', detail: 'Frame-level artifact scoring with ensemble consensus.', icon: ImageIcon },
  { title: 'Video Sampling', detail: 'Every tenth frame scored with clip-level aggregation.', icon: Film },
  { title: 'Audio Spoofing', detail: 'Waveform-backed audio verdicts with model transparency.', icon: AudioLines },
];

const trustPoints = [
  'Upload-first flow designed for reliability over gimmicks.',
  'Responsive interface tuned for desktop review and mobile submissions.',
  'No browser extension dependency. The product is now fully web-native.',
];

const workflowSteps = [
  'Drop a file or browse from any device.',
  'The backend validates size, mime type, and processing path.',
  'Results explain verdict, confidence, model scores, and media evidence.',
];

function Home() {
  return (
    <div className="scan-shell px-4 py-5 sm:px-6 lg:px-10 lg:py-8">
      <div className="mx-auto flex min-h-[calc(100vh-2.5rem)] max-w-7xl flex-col gap-10">
        <header className="flex flex-col gap-4 rounded-[28px] border border-white/6 bg-black/10 px-4 py-4 backdrop-blur sm:flex-row sm:items-center sm:justify-between sm:px-6">
          <div className="flex items-center gap-3 sm:gap-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-cyan-300/30 bg-cyan-300/12 text-cyan-100 shadow-[0_0_18px_rgba(98,244,255,0.22)] sm:h-14 sm:w-14">
              <Shield size={24} />
            </div>
            <div>
              <p className="heading-font text-lg uppercase tracking-[0.32em] text-cyan-100 sm:text-xl">KAVACH-AI</p>
              <p className="label-font text-xs uppercase tracking-[0.18em] text-slate-400">Web forensic lab for media authenticity</p>
            </div>
          </div>
          <div className="flex flex-wrap gap-3">
            <div className="media-badge label-font inline-flex items-center gap-2 rounded-full px-4 py-2 text-xs uppercase tracking-[0.16em] text-slate-300">
              <span className="status-dot" />
              Web-native experience
            </div>
            <Link to="/analyse" className="action-primary heading-font inline-flex items-center justify-center gap-2 rounded-full px-5 py-3 text-sm uppercase tracking-[0.18em] transition">
              Launch analyse
              <ArrowRight size={16} />
            </Link>
          </div>
        </header>

        <main className="grid gap-8 lg:grid-cols-[1.12fr_0.88fr] lg:items-stretch">
          <section className="flex flex-col justify-between gap-8">
            <div className="space-y-6">
              <motion.div
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0 }}
                className="hero-chip label-font inline-flex items-center gap-2 rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-[0.24em] text-cyan-50"
              >
                <Sparkles size={14} />
                Production-ready upload investigation
              </motion.div>

              <div className="space-y-4">
                <motion.p
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.08 }}
                  className="section-kicker label-font text-xs font-semibold"
                >
                  Responsive. Explainable. Web-first.
                </motion.p>
                <h1 className="heading-font max-w-4xl text-5xl uppercase leading-[0.94] text-white sm:text-6xl lg:text-7xl xl:text-[5.4rem]">
                  A stronger
                  <span className="block bg-gradient-to-r from-cyan-200 via-white to-sky-300 bg-clip-text text-transparent">
                    deepfake detection web app
                  </span>
                </h1>
                <motion.p
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.18 }}
                  className="max-w-2xl text-base leading-8 text-slate-300 sm:text-lg"
                >
                  KAVACH-AI now runs as a fully web-native forensic experience. Upload media from any screen size, inspect ensemble confidence,
                  and review frame or waveform evidence in a polished interface designed for operational trust.
                </motion.p>
              </div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.28 }}
                className="flex flex-col gap-3 sm:flex-row sm:flex-wrap"
              >
                <Link to="/analyse" className="action-primary heading-font inline-flex items-center justify-center gap-2 rounded-full px-6 py-4 text-sm uppercase tracking-[0.18em] transition">
                  Upload evidence
                  <ArrowRight size={16} />
                </Link>
                <div className="action-secondary label-font inline-flex items-center justify-center rounded-full px-5 py-4 text-sm font-semibold text-slate-200">
                  JPEG, PNG, WEBP, MP4, WEBM, WAV, MP3, OGG
                </div>
              </motion.div>
            </div>

            <div className="grid gap-4 md:grid-cols-3">
              {mediaCards.map(({ title, detail, icon: Icon }, index) => (
                <motion.article
                  key={title}
                  initial={{ opacity: 0, y: 24 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.36 + index * 0.08 }}
                  className="signal-card rounded-[28px] p-5"
                >
                  <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl border border-cyan-300/18 bg-cyan-300/10 text-cyan-100">
                    <Icon size={20} />
                  </div>
                  <h2 className="heading-font text-lg uppercase tracking-[0.14em] text-white">{title}</h2>
                  <p className="label-font mt-3 text-sm leading-7 text-slate-300">{detail}</p>
                </motion.article>
              ))}
            </div>
          </section>

          <motion.section
            initial={{ opacity: 0, x: 24 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="panel soft-grid relative rounded-[32px] p-5 sm:p-6"
          >
            <div className="relative z-10 flex flex-col gap-6">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                <div>
                  <p className="section-kicker label-font text-xs font-semibold">Signal cockpit</p>
                  <h2 className="heading-font mt-2 text-2xl uppercase tracking-[0.16em] text-white sm:text-3xl">Upload-first mission control</h2>
                </div>
                <div className="media-badge label-font inline-flex w-fit items-center gap-2 rounded-full px-4 py-2 text-xs uppercase tracking-[0.16em] text-slate-300">
                  <Radar size={14} />
                  No extension required
                </div>
              </div>

              <div className="grid gap-4 sm:grid-cols-3">
                <div className="metric-tile rounded-3xl p-4">
                  <p className="label-font text-xs uppercase tracking-[0.16em] text-slate-400">Surface</p>
                  <p className="heading-font mt-3 text-3xl text-cyan-100">Web</p>
                  <p className="label-font mt-2 text-sm text-slate-300">One interface across laptop, tablet, and phone.</p>
                </div>
                <div className="metric-tile rounded-3xl p-4">
                  <p className="label-font text-xs uppercase tracking-[0.16em] text-slate-400">Pipeline</p>
                  <p className="heading-font mt-3 text-3xl text-white">4+1</p>
                  <p className="label-font mt-2 text-sm text-slate-300">Four image slots plus dedicated audio scoring.</p>
                </div>
                <div className="metric-tile rounded-3xl p-4">
                  <p className="label-font text-xs uppercase tracking-[0.16em] text-slate-400">Video</p>
                  <p className="heading-font mt-3 text-3xl text-orange-200">10f</p>
                  <p className="label-font mt-2 text-sm text-slate-300">Every tenth frame sampled for clip review.</p>
                </div>
              </div>

              <div className="grid gap-4 xl:grid-cols-[0.95fr_1.05fr]">
                <div className="signal-card rounded-[28px] p-5">
                  <div className="mb-4 flex items-center gap-3">
                    <Workflow className="text-cyan-200" size={18} />
                    <p className="heading-font text-base uppercase tracking-[0.14em] text-white">Workflow</p>
                  </div>
                  <div className="space-y-3">
                    {workflowSteps.map((step, index) => (
                      <div key={step} className="flex gap-3 rounded-2xl border border-white/6 bg-white/3 px-4 py-3">
                        <div className="heading-font flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-cyan-300/20 bg-cyan-300/10 text-xs text-cyan-100">
                          0{index + 1}
                        </div>
                        <p className="label-font text-sm leading-6 text-slate-300">{step}</p>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="signal-card rounded-[28px] p-5">
                  <div className="mb-4 flex items-center justify-between gap-3">
                    <p className="heading-font text-base uppercase tracking-[0.14em] text-white">Why this feels better</p>
                    <span className="status-dot" />
                  </div>
                  <div className="space-y-3">
                    {trustPoints.map((point) => (
                      <div key={point} className="rounded-2xl border border-white/6 bg-black/12 px-4 py-3">
                        <p className="label-font text-sm leading-6 text-slate-300">{point}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </motion.section>
        </main>
      </div>
    </div>
  );
}

export default Home;
