import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { Shield, Brain, Zap, ArrowRight, Video, Image, Radio } from 'lucide-react';

const HomePage = () => {
    const navigate = useNavigate();

    return (
        <div className="min-h-screen bg-cyber-black text-white selection:bg-neon-blue/30 selection:text-neon-blue">
            {/* Header / Nav */}
            <header className="fixed top-0 w-full z-50 bg-cyber-black/80 backdrop-blur-md border-b border-white/5">
                <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <Shield className="text-neon-blue w-8 h-8" />
                        <span className="text-2xl font-bold tracking-widest text-white">
                            DEEP<span className="text-neon-blue">SHIELD</span>
                        </span>
                    </div>
                    <nav className="hidden md:flex items-center gap-8">
                        <a href="#features" className="text-sm text-gray-400 hover:text-white transition-colors">Core Features</a>
                        <a href="#how-it-works" className="text-sm text-gray-400 hover:text-white transition-colors">How It Works</a>
                        <button
                            onClick={() => navigate('/login')}
                            className="px-6 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-sm font-medium transition-all"
                        >
                            Sign In
                        </button>
                    </nav>
                </div>
            </header>

            {/* Hero Section */}
            <main className="pt-32 pb-20 px-6 max-w-7xl mx-auto">
                <div className="text-center max-w-4xl mx-auto mt-20">
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-neon-blue/10 border border-neon-blue/20 text-neon-blue text-xs font-medium uppercase tracking-wider mb-8"
                    >
                        <span className="w-2 h-2 rounded-full bg-neon-blue animate-pulse"></span>
                        Production-Ready Deepfake Detection
                    </motion.div>

                    <motion.h1
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.1 }}
                        className="text-6xl md:text-7xl font-extrabold tracking-tight mb-8"
                    >
                        Trust What You <span className="text-transparent bg-clip-text bg-gradient-to-r from-neon-blue to-neon-purple">See.</span>
                    </motion.h1>

                    <motion.p
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.2 }}
                        className="text-xl text-gray-400 mb-12 leading-relaxed max-w-2xl mx-auto"
                    >
                        DeepShield AI is a unified threat intelligence platform providing a single, authoritative deepfake risk score based on an orchestrated ensemble of advanced vision and frequency models.
                    </motion.p>

                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3 }}
                        className="flex flex-col sm:flex-row items-center justify-center gap-4"
                    >
                        <button
                            onClick={() => navigate('/login')}
                            className="w-full sm:w-auto px-8 py-4 bg-neon-blue hover:bg-neon-blue/90 text-black font-bold rounded-lg flex items-center justify-center gap-2 transition-all transform hover:scale-105"
                        >
                            Access Command Center <ArrowRight size={20} />
                        </button>
                    </motion.div>
                </div>

                {/* Core Features */}
                <div id="features" className="mt-40">
                    <h2 className="text-3xl font-bold text-center mb-16">Unified Detection Pipeline</h2>
                    <div className="grid md:grid-cols-3 gap-8">
                        <FeatureCard
                            icon={Image}
                            title="Image Analysis"
                            desc="Analyze static images with a balanced tier of ViT and Frequency models to detect sub-pixel manipulation."
                            color="neon-blue"
                        />
                        <FeatureCard
                            icon={Video}
                            title="Video Orchestration"
                            desc="Temporal frame aggregation analyzes video streams across multiple models to find fleeting anomalies."
                            color="neon-purple"
                        />
                        <FeatureCard
                            icon={Radio}
                            title="Live Enforcement"
                            desc="Optimized <100ms pipeline utilizing fast frequency models to protect live webcam feeds and online meetings."
                            color="neon-green"
                        />
                    </div>
                </div>

                {/* How It Works */}
                <div id="how-it-works" className="mt-40 text-center">
                    <h2 className="text-3xl font-bold mb-8">Architected for Accuracy</h2>
                    <p className="max-w-3xl mx-auto text-gray-400 leading-relaxed mb-16">
                        Unlike fragmented scripts, DeepShield uses a <strong className="text-white">Central Model Orchestrator</strong>. Every scan queries a registry of healthy AI models concurrently. Results are normalized via temperature scaling and fused using weighted soft-voting to produce a single, deterministic Deepfake Risk Score (0-100).
                    </p>

                    <div className="p-8 border border-white/5 rounded-2xl bg-white/[0.02] flex items-center justify-center gap-4 md:gap-12 flex-wrap">
                        <Badge text="ViT-Patch" />
                        <span className="text-gray-600">+</span>
                        <Badge text="EfficientNet" />
                        <span className="text-gray-600">+</span>
                        <Badge text="DCT Frequency" />
                        <span className="text-neon-blue font-bold">=</span>
                        <div className="px-6 py-3 bg-neon-blue/10 border border-neon-blue/30 text-neon-blue rounded-xl font-bold text-xl">
                            Verified Result
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
};

const FeatureCard = ({ icon: Icon, title, desc, color }) => (
    <div className={`p-8 rounded-2xl border border-white/5 bg-white/[0.02] hover:bg-white/[0.04] hover:border-${color}/30 transition-all group`}>
        <div className={`w-14 h-14 rounded-xl bg-${color}/10 border border-${color}/20 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform`}>
            <Icon className={`text-${color}`} size={28} />
        </div>
        <h3 className="text-xl font-bold mb-3">{title}</h3>
        <p className="text-gray-400 leading-relaxed">{desc}</p>
    </div>
);

const Badge = ({ text }) => (
    <div className="px-4 py-2 border border-white/10 rounded-lg text-sm text-gray-300 bg-white/5">
        {text}
    </div>
);

export default HomePage;
