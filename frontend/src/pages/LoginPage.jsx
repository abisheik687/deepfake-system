
import { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';
import { Shield, Lock, User, Zap } from 'lucide-react';
import { motion } from 'framer-motion';

const DEMO_EMAIL = 'demo@kavach.ai';
const DEMO_PASSWORD = 'kavach2026';

const LoginPage = () => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const { login } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);
        try {
            await login(username, password);
            navigate('/dashboard');
        } catch (_) {
            setError('Invalid credentials. Try demo@kavach.ai / kavach2026');
        } finally {
            setLoading(false);
        }
    };

    const handleDemoLogin = async () => {
        setError('');
        setLoading(true);
        try {
            await login(DEMO_EMAIL, DEMO_PASSWORD);
            navigate('/dashboard');
        } catch (_) {
            setError('Demo login failed — ensure the backend server is running.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-cyber-black flex items-center justify-center relative overflow-hidden">
            {/* Animated Background Grid */}
            <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-10 pointer-events-none"></div>
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-neon-blue to-transparent opacity-50"></div>

            {/* Ambient glow orbs */}
            <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-neon-blue/5 rounded-full blur-3xl pointer-events-none"></div>
            <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-neon-purple/5 rounded-full blur-3xl pointer-events-none"></div>

            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
                className="bg-cyber-gray border border-white/10 p-8 rounded-2xl w-full max-w-md shadow-2xl relative z-10"
            >
                <div className="text-center mb-8">
                    <div className="inline-block p-4 bg-neon-blue/10 rounded-full mb-4 border border-neon-blue/20">
                        <Shield size={40} className="text-neon-blue" />
                    </div>
                    <h1 className="text-3xl font-bold text-white tracking-wider">KAVACH<span className="text-neon-blue">.AI</span></h1>
                    <p className="text-gray-500 text-sm mt-2">Restricted Access • Forensic Division</p>
                </div>

                {/* ─── Quick Demo Login Button ─── */}
                <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={handleDemoLogin}
                    disabled={loading}
                    className="w-full mb-6 flex items-center justify-center gap-3 py-3 rounded-xl font-bold text-black
                               bg-gradient-to-r from-neon-blue via-cyan-400 to-neon-blue bg-size-200 bg-pos-0
                               hover:bg-pos-100 transition-all shadow-lg shadow-neon-blue/30
                               disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    <Zap size={20} className="shrink-0" />
                    <span>⚡ INSTANT DEMO ACCESS</span>
                </motion.button>

                {/* Divider */}
                <div className="flex items-center gap-3 mb-6">
                    <div className="flex-1 h-px bg-white/10"></div>
                    <span className="text-xs text-gray-600 uppercase tracking-widest">or manual login</span>
                    <div className="flex-1 h-px bg-white/10"></div>
                </div>

                <form onSubmit={handleSubmit} className="space-y-5">
                    <div>
                        <label className="block text-xs font-bold text-gray-400 uppercase mb-2">Officer ID</label>
                        <div className="relative">
                            <User className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={18} />
                            <input
                                type="text"
                                className="w-full bg-black/50 border border-white/10 rounded-lg py-3 pl-10 pr-4 text-white focus:outline-none focus:border-neon-blue transition-colors"
                                placeholder="Enter email address"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                            />
                        </div>
                    </div>

                    <div>
                        <label className="block text-xs font-bold text-gray-400 uppercase mb-2">Secure Key</label>
                        <div className="relative">
                            <Lock className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={18} />
                            <input
                                type="password"
                                className="w-full bg-black/50 border border-white/10 rounded-lg py-3 pl-10 pr-4 text-white focus:outline-none focus:border-neon-blue transition-colors"
                                placeholder="Enter password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                            />
                        </div>
                    </div>

                    {error && (
                        <div className="text-red-400 text-sm text-center bg-red-500/10 py-2 px-3 rounded-lg border border-red-500/20">
                            {error}
                        </div>
                    )}

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full bg-white/10 hover:bg-white/20 border border-white/20 text-white font-bold py-3 rounded-lg transition-all disabled:opacity-50"
                    >
                        {loading ? 'AUTHENTICATING...' : 'AUTHENTICATE'}
                    </button>
                </form>

                {/* Demo hint */}
                <div className="mt-5 p-3 rounded-lg bg-neon-blue/5 border border-neon-blue/20">
                    <p className="text-center text-xs text-gray-500">
                        Demo credentials: <span className="text-neon-blue font-mono">demo@kavach.ai</span> / <span className="text-neon-blue font-mono">kavach2026</span>
                    </p>
                </div>

                <div className="mt-4 text-center text-xs text-gray-700">
                    Authorized Personnel Only. Connection Monitored.
                </div>
            </motion.div>
        </div>
    );
};

export default LoginPage;
