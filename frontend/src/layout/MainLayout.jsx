
import { useState } from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, Upload, Activity, FileText, Menu, X, Shield, Lock } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const SidebarItem = ({ icon: Icon, label, path, active }) => (
    <Link to={path}>
        <div className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${active ? 'bg-neon-blue/10 text-neon-blue border-r-2 border-neon-blue' : 'text-gray-400 hover:bg-white/5 hover:text-white'}`}>
            <Icon size={20} />
            <span className="font-medium">{label}</span>
        </div>
    </Link>
);

const MainLayout = () => {
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const location = useLocation();

    const navItems = [
        { icon: LayoutDashboard, label: 'Dashboard', path: '/' },
        { icon: Upload, label: 'New Scan', path: '/upload' },
        { icon: Activity, label: 'Live Monitor', path: '/monitor' },
        { icon: FileText, label: 'Reports', path: '/reports' },
    ];

    return (
        <div className="flex h-screen bg-cyber-black text-white overflow-hidden">
            {/* Sidebar */}
            <AnimatePresence mode="wait">
                {isSidebarOpen && (
                    <motion.div
                        initial={{ width: 0, opacity: 0 }}
                        animate={{ width: 260, opacity: 1 }}
                        exit={{ width: 0, opacity: 0 }}
                        className="h-full bg-cyber-gray border-r border-white/10 flex flex-col"
                    >
                        <div className="p-6 flex items-center gap-2 border-b border-white/10">
                            <Shield className="text-neon-blue" size={32} />
                            <div>
                                <h1 className="text-xl font-bold tracking-wider">KAVACH<span className="text-neon-blue">.AI</span></h1>
                                <p className="text-xs text-gray-500">Forensic Intelligence</p>
                            </div>
                        </div>

                        <nav className="flex-1 p-4 space-y-2 mt-4">
                            {navItems.map((item) => (
                                <SidebarItem
                                    key={item.path}
                                    {...item}
                                    active={location.pathname === item.path}
                                />
                            ))}
                        </nav>

                        <div className="p-4 border-t border-white/10">
                            <div className="flex items-center gap-3 text-gray-400 text-sm">
                                <Lock size={16} />
                                <span>Secure Connection</span>
                            </div>
                            <div className="mt-2 text-xs text-gray-600">
                                v1.0.0 (Beta)
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Main Content */}
            <div className="flex-1 flex flex-col min-w-0">
                <header className="h-16 border-b border-white/10 bg-cyber-black/50 backdrop-blur-md flex items-center justify-between px-6">
                    <button
                        onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                        className="p-2 hover:bg-white/5 rounded-full text-gray-400 hover:text-white"
                    >
                        {isSidebarOpen ? <X size={20} /> : <Menu size={20} />}
                    </button>

                    <div className="flex items-center gap-4">
                        <span className="text-sm text-gray-400">Officer: <span className="text-white">Admin User</span></span>
                        <div className="w-8 h-8 bg-neon-blue rounded-full flex items-center justify-center text-black font-bold">
                            A
                        </div>
                    </div>
                </header>

                <main className="flex-1 overflow-auto p-6 relative">
                    <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-5 pointer-events-none"></div>
                    <Outlet />
                </main>
            </div>
        </div>
    );
};

export default MainLayout;
