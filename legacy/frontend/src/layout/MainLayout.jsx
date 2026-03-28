import { useState } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { useKeyboardShortcuts } from '../hooks/useKeyboardShortcuts';

const NAV = [
    { to: '/dashboard', icon: '⬡', label: 'Dashboard', shortcut: 'D' },
    { to: '/scan', icon: '⊕', label: 'New Scan', shortcut: 'S' },
    { to: '/monitor', icon: '👁', label: 'Monitoring', shortcut: 'O' },
    { to: '/alerts', icon: '⚠', label: 'Alerts', shortcut: 'A' },
    { to: '/reports', icon: '📄', label: 'Reports', shortcut: 'R' },
    { to: '/models', icon: '🧠', label: 'Models', shortcut: 'E' },
    { to: '/agency', icon: '🕵️', label: 'AI Agency', shortcut: 'Y' },
    { to: '/webcam', icon: '📹', label: 'Webcam', shortcut: 'W' },
    { to: '/live', icon: '◉', label: 'RTSP Stream', shortcut: 'L' },
    { to: '/training', icon: '🏋', label: 'Training', shortcut: 'T' },
    { to: '/audit', icon: '📊', label: 'System Audit', shortcut: 'U' },
    { to: '/advanced', icon: '⚡', label: 'Advanced', shortcut: 'V' },
    { to: '/settings', icon: '⚙', label: 'Settings', shortcut: '' },
];

export default function MainLayout() {
    const [collapsed, setCollapsed] = useState(false);
    useKeyboardShortcuts();

    return (
        <div className='flex h-screen overflow-hidden'
            style={{ background: 'var(--bg-base)' }}>

            {/* ── Sidebar ── */}
            <aside
                style={{
                    width: collapsed ? 64 : 220,
                    background: 'var(--bg-secondary)',
                    borderRight: '1px solid var(--border)',
                    transition: 'width 250ms ease',
                    flexShrink: 0,
                }}
                className='flex flex-col py-4'
            >
                {/* Logo */}
                <div className='flex items-center gap-3 px-4 mb-8'>
                    <div className='w-8 h-8 rounded flex items-center justify-center font-bold text-sm'
                        style={{ background: 'var(--cyan)', color: '#0A1628' }}>K</div>
                    {!collapsed && (
                        <span className='font-bold tracking-wide'
                            style={{ color: 'var(--cyan)', fontSize: 15 }}>KAVACH-AI</span>
                    )}
                </div>

                {/* Nav Links */}
                <nav className='flex-1 flex flex-col gap-1 px-2'>
                    {NAV.map(({ to, icon, label, shortcut }) => (
                        <NavLink
                            key={to} to={to}
                            className={({ isActive }) =>
                                `flex items-center gap-3 px-3 py-2.5 rounded-md text-sm
                 transition-all duration-150 group
                 ${isActive
                                    ? 'bg-cyan-900/40 text-cyan-400 font-semibold'
                                    : 'text-gray-400 hover:bg-white/5 hover:text-gray-200'}`
                            }
                        >
                            <span className='text-base w-5 text-center flex-shrink-0'>{icon}</span>
                            {!collapsed && (
                                <span className='flex-1'>{label}</span>
                            )}
                            {!collapsed && shortcut && (
                                <kbd className='text-xs px-1.5 py-0.5 rounded opacity-40
                                group-hover:opacity-70'
                                    style={{
                                        background: 'var(--bg-card)',
                                        border: '1px solid var(--border)',
                                        color: 'var(--text-muted)'
                                    }}>
                                    {shortcut}
                                </kbd>
                            )}
                        </NavLink>
                    ))}
                </nav>

                {/* Collapse toggle */}
                <button
                    onClick={() => setCollapsed(c => !c)}
                    className='mx-2 mt-2 px-3 py-2 rounded-md text-xs transition-colors
                     hover:bg-white/5'
                    style={{ color: 'var(--text-muted)' }}
                    aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
                >
                    {collapsed ? '→' : '← Collapse'}
                </button>
            </aside>

            {/* ── Main content ── */}
            <main className='flex-1 overflow-y-auto'
                style={{ background: 'var(--bg-primary)' }}>
                <Outlet />
            </main>
        </div>
    );
}
