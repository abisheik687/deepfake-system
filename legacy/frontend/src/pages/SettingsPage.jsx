import { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { Settings, User, Bell, Key, Shield, Save, Check } from 'lucide-react';
import { motion } from 'framer-motion';

export default function SettingsPage() {
    const { user } = useAuth();
    const [activeTab, setActiveTab] = useState('profile');
    const [saved, setSaved] = useState(false);

    // Profile settings
    const [profile, setProfile] = useState({
        name: user?.username || '',
        email: user?.email || '',
        role: user?.role || 'user',
        organization: ''
    });

    // Notification settings
    const [notifications, setNotifications] = useState({
        emailAlerts: true,
        highRiskOnly: false,
        dailyDigest: true,
        soundEnabled: true,
        desktopNotifications: false
    });

    // API settings
    const [apiKey, setApiKey] = useState('');
    const [showApiKey, setShowApiKey] = useState(false);

    // Security settings
    const [security, setSecurity] = useState({
        twoFactorEnabled: false,
        sessionTimeout: 30,
        ipWhitelist: ''
    });

    useEffect(() => {
        // Load settings from localStorage or API
        const savedNotifications = localStorage.getItem('kavach_notifications');
        if (savedNotifications) {
            setNotifications(JSON.parse(savedNotifications));
        }

        const savedApiKey = localStorage.getItem('kavach_api_key');
        if (savedApiKey) {
            setApiKey(savedApiKey);
        }
    }, []);

    const handleSave = () => {
        // Save to localStorage (in production, this would be an API call)
        localStorage.setItem('kavach_notifications', JSON.stringify(notifications));
        if (apiKey) {
            localStorage.setItem('kavach_api_key', apiKey);
        }

        setSaved(true);
        setTimeout(() => setSaved(false), 3000);
    };

    const generateApiKey = () => {
        const key = 'kvch_' + Array.from({ length: 32 }, () => 
            Math.random().toString(36)[2] || '0'
        ).join('');
        setApiKey(key);
    };

    const tabs = [
        { id: 'profile', label: 'Profile', icon: User },
        { id: 'notifications', label: 'Notifications', icon: Bell },
        { id: 'api', label: 'API Keys', icon: Key },
        { id: 'security', label: 'Security', icon: Shield }
    ];

    return (
        <div className='p-6 max-w-7xl mx-auto'>
            {/* Header */}
            <div className='mb-6'>
                <h1 className='text-2xl font-bold mb-2' style={{ color: 'var(--text-primary)' }}>
                    Settings
                </h1>
                <p className='text-sm' style={{ color: 'var(--text-muted)' }}>
                    Manage your account preferences and configuration
                </p>
            </div>

            <div className='grid grid-cols-1 lg:grid-cols-4 gap-6'>
                {/* Sidebar Tabs */}
                <div className='lg:col-span-1'>
                    <div className='rounded-xl overflow-hidden' style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                        {tabs.map((tab) => {
                            const Icon = tab.icon;
                            const isActive = activeTab === tab.id;
                            return (
                                <button
                                    key={tab.id}
                                    onClick={() => setActiveTab(tab.id)}
                                    className='w-full flex items-center gap-3 px-4 py-3 text-left transition-all'
                                    style={{
                                        background: isActive ? 'var(--bg-hover)' : 'transparent',
                                        borderLeft: isActive ? '3px solid var(--cyan)' : '3px solid transparent',
                                        color: isActive ? 'var(--text-primary)' : 'var(--text-muted)'
                                    }}
                                >
                                    <Icon size={18} />
                                    <span className='text-sm font-semibold'>{tab.label}</span>
                                </button>
                            );
                        })}
                    </div>
                </div>

                {/* Content Area */}
                <div className='lg:col-span-3'>
                    <div className='rounded-xl p-6' style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                        {/* Profile Tab */}
                        {activeTab === 'profile' && (
                            <motion.div
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className='space-y-6'
                            >
                                <div>
                                    <h2 className='text-lg font-bold mb-4' style={{ color: 'var(--text-primary)' }}>
                                        Profile Information
                                    </h2>
                                </div>

                                <div className='grid grid-cols-1 md:grid-cols-2 gap-4'>
                                    <div>
                                        <label className='block text-xs font-semibold uppercase tracking-wider mb-2' style={{ color: 'var(--text-secondary)' }}>
                                            Full Name
                                        </label>
                                        <input
                                            type='text'
                                            value={profile.name}
                                            onChange={(e) => setProfile({ ...profile, name: e.target.value })}
                                            className='w-full px-4 py-2.5 rounded-lg text-sm'
                                            style={{
                                                background: 'var(--bg-hover)',
                                                border: '1px solid var(--border)',
                                                color: 'var(--text-primary)',
                                                outline: 'none'
                                            }}
                                        />
                                    </div>

                                    <div>
                                        <label className='block text-xs font-semibold uppercase tracking-wider mb-2' style={{ color: 'var(--text-secondary)' }}>
                                            Email Address
                                        </label>
                                        <input
                                            type='email'
                                            value={profile.email}
                                            onChange={(e) => setProfile({ ...profile, email: e.target.value })}
                                            className='w-full px-4 py-2.5 rounded-lg text-sm'
                                            style={{
                                                background: 'var(--bg-hover)',
                                                border: '1px solid var(--border)',
                                                color: 'var(--text-primary)',
                                                outline: 'none'
                                            }}
                                        />
                                    </div>

                                    <div>
                                        <label className='block text-xs font-semibold uppercase tracking-wider mb-2' style={{ color: 'var(--text-secondary)' }}>
                                            Role
                                        </label>
                                        <input
                                            type='text'
                                            value={profile.role}
                                            disabled
                                            className='w-full px-4 py-2.5 rounded-lg text-sm opacity-50 cursor-not-allowed'
                                            style={{
                                                background: 'var(--bg-hover)',
                                                border: '1px solid var(--border)',
                                                color: 'var(--text-primary)'
                                            }}
                                        />
                                    </div>

                                    <div>
                                        <label className='block text-xs font-semibold uppercase tracking-wider mb-2' style={{ color: 'var(--text-secondary)' }}>
                                            Organization
                                        </label>
                                        <input
                                            type='text'
                                            value={profile.organization}
                                            onChange={(e) => setProfile({ ...profile, organization: e.target.value })}
                                            placeholder='Optional'
                                            className='w-full px-4 py-2.5 rounded-lg text-sm'
                                            style={{
                                                background: 'var(--bg-hover)',
                                                border: '1px solid var(--border)',
                                                color: 'var(--text-primary)',
                                                outline: 'none'
                                            }}
                                        />
                                    </div>
                                </div>
                            </motion.div>
                        )}

                        {/* Notifications Tab */}
                        {activeTab === 'notifications' && (
                            <motion.div
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className='space-y-6'
                            >
                                <div>
                                    <h2 className='text-lg font-bold mb-1' style={{ color: 'var(--text-primary)' }}>
                                        Notification Preferences
                                    </h2>
                                    <p className='text-sm' style={{ color: 'var(--text-muted)' }}>
                                        Configure how you receive alerts and updates
                                    </p>
                                </div>

                                <div className='space-y-4'>
                                    {[
                                        { key: 'emailAlerts', label: 'Email Alerts', desc: 'Receive email notifications for new detections' },
                                        { key: 'highRiskOnly', label: 'High Risk Only', desc: 'Only notify for high-risk deepfakes' },
                                        { key: 'dailyDigest', label: 'Daily Digest', desc: 'Receive a daily summary of all detections' },
                                        { key: 'soundEnabled', label: 'Sound Alerts', desc: 'Play sound when deepfake is detected' },
                                        { key: 'desktopNotifications', label: 'Desktop Notifications', desc: 'Show browser notifications' }
                                    ].map(({ key, label, desc }) => (
                                        <div key={key} className='flex items-center justify-between p-4 rounded-lg' style={{ background: 'var(--bg-hover)' }}>
                                            <div>
                                                <h3 className='text-sm font-semibold mb-1' style={{ color: 'var(--text-primary)' }}>
                                                    {label}
                                                </h3>
                                                <p className='text-xs' style={{ color: 'var(--text-muted)' }}>
                                                    {desc}
                                                </p>
                                            </div>
                                            <button
                                                onClick={() => setNotifications({ ...notifications, [key]: !notifications[key] })}
                                                className='relative w-12 h-6 rounded-full transition-all'
                                                style={{ background: notifications[key] ? 'var(--cyan)' : 'var(--border)' }}
                                            >
                                                <div
                                                    className='absolute top-1 w-4 h-4 bg-white rounded-full transition-all'
                                                    style={{ left: notifications[key] ? '26px' : '4px' }}
                                                />
                                            </button>
                                        </div>
                                    ))}
                                </div>
                            </motion.div>
                        )}

                        {/* API Keys Tab */}
                        {activeTab === 'api' && (
                            <motion.div
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className='space-y-6'
                            >
                                <div>
                                    <h2 className='text-lg font-bold mb-1' style={{ color: 'var(--text-primary)' }}>
                                        API Keys
                                    </h2>
                                    <p className='text-sm' style={{ color: 'var(--text-muted)' }}>
                                        Manage API keys for programmatic access
                                    </p>
                                </div>

                                <div className='p-4 rounded-lg' style={{ background: 'var(--bg-hover)', border: '1px solid var(--border)' }}>
                                    <label className='block text-xs font-semibold uppercase tracking-wider mb-2' style={{ color: 'var(--text-secondary)' }}>
                                        Your API Key
                                    </label>
                                    <div className='flex gap-2'>
                                        <input
                                            type={showApiKey ? 'text' : 'password'}
                                            value={apiKey}
                                            readOnly
                                            placeholder='No API key generated'
                                            className='flex-1 px-4 py-2.5 rounded-lg text-sm font-mono'
                                            style={{
                                                background: 'var(--bg-card)',
                                                border: '1px solid var(--border)',
                                                color: 'var(--text-primary)'
                                            }}
                                        />
                                        <button
                                            onClick={() => setShowApiKey(!showApiKey)}
                                            className='px-4 py-2.5 rounded-lg text-sm font-semibold'
                                            style={{ background: 'var(--bg-card)', color: 'var(--text-primary)', border: '1px solid var(--border)' }}
                                        >
                                            {showApiKey ? 'Hide' : 'Show'}
                                        </button>
                                        <button
                                            onClick={generateApiKey}
                                            className='px-4 py-2.5 rounded-lg text-sm font-semibold'
                                            style={{ background: 'var(--cyan)', color: '#0A1628' }}
                                        >
                                            Generate New
                                        </button>
                                    </div>
                                </div>

                                <div className='p-4 rounded-lg' style={{ background: '#2c1f0d', border: '1px solid #F4A26133' }}>
                                    <h3 className='text-sm font-semibold mb-2' style={{ color: '#F4A261' }}>
                                        ⚠️ Security Warning
                                    </h3>
                                    <p className='text-xs' style={{ color: 'var(--text-muted)' }}>
                                        Keep your API key secure. Do not share it publicly or commit it to version control.
                                        If compromised, generate a new key immediately.
                                    </p>
                                </div>

                                <div>
                                    <h3 className='text-sm font-semibold mb-3' style={{ color: 'var(--text-secondary)' }}>
                                        API Usage Example
                                    </h3>
                                    <pre className='p-4 rounded-lg text-xs font-mono overflow-x-auto' style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}>
{`curl -X POST http://localhost:8000/api/scan/analyze-unified \\
  -H "Authorization: Bearer ${apiKey || 'YOUR_API_KEY'}" \\
  -F "file=@image.jpg"`}
                                    </pre>
                                </div>
                            </motion.div>
                        )}

                        {/* Security Tab */}
                        {activeTab === 'security' && (
                            <motion.div
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className='space-y-6'
                            >
                                <div>
                                    <h2 className='text-lg font-bold mb-1' style={{ color: 'var(--text-primary)' }}>
                                        Security Settings
                                    </h2>
                                    <p className='text-sm' style={{ color: 'var(--text-muted)' }}>
                                        Configure security and access controls
                                    </p>
                                </div>

                                <div className='flex items-center justify-between p-4 rounded-lg' style={{ background: 'var(--bg-hover)' }}>
                                    <div>
                                        <h3 className='text-sm font-semibold mb-1' style={{ color: 'var(--text-primary)' }}>
                                            Two-Factor Authentication
                                        </h3>
                                        <p className='text-xs' style={{ color: 'var(--text-muted)' }}>
                                            Add an extra layer of security to your account
                                        </p>
                                    </div>
                                    <button
                                        onClick={() => setSecurity({ ...security, twoFactorEnabled: !security.twoFactorEnabled })}
                                        className='relative w-12 h-6 rounded-full transition-all'
                                        style={{ background: security.twoFactorEnabled ? 'var(--cyan)' : 'var(--border)' }}
                                    >
                                        <div
                                            className='absolute top-1 w-4 h-4 bg-white rounded-full transition-all'
                                            style={{ left: security.twoFactorEnabled ? '26px' : '4px' }}
                                        />
                                    </button>
                                </div>

                                <div>
                                    <label className='block text-xs font-semibold uppercase tracking-wider mb-2' style={{ color: 'var(--text-secondary)' }}>
                                        Session Timeout (minutes)
                                    </label>
                                    <input
                                        type='number'
                                        value={security.sessionTimeout}
                                        onChange={(e) => setSecurity({ ...security, sessionTimeout: parseInt(e.target.value) })}
                                        min='5'
                                        max='1440'
                                        className='w-full px-4 py-2.5 rounded-lg text-sm'
                                        style={{
                                            background: 'var(--bg-hover)',
                                            border: '1px solid var(--border)',
                                            color: 'var(--text-primary)',
                                            outline: 'none'
                                        }}
                                    />
                                </div>

                                <div>
                                    <label className='block text-xs font-semibold uppercase tracking-wider mb-2' style={{ color: 'var(--text-secondary)' }}>
                                        IP Whitelist (comma-separated)
                                    </label>
                                    <textarea
                                        value={security.ipWhitelist}
                                        onChange={(e) => setSecurity({ ...security, ipWhitelist: e.target.value })}
                                        placeholder='192.168.1.1, 10.0.0.1'
                                        rows={3}
                                        className='w-full px-4 py-2.5 rounded-lg text-sm font-mono'
                                        style={{
                                            background: 'var(--bg-hover)',
                                            border: '1px solid var(--border)',
                                            color: 'var(--text-primary)',
                                            outline: 'none',
                                            resize: 'vertical'
                                        }}
                                    />
                                </div>
                            </motion.div>
                        )}

                        {/* Save Button */}
                        <div className='mt-6 flex items-center justify-end gap-3'>
                            {saved && (
                                <motion.div
                                    initial={{ opacity: 0, x: 10 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    className='flex items-center gap-2 text-sm font-semibold'
                                    style={{ color: '#2DC653' }}
                                >
                                    <Check size={16} />
                                    Settings saved successfully
                                </motion.div>
                            )}
                            <button
                                onClick={handleSave}
                                className='flex items-center gap-2 px-6 py-2.5 rounded-lg text-sm font-bold transition-all'
                                style={{ background: 'var(--cyan)', color: '#0A1628' }}
                            >
                                <Save size={18} />
                                Save Changes
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

// Made with Bob
