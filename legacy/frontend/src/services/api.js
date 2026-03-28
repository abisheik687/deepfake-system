import client from '../api/client';

export const authAPI = {
    login: async (username, password) => {
        const params = new URLSearchParams();
        params.append('username', username);
        params.append('password', password);
        const response = await client.post('/auth/token', params, {
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
        });
        if (response.status >= 400) throw new Error(response.data?.detail || 'Login failed');
        return response.data;
    }
};

export const detectionsAPI = {
    getStats: async () => {
        const response = await client.get('/api/detections/stats');
        if (response.status >= 400) throw new Error(response.data?.detail || 'Failed to fetch stats');
        return response.data;
    },

    getStatsSummary: async () => {
        const response = await client.get('/api/detections/stats/summary');
        if (response.status >= 400) throw new Error(response.data?.detail || 'Failed to fetch summary');
        return response.data;
    },

    getHistory: async (params = { limit: 100, offset: 0 }) => {
        const response = await client.get('/api/detections/', { params });
        if (response.status >= 400) throw new Error(response.data?.detail || 'Failed to fetch history');
        return response.data;
    },

    getDetection: async (id) => {
        const response = await client.get(`/api/detections/${id}`);
        if (response.status >= 400) throw new Error(response.data?.detail || 'Detection not found');
        return response.data;
    }
};

export const streamsAPI = {
    getStreams: async (_active_only = true) => {
        try {
            const response = await client.get('/api/streams/');
            return response.data;
        } catch {
            return [];
        }
    }
};

export const alertsAPI = {
    getAlerts: async ({ status, severity, verdict, page = 1, limit = 50 }) => {
        const params = { limit, offset: (page - 1) * limit };
        if (status && status !== 'All') params.status = status;
        if (severity && severity !== 'All') params.severity = severity.toLowerCase();
        if (verdict && verdict !== 'All') params.verdict = verdict.toUpperCase();
        
        const response = await client.get('/api/alerts/', { params });
        if (response.status >= 400) throw new Error(response.data?.detail || 'Failed to fetch alerts');
        return response.data;
    },
    acknowledge: async (alertId, body) => {
        const response = await client.post(`/api/alerts/${alertId}/acknowledge`, body);
        if (response.status >= 400) throw new Error(response.data?.detail || 'Failed to acknowledge');
        return response.data;
    }
};

export const unifiedAPI = {
    analyzeImage: async (payload) => {
        const response = await client.post('/api/scan/analyze-unified', payload, {
            params: payload.tier ? { tier: payload.tier, return_heatmap: payload.return_heatmap, detect_faces: payload.detect_faces } : {}
        });
        if (response.status >= 400) throw new Error(response.data?.detail || 'Analysis failed');
        return response.data;
    },

    analyzeUnifiedFile: async (file, options = {}) => {
        const formData = new FormData();
        formData.append('file', file);
        const response = await client.post('/api/scan/analyze-unified', formData, {
            params: { tier: options.tier || 'balanced', return_heatmap: options.return_heatmap ?? false, detect_faces: options.detect_faces ?? true },
            headers: { 'Content-Type': 'multipart/form-data' }
        });
        if (response.status >= 400) throw new Error(response.data?.detail || response.data?.message || 'Analysis failed');
        return response.data;
    },

    analyzeVideo: async (formData) => {
        const response = await client.post('/api/scan/analyze-unified-video', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
        if (response.status >= 400) throw new Error(response.data?.detail || 'Video analysis failed');
        return response.data;
    },

    analyzeAudio: async (file) => {
        const formData = new FormData();
        formData.append('file', file);
        const response = await client.post('/api/audio/analyze', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
        if (response.status >= 400) throw new Error(response.data?.detail || 'Audio analysis failed');
        return response.data;
    },

    analyzeLiveFrame: async (payload) => {
        const response = await client.post('/api/scan/live-unified', payload);
        if (response.status >= 400) throw new Error(response.data?.detail || 'Live analysis failed');
        return response.data;
    },

    extensionScan: async (payload) => {
        const response = await client.post('/api/scan/extension-scan', payload);
        if (response.status >= 400) throw new Error(response.data?.detail || 'URL scan failed');
        return response.data;
    }
};

export const liveVideoAPI = {
    getSessions: async () => {
        const response = await client.get('/api/live-video/sessions');
        if (response.status >= 400) throw new Error('Failed to fetch sessions');
        return response.data;
    },
    
    getSessionSummary: async (sessionId) => {
        const response = await client.get(`/api/live-video/session/${sessionId}/summary`);
        if (response.status >= 400) throw new Error('Failed to fetch session summary');
        return response.data;
    },
    
    exportSession: async (sessionId) => {
        const response = await client.post(`/api/live-video/session/${sessionId}/export`);
        if (response.status >= 400) throw new Error('Failed to export session');
        return response.data;
    }
};

export const liveAudioAPI = {
    getSessions: async () => {
        const response = await client.get('/api/live-audio/sessions');
        if (response.status >= 400) throw new Error('Failed to fetch sessions');
        return response.data;
    },
    
    getSessionSummary: async (sessionId) => {
        const response = await client.get(`/api/live-audio/session/${sessionId}/summary`);
        if (response.status >= 400) throw new Error('Failed to fetch session summary');
        return response.data;
    },
    
    getTranscript: async (sessionId) => {
        const response = await client.get(`/api/live-audio/session/${sessionId}/transcript`);
        if (response.status >= 400) throw new Error('Failed to fetch transcript');
        return response.data;
    },
    
    exportSession: async (sessionId) => {
        const response = await client.post(`/api/live-audio/session/${sessionId}/export`);
        if (response.status >= 400) throw new Error('Failed to export session');
        return response.data;
    }
};

export const interviewAPI = {
    getSessions: async () => {
        const response = await client.get('/api/interview/sessions');
        if (response.status >= 400) throw new Error('Failed to fetch sessions');
        return response.data;
    },
    
    getSessionSummary: async (sessionId) => {
        const response = await client.get(`/api/interview/session/${sessionId}/summary`);
        if (response.status >= 400) throw new Error('Failed to fetch session summary');
        return response.data;
    },
    
    generateReport: async (sessionId) => {
        const response = await client.post(`/api/interview/session/${sessionId}/report`);
        if (response.status >= 400) throw new Error('Failed to generate report');
        return response.data;
    }
};

export const socialMediaAPI = {
    scanURL: async (url, priority = 'normal') => {
        const response = await client.post('/api/social/scan', { url, priority });
        if (response.status >= 400) throw new Error(response.data?.detail || 'URL scan failed');
        return response.data;
    },
    
    getScanResult: async (scanId) => {
        const response = await client.get(`/api/social/scan/${scanId}`);
        if (response.status >= 400) throw new Error('Scan not found');
        return response.data;
    },
    
    getQueue: async () => {
        const response = await client.get('/api/social/queue');
        if (response.status >= 400) throw new Error('Failed to fetch queue');
        return response.data;
    },
    
    getPlatforms: async () => {
        const response = await client.get('/api/social/platforms');
        if (response.status >= 400) throw new Error('Failed to fetch platforms');
        return response.data;
    }
};

export const modelsAPI = {
    getModels: async () => {
        const response = await client.get('/api/models/');
        if (response.status >= 400) throw new Error('Failed to fetch models');
        return response.data;
    },
    
    getActiveModels: async () => {
        const response = await client.get('/api/models/active');
        if (response.status >= 400) throw new Error('Failed to fetch active models');
        return response.data;
    },
    
    loadModel: async (modelName) => {
        const response = await client.post('/api/models/load', { model_name: modelName });
        if (response.status >= 400) throw new Error('Failed to load model');
        return response.data;
    },
    
    benchmark: async () => {
        const response = await client.get('/api/models/benchmark');
        if (response.status >= 400) throw new Error('Failed to run benchmark');
        return response.data;
    },

    getStatus: async () => {
        const response = await client.get('/api/scan/orchestrator-status');
        if (response.status >= 400) throw new Error(response.data?.detail || 'Failed to fetch model status');
        return response.data;
    }
};

export const agencyAPI = {
    getStatus: async () => {
        const response = await client.get('/api/agency/status');
        if (response.status >= 400) throw new Error('Failed to fetch agency status');
        return response.data;
    },
    
    getLogs: async () => {
        const response = await client.get('/api/agency/logs');
        if (response.status >= 400) throw new Error('Failed to fetch logs');
        return response.data;
    },
    
    getInvestigationHistory: async () => {
        const response = await client.get('/api/agency/investigations/history');
        if (response.status >= 400) throw new Error('Failed to fetch history');
        return response.data;
    },
    
    getForensicReport: async (detectionId) => {
        const response = await client.get(`/api/agency/forensic-report/${detectionId}`);
        if (response.status >= 400) throw new Error('Failed to fetch report');
        return response.data;
    },
    
    factCheck: async (detectionId) => {
        const response = await client.post(`/api/agency/fact-check/${detectionId}`);
        if (response.status >= 400) throw new Error('Fact check failed');
        return response.data;
    },
    
    generateBriefing: async (detectionId) => {
        const response = await client.post(`/api/agency/briefing/${detectionId}`);
        if (response.status >= 400) throw new Error('Briefing generation failed');
        return response.data;
    }
};

// Legacy compatibility
export const extensionScan = unifiedAPI.extensionScan;

// Made with Bob
