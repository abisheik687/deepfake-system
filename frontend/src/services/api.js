import client from '../api/client';

export const authAPI = {
    /**
     * Authenticate user & get JWT token
     * POST /auth/token  (OAuth2 form-encoded)
     */
    login: async (username, password) => {
        const params = new URLSearchParams();
        params.append('username', username);
        params.append('password', password);
        const response = await client.post('/auth/token', params, {
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
        });
        return response.data;
    }
};

export const detectionsAPI = {
    /**
     * GET /api/detections/stats/summary
     * Returns: { total_detections, total_fakes, total_alerts, average_confidence }
     */
    getStats: async () => {
        const response = await client.get('/api/detections/stats/summary');
        return response.data;
    },

    /**
     * GET /api/detections/?limit=N&offset=N
     * Returns paginated scan history array
     */
    getHistory: async (params = { limit: 100, offset: 0 }) => {
        const response = await client.get('/api/detections/', { params });
        return response.data;
    },

    /**
     * GET /api/detections/:id
     * Returns full detail for a single scan
     */
    getDetection: async (id) => {
        const response = await client.get(`/api/detections/${id}`);
        return response.data;
    }
};

export const streamsAPI = {
    /**
     * GET /api/streams/ — graceful fallback if not yet implemented
     */
    getStreams: async (_active_only = true) => {
        try {
            const response = await client.get('/api/streams/');
            return response.data;
        } catch {
            // Streams endpoint not yet implemented — return empty array silently
            return [];
        }
    }
};

export const alertsAPI = {
    /**
     * GET /api/alerts/?limit=N&offset=N
     */
    getAlerts: async (params = { limit: 50, offset: 0 }) => {
        const response = await client.get('/api/alerts/', { params });
        return response.data;
    }
};

export const unifiedAPI = {
    /**
     * POST /api/scan/analyze-unified
     * Body: { data: base64, source, tier, ... }
     */
    analyzeImage: async (payload) => {
        const response = await client.post('/api/scan/analyze-unified', payload);
        return response.data;
    },

    /**
     * POST /api/scan/analyze-unified-video
     * Body: FormData (file, tier, sample_fps, max_frames)
     */
    analyzeVideo: async (formData) => {
        const response = await client.post('/api/scan/analyze-unified-video', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
        return response.data;
    },

    /**
     * POST /api/scan/live-unified
     * Body: { frame: base64 }
     */
    analyzeLiveFrame: async (payload) => {
        const response = await client.post('/api/scan/live-unified', payload);
        return response.data;
    }
};
