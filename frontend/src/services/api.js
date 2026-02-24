import client from '../api/client';

export const authAPI = {
    /**
     * Authenticate user & get JWT token
     * @param {string} username 
     * @param {string} password 
     * @returns {Promise<{access_token: string, token_type: string}>}
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
     * Get aggregate statistics for detections
     * @returns {Promise<{total_detections: number, total_alerts: number, severity_distribution: any, average_confidence: number}>}
     */
    getStats: async () => {
        const response = await client.get('/api/detections/stats/summary');
        return response.data;
    },

    /**
     * Get paginated history of detections
     * @param {Object} params - Query parameters (limit, offset)
     * @returns {Promise<Array<any>>}
     */
    getHistory: async (params = { limit: 100, offset: 0 }) => {
        const response = await client.get('/api/detections/', { params });
        return response.data;
    },

    /**
     * Get detailed information for a single detection
     * @param {number} id 
     * @returns {Promise<any>}
     */
    getDetection: async (id) => {
        const response = await client.get(`/api/detections/${id}`);
        return response.data;
    }
};

export const streamsAPI = {
    /**
     * Get active and/or inactive streams
     * @param {boolean} active_only 
     * @returns {Promise<Array<any>>}
     */
    getStreams: async (active_only = true) => {
        const response = await client.get('/api/streams/', { params: { active_only } });
        return response.data;
    }
};

export const alertsAPI = {
    /**
     * Get alerts
     * @param {Object} params - Query parameters
     * @returns {Promise<Array<any>>}
     */
    getAlerts: async (params = { limit: 50, offset: 0 }) => {
        const response = await client.get('/api/alerts/', { params });
        return response.data;
    }
};
