/**
 * Internal trace:
 * - Wrong before: multiple clients/services disagreed on base URLs, timeout behavior, and error message shape.
 * - Fixed now: one axios client owns the API base URL, 120s timeout, and normalized frontend-friendly errors.
 */

import axios from 'axios';

const client = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 120000,
});

client.interceptors.response.use(
  (response) => response,
  (error) => {
    const normalized = new Error(
      error.response?.data?.error || error.message || 'Unknown error',
    );
    normalized.status = error.response?.status || 0;
    normalized.code = error.response?.data?.code || (error.code === 'ECONNABORTED' ? 'TIMEOUT' : 'REQUEST_FAILED');
    normalized.isTimeout = error.code === 'ECONNABORTED';
    normalized.isNetworkError = !error.response;
    throw normalized;
  },
);

export default client;
