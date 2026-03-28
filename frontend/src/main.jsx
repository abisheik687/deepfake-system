/**
 * Internal trace:
 * - Wrong before: the frontend bootstrapped a broad legacy shell that centered auth, dashboards, and realtime pages instead of the upload workflow.
 * - Fixed now: the app starts directly into the rebuilt upload/results experience and shared global styles.
 */

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './styles/globals.css';
import App from './App.jsx';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
