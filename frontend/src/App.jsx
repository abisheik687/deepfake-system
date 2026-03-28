/**
 * Internal trace:
 * - Wrong before: the app routed through protected dashboards, live feeds, webcam tools, and other incomplete surfaces that distracted from the core detection flow.
 * - Fixed now: routing is limited to home, analyse, and results, all driven by one shared analysis hook.
 */

import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import Home from './pages/Home.jsx';
import Analyse from './pages/Analyse.jsx';
import Results from './pages/Results.jsx';
import { useAnalysis } from './hooks/useAnalysis.js';

function App() {
  const analysis = useAnalysis();

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/analyse" element={<Analyse analysis={analysis} />} />
        <Route path="/results" element={<Results analysis={analysis} />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
