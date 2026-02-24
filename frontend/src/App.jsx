
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import MainLayout from './layout/MainLayout';
import HomePage from './pages/HomePage';
import ImageScan from './pages/ImageScan';
import VideoScan from './pages/VideoScan';
import LiveScan from './pages/LiveScan';
import LoginPage from './pages/LoginPage';
import Dashboard from './pages/Dashboard';
import MonitorPage from './pages/MonitorPage';
import ReportsPage from './pages/ReportsPage';
import ModelsPage from './pages/ModelsPage';

const ProtectedRoute = ({ children }) => {
  const { token } = useAuth();
  if (!token) return <Navigate to="/login" replace />;
  return children;
};

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/dashboard" element={
            <ProtectedRoute>
              <MainLayout />
            </ProtectedRoute>
          }>
            <Route index element={<Dashboard />} />
            <Route path="scan/image" element={<ImageScan />} />
            <Route path="scan/video" element={<VideoScan />} />
            <Route path="scan/live" element={<LiveScan />} />
            <Route path="monitor" element={<MonitorPage />} />
            <Route path="reports" element={<ReportsPage />} />
            <Route path="models" element={<ModelsPage />} />
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;
