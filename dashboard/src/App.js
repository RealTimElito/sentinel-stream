import React, { useState, useEffect, useCallback } from 'react';
import NetworkTopology from './components/NetworkTopology';
import StatsPanel from './components/StatsPanel';
import AlertPanel from './components/AlertPanel';
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [networkData, setNetworkData] = useState({ nodes: [], links: [] });
  const [stats, setStats] = useState({
    totalFlows: 0,
    anomalies: 0,
    normal: 0,
    anomalyRate: 0
  });
  const [alerts, setAlerts] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [modelTrained, setModelTrained] = useState(false);
  const [mode, setMode] = useState('connecting');

  const applyPrediction = useCallback((data) => {
    if (data.topology) {
      setNetworkData(data.topology);
    }

    setStats((prev) => {
      const anomalies = prev.anomalies + (data.is_anomaly ? 1 : 0);
      const normal = prev.normal + (data.is_anomaly ? 0 : 1);
      const totalFlows = prev.totalFlows + 1;
      return {
        totalFlows,
        anomalies,
        normal,
        anomalyRate: totalFlows ? (anomalies / totalFlows) * 100 : 0
      };
    });

    if (data.is_anomaly) {
      setAlerts((prev) =>
        [
          {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            score: data.anomaly_score,
            explanation: data.explanation
          },
          ...prev
        ].slice(0, 50)
      );
    }
  }, []);

  const pollDemo = useCallback(async () => {
    try {
      const response = await axios.post(`${API_URL}/demo/predict`);
      setIsConnected(true);
      setMode('live');
      applyPrediction(response.data);
    } catch (error) {
      console.error('Demo predict failed', error);
      setIsConnected(false);
      setMode('offline');
    }
  }, [applyPrediction]);

  useEffect(() => {
    let cancelled = false;
    let intervalId;

    const bootstrap = async () => {
      try {
        const health = await axios.get(`${API_URL}/health`);
        if (cancelled) return;
        setIsConnected(true);
        setModelTrained(Boolean(health.data.model_trained));
        setMode('live');
        await pollDemo();
        intervalId = setInterval(pollDemo, 5000);
      } catch (error) {
        if (cancelled) return;
        console.error('API unavailable', error);
        setIsConnected(false);
        setMode('offline');
        setNetworkData({ nodes: [], links: [] });
      }
    };

    bootstrap();
    return () => {
      cancelled = true;
      if (intervalId) clearInterval(intervalId);
    };
  }, [pollDemo]);

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100">
      <header className="bg-slate-800 border-b border-slate-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h1 className="text-2xl font-bold text-sentinel-blue">
              Sentinel-Stream
            </h1>
            <span className="text-sm text-slate-400">
              Real-Time Network Anomaly Detection
            </span>
          </div>
          <div className="flex items-center space-x-4 text-sm">
            <span className="text-slate-400">
              {mode === 'live' ? 'API demo stream' : mode}
            </span>
            <span className={modelTrained ? 'text-green-400' : 'text-amber-400'}>
              {modelTrained ? 'trained model' : 'untrained weights'}
            </span>
            <div className={`flex items-center space-x-2 ${isConnected ? 'text-green-400' : 'text-red-400'}`}>
              <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
              <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
            </div>
          </div>
        </div>
      </header>

      <main className="p-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h2 className="text-xl font-semibold mb-4">Network Topology</h2>
              <NetworkTopology data={networkData} />
            </div>
          </div>
          <div className="space-y-6">
            <StatsPanel stats={stats} />
            <AlertPanel alerts={alerts} />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
