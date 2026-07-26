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

  useEffect(() => {
    // Check API health
    axios.get(`${API_URL}/health`)
      .then(() => setIsConnected(true))
      .catch(() => setIsConnected(false));

    // WebSocket connection for real-time updates
    const ws = new WebSocket(`ws://${API_URL.replace('http://', '').replace('https://', '')}/ws`);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.is_anomaly) {
        handleAnomaly(data);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
    };

    return () => {
      ws.close();
    };
  }, []);

  const handleAnomaly = useCallback((anomalyData) => {
    const alert = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      score: anomalyData.anomaly_score,
      explanation: anomalyData.explanation
    };
    
    setAlerts(prev => [alert, ...prev].slice(0, 50)); // Keep last 50 alerts
    
    // Update stats
    setStats(prev => ({
      ...prev,
      anomalies: prev.anomalies + 1,
      totalFlows: prev.totalFlows + 1,
      anomalyRate: ((prev.anomalies + 1) / (prev.totalFlows + 1)) * 100
    }));
  }, []);

  const simulateNetworkData = useCallback(() => {
    // Simulate network topology data
    const nodes = Array.from({ length: 20 }, (_, i) => ({
      id: `node_${i}`,
      label: `192.168.1.${i + 1}`,
      group: Math.floor(Math.random() * 3),
      anomaly: Math.random() > 0.9
    }));

    const links = Array.from({ length: 30 }, () => ({
      source: `node_${Math.floor(Math.random() * 20)}`,
      target: `node_${Math.floor(Math.random() * 20)}`,
      value: Math.random() * 10
    }));

    setNetworkData({ nodes, links });
  }, []);

  useEffect(() => {
    simulateNetworkData();
    const interval = setInterval(simulateNetworkData, 5000);
    return () => clearInterval(interval);
  }, [simulateNetworkData]);

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100">
      {/* Header */}
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
          <div className="flex items-center space-x-4">
            <div className={`flex items-center space-x-2 ${isConnected ? 'text-green-400' : 'text-red-400'}`}>
              <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
              <span className="text-sm">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="p-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Network Topology */}
          <div className="lg:col-span-2">
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h2 className="text-xl font-semibold mb-4">Network Topology</h2>
              <NetworkTopology data={networkData} />
            </div>
          </div>

          {/* Sidebar */}
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

