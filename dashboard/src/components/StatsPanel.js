import React from 'react';

const StatsPanel = ({ stats }) => {
  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
      <h2 className="text-xl font-semibold mb-4">Statistics</h2>
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <span className="text-slate-400">Total Flows</span>
          <span className="text-2xl font-bold text-sentinel-blue">
            {stats.totalFlows.toLocaleString()}
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-slate-400">Anomalies Detected</span>
          <span className="text-2xl font-bold text-sentinel-red">
            {stats.anomalies.toLocaleString()}
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-slate-400">Normal Flows</span>
          <span className="text-2xl font-bold text-green-400">
            {stats.normal.toLocaleString()}
          </span>
        </div>
        <div className="pt-4 border-t border-slate-700">
          <div className="flex justify-between items-center">
            <span className="text-slate-400">Anomaly Rate</span>
            <span className="text-2xl font-bold">
              {stats.anomalyRate.toFixed(2)}%
            </span>
          </div>
          <div className="mt-2 w-full bg-slate-700 rounded-full h-2">
            <div
              className="bg-sentinel-red h-2 rounded-full transition-all"
              style={{ width: `${Math.min(stats.anomalyRate, 100)}%` }}
            ></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StatsPanel;

