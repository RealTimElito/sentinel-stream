import React from 'react';

const AlertPanel = ({ alerts }) => {
  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
      <h2 className="text-xl font-semibold mb-4">Recent Alerts</h2>
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {alerts.length === 0 ? (
          <div className="text-slate-400 text-center py-8">
            No alerts detected
          </div>
        ) : (
          alerts.map(alert => (
            <div
              key={alert.id}
              className="bg-slate-900 rounded p-3 border border-red-500/50"
            >
              <div className="flex justify-between items-start mb-2">
                <span className="text-sm font-semibold text-sentinel-red">
                  ANOMALY DETECTED
                </span>
                <span className="text-xs text-slate-400">
                  {new Date(alert.timestamp).toLocaleTimeString()}
                </span>
              </div>
              <div className="text-sm text-slate-300">
                Score: {(alert.score * 100).toFixed(2)}%
              </div>
              {alert.explanation && (
                <div className="mt-2 text-xs text-slate-400">
                  Top contributing nodes identified
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default AlertPanel;

