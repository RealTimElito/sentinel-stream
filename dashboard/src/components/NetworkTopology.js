import React, { useRef, useEffect } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

const NetworkTopology = ({ data }) => {
  const fgRef = useRef();

  useEffect(() => {
    if (fgRef.current) {
      fgRef.current.d3Force('charge').strength(-300);
      fgRef.current.d3Force('link').distance(100);
    }
  }, []);

  return (
    <div className="w-full h-[600px] bg-slate-900 rounded">
      <ForceGraph2D
        ref={fgRef}
        graphData={data}
        nodeLabel={node => `${node.label}\n${node.id}`}
        nodeColor={node => {
          if (node.anomaly) {
            return '#ef4444'; // Red for anomalies
          }
          return node.group === 0 ? '#3b82f6' : node.group === 1 ? '#10b981' : '#8b5cf6';
        }}
        nodeRelSize={8}
        linkColor={() => 'rgba(255, 255, 255, 0.2)'}
        linkWidth={link => Math.sqrt(link.value)}
        onNodeHover={node => {
          if (node) {
            fgRef.current.pauseAnimation();
          } else {
            fgRef.current.resumeAnimation();
          }
        }}
        nodeCanvasObject={(node, ctx, globalScale) => {
          const label = node.label || node.id;
          const fontSize = 12 / globalScale;
          ctx.font = `${fontSize}px Sans-Serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillStyle = '#e2e8f0';
          ctx.fillText(label, node.x, node.y + 10);
        }}
        nodeCanvasObjectMode={() => 'after'}
      />
    </div>
  );
};

export default NetworkTopology;

