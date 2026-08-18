import React from 'react';
import ReactFlow, { Background, Controls } from 'reactflow';
import 'reactflow/dist/style.css';

/**
 * GRAPH VIEW COMPONENT (MODULE 8)
 * Renders the relationship graph pulled from Neo4j.
 */
const initialNodes = [
  { id: '1', position: { x: 250, y: 50 }, data: { label: 'Suspect: John Doe' }, style: { background: '#1e293b', color: '#fff', border: '1px solid #3b82f6' } },
  { id: '2', position: { x: 100, y: 150 }, data: { label: 'Vehicle: Black SUV' }, style: { background: '#1e293b', color: '#fff', border: '1px solid #64748b' } },
  { id: '3', position: { x: 400, y: 150 }, data: { label: 'Location: 5th Ave' }, style: { background: '#1e293b', color: '#fff', border: '1px solid #64748b' } },
];

const initialEdges = [
  { id: 'e1-2', source: '1', target: '2', label: 'OWNS', animated: true, style: { stroke: '#3b82f6' } },
  { id: 'e1-3', source: '1', target: '3', label: 'SPOTTED_AT', style: { stroke: '#ef4444' } }, // Red edge indicates contradiction
];

export default function GraphView() {
  return (
    <div className="bg-slate-800 p-4 rounded-xl border border-slate-700 shadow-lg h-96 flex flex-col">
      <h3 className="text-lg font-bold text-slate-100 mb-2">Evidence Relationship Graph</h3>
      <div className="flex-grow rounded-lg overflow-hidden border border-slate-600">
        <ReactFlow nodes={initialNodes} edges={initialEdges} fitView>
          <Background color="#334155" gap={16} />
          <Controls className="bg-slate-700 fill-white" />
        </ReactFlow>
      </div>
    </div>
  );
}