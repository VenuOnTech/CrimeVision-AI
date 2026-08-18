import React from 'react';
import Heatmap from '../components/Heatmap';
import GraphView from '../components/GraphView';
import IngestionZone from '../components/IngestionZone';
import { ShieldAlert, Activity, FileSearch } from 'lucide-react';

/**
 * MAIN DASHBOARD (WAR ROOM)
 * This is the central hub where all modules come together.
 */
export default function Dashboard() {
  return (
    <div className="min-h-screen bg-slate-900 p-8">
      {/* Header */}
      <header className="flex justify-between items-center mb-8 border-b border-slate-700 pb-4">
        <div className="flex items-center gap-3">
          <ShieldAlert size={32} className="text-blue-500" />
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white">CrimeVision AI</h1>
            <p className="text-sm text-slate-400">Uncertainty-Aware Investigation Decision Support</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right">
            <p className="text-sm font-bold text-white">Active Case: #2026-001</p>
            <p className="text-xs text-green-400 flex items-center gap-1 justify-end">
              <Activity size={12} /> System Online
            </p>
          </div>
          <div className="h-10 w-10 bg-slate-700 rounded-full flex items-center justify-center font-bold text-slate-300">
            INV
          </div>
        </div>
      </header>

      {/* Main Grid Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left Column: Uploads & Stats */}
        <div className="space-y-8">
          <IngestionZone />
          
          <div className="bg-slate-800 p-4 rounded-xl border border-slate-700 shadow-lg">
            <h3 className="text-lg font-bold text-slate-100 mb-4 flex items-center gap-2">
              <FileSearch size={20} className="text-blue-400"/> AI Audit Logs
            </h3>
            <ul className="space-y-3 text-sm">
              <li className="flex justify-between border-b border-slate-700 pb-2">
                <span className="text-slate-400">Entities Extracted</span>
                <span className="font-bold text-white">42</span>
              </li>
              <li className="flex justify-between border-b border-slate-700 pb-2">
                <span className="text-slate-400">Chronological Anomalies</span>
                <span className="font-bold text-yellow-400">2</span>
              </li>
              <li className="flex justify-between">
                <span className="text-slate-400">Verified Contradictions</span>
                <span className="font-bold text-red-400">1</span>
              </li>
            </ul>
          </div>
        </div>

        {/* Right Column: Research Engine Output & Knowledge Graph */}
        <div className="lg:col-span-2 space-y-8">
          {/* TRACK 1 DEMO COMPONENT */}
          <Heatmap visualUncertainty={0.85} />
          
          <GraphView />
        </div>

      </div>
    </div>
  );
}