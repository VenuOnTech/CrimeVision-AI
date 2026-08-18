import React from 'react';

/**
 * HEATMAP COMPONENT (MODULE 6 & 9)
 * Visually renders the output of the Evidential Optimal Transport PyTorch script.
 * Demonstrates to the faculty how visual uncertainty dampens contradiction scores.
 */
export default function Heatmap({ data, visualUncertainty }) {
  // Dummy tokens and patches for the August 17 Mockup
  const tokens = ["Suspect", "wore", "a", "red", "jacket"];
  const patches = ["Patch 1", "Patch 2", "Patch 3"];

  // If data is null, use fallback mockup data
  const matrix = data || [
    [0.1, 0.1, 0.1, 0.8, 0.9],
    [0.2, 0.1, 0.1, 0.2, 0.3],
    [0.1, 0.2, 0.1, 0.1, 0.1],
  ];

  // Dynamic styling based on Track 1 Uncertainty Math
  const isHighUncertainty = visualUncertainty > 0.5;
  const badgeColor = isHighUncertainty ? "bg-yellow-500" : "bg-red-500";
  const badgeText = isHighUncertainty ? "High Visual Doubt (Penalty Dampened)" : "High Confidence Contradiction";

  return (
    <div className="bg-slate-800 p-4 rounded-xl border border-slate-700 shadow-lg">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-bold text-slate-100">Evidential OT Alignment</h3>
        <span className={`px-3 py-1 rounded-full text-xs font-bold text-white ${badgeColor}`}>
          {badgeText}
        </span>
      </div>
      
      <p className="text-sm text-slate-400 mb-4">
        Matrix maps witness statement tokens against CCTV visual patches. 
        Uncertainty Score: <strong className="text-white">{(visualUncertainty * 100).toFixed(1)}%</strong>
      </p>

      <div className="overflow-x-auto">
        <table className="w-full text-center border-collapse">
          <thead>
            <tr>
              <th className="p-2 border border-slate-600 text-slate-300"></th>
              {tokens.map((token, i) => (
                <th key={i} className="p-2 border border-slate-600 font-mono text-sm text-blue-400">{token}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={i}>
                <td className="p-2 border border-slate-600 font-mono text-sm text-slate-300">{patches[i]}</td>
                {row.map((val, j) => {
                  // Color intensity based on mathematical alignment score
                  const intensity = Math.floor(val * 255);
                  const cellColor = isHighUncertainty 
                    ? `rgba(234, 179, 8, ${val})` // Yellow for uncertain
                    : `rgba(239, 68, 68, ${val})`; // Red for confident contradiction

                  return (
                    <td 
                      key={j} 
                      className="p-2 border border-slate-600 text-slate-900 font-bold"
                      style={{ backgroundColor: cellColor }}
                    >
                      {val.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}