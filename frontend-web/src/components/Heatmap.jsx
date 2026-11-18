import React, { useState } from 'react';
import { Layers, Info } from 'lucide-react';

const VALID_ASSESSMENTS = new Set(['MATCH', 'CONTRADICTION', 'INDETERMINATE']);

const STATUS_STYLES = {
  MATCH: 'bg-emerald-950/80 text-emerald-300 border-emerald-700',
  CONTRADICTION: 'bg-red-950/80 text-red-300 border-red-700',
  INDETERMINATE: 'bg-slate-900 text-slate-300 border-slate-600',
};

export default function Heatmap({
  data = [],
  tokens = [],
  patches = [],
  assessmentStatus = 'INDETERMINATE',
  evidenceName,
  cost,
  modelEpistemicUncertainty,
  visualQualityMetrics,
}) {
  const [hoveredCell, setHoveredCell] = useState(null);

  // Legacy / malformed / lowercase assessment values fail closed to
  // INDETERMINATE, matching EvidenceTable and EvidenceModal. Never convert
  // an unknown status into MATCH.
  const rawStatus = String(assessmentStatus || '').toUpperCase();
  const normalizedStatus = VALID_ASSESSMENTS.has(rawStatus) ? rawStatus : 'INDETERMINATE';

  const hasData = Array.isArray(data) && data.length > 0 && Array.isArray(tokens) && tokens.length > 0;
  const statusStyle = STATUS_STYLES[normalizedStatus];

  const qualitySummary = visualQualityMetrics
    ? `HF residual ${Number(visualQualityMetrics.mean_absolute_high_frequency_residual).toFixed(2)} · Laplacian variance ${Number(visualQualityMetrics.laplacian_variance).toFixed(2)}`
    : null;

  return (
    <div className="bg-slate-800/95 backdrop-blur-md p-5 rounded-2xl border border-slate-700/80 shadow-2xl flex flex-col space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div className="p-1.5 bg-blue-500/10 border border-blue-500/30 rounded-lg text-blue-400">
            <Layers size={18} />
          </div>
          <div>
            <h3 className="text-base font-black text-slate-100">Projected Patch-to-Token Dissimilarity Matrix</h3>
            <p className="text-[11px] text-slate-400 font-mono">
              {evidenceName ? `Evidence: ${evidenceName}` : 'Awaiting evidence analysis'}
            </p>
          </div>
        </div>
        <span className={`px-3 py-1 rounded-full border text-xs font-bold ${statusStyle}`}>
          {normalizedStatus}
        </span>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-900/80 p-3 rounded-xl border border-slate-700/60 text-xs text-slate-300">
        <div className="flex items-center gap-2 text-slate-400">
          <Info size={14} />
          <span>Exploratory CLIP sequence-feature dissimilarities; not supervised region-word grounding, object detections or verdicts.</span>
        </div>
        <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11px]">
          {typeof cost === 'number' && (
            <span>Debiased Sinkhorn divergence: <strong className="text-blue-300">{cost.toFixed(6)}</strong></span>
          )}
          {typeof modelEpistemicUncertainty === 'number' && (
            <span>Model uncertainty: <strong className="text-amber-300">{(modelEpistemicUncertainty * 100).toFixed(1)}%</strong></span>
          )}
          {qualitySummary && <span className="text-slate-400">Image quality: {qualitySummary}</span>}
        </div>
      </div>

      {!hasData ? (
        <div className="rounded-xl border border-slate-700/80 bg-slate-950 p-8 text-center space-y-2">
          <Layers size={30} className="mx-auto text-slate-500" />
          <p className="text-sm font-bold text-slate-300">No stored patch/token matrix</p>
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            A matrix is shown immediately after live analysis. Older graph records do not contain the original matrix, so it is not reconstructed or fabricated.
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-slate-700/80 bg-slate-950">
          <table className="w-full text-center border-collapse text-xs">
            <thead>
              <tr className="bg-slate-900/90 font-mono text-[11px] text-slate-400">
                <th className="p-2.5 border border-slate-700/80 text-left font-semibold">Visual patch / text token</th>
                {tokens.map((token, index) => (
                  <th key={`${token}-${index}`} className="p-2.5 border border-slate-700/80 font-bold text-blue-300 bg-slate-800/60">
                    {token}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map((row, rowIndex) => (
                <tr key={rowIndex} className="hover:bg-slate-800/30 transition-colors">
                  <td className="p-2.5 border border-slate-700/80 font-mono text-left text-slate-300 bg-slate-900/60 font-semibold">
                    {patches[rowIndex] || `Patch ${rowIndex + 1}`}
                  </td>
                  {row.map((rawValue, columnIndex) => {
                    const value = Number(rawValue);
                    const safeValue = Number.isFinite(value) ? value : 0;
                    const intensity = Math.min(0.75, Math.max(0.12, safeValue / 2.0));
                    const isHovered = hoveredCell?.row === rowIndex && hoveredCell?.col === columnIndex;
                    return (
                      <td
                        key={columnIndex}
                        onMouseEnter={() => setHoveredCell({ row: rowIndex, col: columnIndex, value: safeValue, token: tokens[columnIndex], patch: patches[rowIndex] })}
                        onMouseLeave={() => setHoveredCell(null)}
                        className={`p-2.5 border border-slate-700/80 text-slate-100 font-mono font-bold cursor-pointer transition-all ${isHovered ? 'ring-2 ring-blue-400 scale-105 z-10' : ''}`}
                        style={{ backgroundColor: `rgba(56, 189, 248, ${intensity})` }}
                      >
                        {safeValue.toFixed(3)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {hoveredCell && (
        <div className="p-2 bg-slate-900 border border-slate-700 rounded-lg text-[11px] text-slate-300 flex flex-wrap items-center justify-between gap-2 font-mono">
          <span>Patch: <strong className="text-white">{hoveredCell.patch || `Patch ${hoveredCell.row + 1}`}</strong> ↔ Token: <strong className="text-blue-300">{hoveredCell.token}</strong></span>
          <span>Cosine dissimilarity: <strong className="text-blue-300">{hoveredCell.value.toFixed(6)}</strong></span>
        </div>
      )}
    </div>
  );
}