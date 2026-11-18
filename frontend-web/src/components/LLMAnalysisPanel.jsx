import React, { useState } from 'react';
import {
  Sparkles,
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  Copy,
  Check,
  Info,
  Cpu,
  Database,
  HardDrive,
} from 'lucide-react';

const STATUS_STYLES = {
  MATCH: {
    panel: 'bg-emerald-950/40 border-emerald-800/80 text-emerald-300',
    icon: 'bg-emerald-500/20 text-emerald-400',
    text: 'text-emerald-400',
    label: 'MATCH',
  },
  CONTRADICTION: {
    panel: 'bg-red-950/40 border-red-800/80 text-red-300',
    icon: 'bg-red-500/20 text-red-400',
    text: 'text-red-400',
    label: 'CONTRADICTION',
  },
  INDETERMINATE: {
    panel: 'bg-slate-900/80 border-slate-700/80 text-slate-300',
    icon: 'bg-slate-700/50 text-slate-300',
    text: 'text-slate-300',
    label: 'INDETERMINATE',
  },
};

const displayNumber = (value, digits = 3) => (
  typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : 'N/A'
);

export default function LLMAnalysisPanel({
  evidenceName,
  statementText,
  cost,
  assessmentStatus = 'INDETERMINATE',
  modelEpistemicUncertainty,
  visualQualityMetrics,
  auditReport,
  investigatorAdvisory,
  graphPersistenceStatus,
  caseId = 'CASE-1',
  caseName = '',
  crossCaseLinks = [],
  isMultiCaseView = false,
}) {
  const [copied, setCopied] = useState(false);
  const status = STATUS_STYLES[assessmentStatus] ? assessmentStatus : 'INDETERMINATE';
  const style = STATUS_STYLES[status];
  const hasAnalysis = Boolean(evidenceName || auditReport || statementText || typeof cost === 'number');
  const relevantLinks = crossCaseLinks.filter(
    (link) => link.source_case_id === caseId || link.target_case_id === caseId
  );

  const reportText = auditReport || (
    'No audit report was returned by the analysis service. The interface does not infer a physical finding or a MATCH/CONTRADICTION conclusion.'
  );
  const advisoryText = investigatorAdvisory || '';
  const persistenceMessage = graphPersistenceStatus?.message || 'Persistence status has not been returned.';
  const hasDurablePersistence = graphPersistenceStatus?.persisted === true
    || (graphPersistenceStatus?.backend === 'neo4j' && graphPersistenceStatus?.durable === true);
  const isMemoryOnly = graphPersistenceStatus?.backend === 'memory'
    || graphPersistenceStatus?.durable === false;

  const handleCopy = async () => {
    const summary = [
      'CRIMEVISION AI — RESEARCH ANALYSIS SUMMARY',
      `Case: ${caseName || caseId}`,
      `Evidence: ${evidenceName || 'N/A'}`,
      `Witness statement: ${statementText || 'N/A'}`,
      `Classifier assessment: ${status}`,
      `Debiased Sinkhorn divergence (diagnostic only): ${displayNumber(cost, 6)}`,
      `Model epistemic uncertainty: ${displayNumber(modelEpistemicUncertainty, 6)}`,
      `Image quality diagnostics: ${visualQualityMetrics ? JSON.stringify(visualQualityMetrics) : 'Unavailable'}`,
      `Graph persistence: ${persistenceMessage}`,
      `Investigator advisory: ${advisoryText || 'Human review remains required.'}`,
      `Cross-case links: ${relevantLinks.length}`,
      '',
      'Analysis service report:',
      reportText,
      '',
      'This is an automated research output, not an independently verified physical finding or legal conclusion.',
    ].join('\n');

    try {
      await navigator.clipboard.writeText(summary);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch (error) {
      console.error('Could not copy analysis summary:', error);
    }
  };

  return (
    <div className="bg-slate-800/95 backdrop-blur-md p-6 rounded-2xl border border-slate-700/80 shadow-2xl space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-700/60">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-purple-500/10 border border-purple-500/30 rounded-xl text-purple-400">
            <Sparkles size={22} />
          </div>
          <div>
            <h3 className="text-lg font-black text-slate-100">Evidential Analysis Summary</h3>
            <p className="text-xs text-slate-400">Model outputs and diagnostics for investigator review</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {!isMultiCaseView && relevantLinks.length > 0 && (
            <span className="text-xs px-2.5 py-1 rounded-xl bg-amber-950/80 text-amber-300 border border-amber-800 font-mono font-bold">
              {relevantLinks.length} Cross-Case Links
            </span>
          )}
          <span className="text-xs font-mono text-purple-300 bg-purple-950/80 px-3 py-1 rounded-xl border border-purple-800 font-bold truncate max-w-[200px]">
            {caseName || `Case: ${caseId}`}
          </span>
          <button
            type="button"
            onClick={handleCopy}
            disabled={!hasAnalysis}
            className="flex items-center gap-1.5 text-xs font-semibold px-3 py-1 rounded-xl bg-slate-900 hover:bg-slate-700 disabled:opacity-50 text-slate-300 border border-slate-700 transition-colors shadow"
          >
            {copied ? <Check size={13} className="text-emerald-400" /> : <Copy size={13} className="text-slate-400" />}
            <span>{copied ? 'Copied' : 'Copy Summary'}</span>
          </button>
        </div>
      </div>

      {isMultiCaseView ? (
        <div className="py-8 text-center text-slate-500 space-y-2">
          <Database size={36} className="mx-auto mb-1 opacity-40 text-purple-400" />
          <p className="text-sm font-bold text-slate-300">Multi-case graph view</p>
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            A global graph does not have one active evidence assessment. Select an individual case and evidence record to inspect its analysis.
          </p>
        </div>
      ) : !hasAnalysis ? (
        <div className="py-8 text-center text-slate-500 space-y-2">
          <Cpu size={36} className="mx-auto mb-1 opacity-40 text-purple-400" />
          <p className="text-sm font-bold text-slate-300">Awaiting Evidence Analysis</p>
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            Submit a genuine image/video and witness statement to display the model assessment and alignment diagnostics.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className={`p-3.5 rounded-xl border flex items-center gap-3 ${style.panel}`}>
              <div className={`p-2 rounded-lg ${style.icon}`}>
                {status === 'CONTRADICTION'
                  ? <AlertTriangle size={20} />
                  : status === 'MATCH'
                    ? <CheckCircle2 size={20} />
                    : <HelpCircle size={20} />}
              </div>
              <div>
                <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider">Classifier Assessment</span>
                <strong className={`text-sm font-black ${style.text}`}>{style.label}</strong>
              </div>
            </div>

            <div className="bg-slate-900/80 p-3.5 rounded-xl border border-slate-700/60 flex items-center justify-between">
              <div>
                <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider">Debiased Sinkhorn Divergence</span>
                <p className="text-xs text-slate-500 font-mono mt-0.5">Alignment diagnostic only</p>
              </div>
              <span className="font-mono text-base font-black px-2.5 py-1 rounded-lg border bg-slate-800 text-blue-300 border-slate-700">
                {displayNumber(cost, 6)}
              </span>
            </div>

            <div className="p-3.5 rounded-xl border bg-slate-900/80 border-slate-700/60 text-slate-300 flex items-center justify-between">
              <div>
                <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider">Model Epistemic Uncertainty</span>
                <p className="text-xs text-slate-500 font-mono mt-0.5">From trained evidential head</p>
              </div>
              <span className="font-mono text-base font-black px-2.5 py-1 rounded-lg border bg-slate-800 text-amber-300 border-slate-700">
                {typeof modelEpistemicUncertainty === 'number' && Number.isFinite(modelEpistemicUncertainty)
                  ? `${(modelEpistemicUncertainty * 100).toFixed(1)}%`
                  : 'Unavailable'}
              </span>
            </div>
          </div>

          <div className="rounded-xl border border-slate-700/70 bg-slate-900/70 p-4 space-y-2">
            <div className="flex items-center justify-between gap-2">
              <h4 className="text-sm font-bold text-slate-200">Evidence and statement</h4>
              <span className="text-[10px] text-slate-500 font-mono">{evidenceName || 'Evidence file unavailable'}</span>
            </div>
            <p className="text-xs text-slate-300 whitespace-pre-wrap">{statementText || 'No witness statement recorded.'}</p>
          </div>

          {visualQualityMetrics && (
            <div className="rounded-xl border border-slate-700/70 bg-slate-900/70 p-4">
              <h4 className="text-sm font-bold text-slate-200 mb-2">Image-quality diagnostics (not epistemic uncertainty)</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs font-mono text-slate-300">
                <span>Width: {visualQualityMetrics.width ?? 'N/A'}</span>
                <span>Height: {visualQualityMetrics.height ?? 'N/A'}</span>
                <span>HF residual: {displayNumber(visualQualityMetrics.mean_absolute_high_frequency_residual)}</span>
                <span>Laplacian variance: {displayNumber(visualQualityMetrics.laplacian_variance)}</span>
              </div>
            </div>
          )}

          {advisoryText && (
            <div className="rounded-xl border border-amber-800/60 bg-amber-950/20 p-4 space-y-1">
              <h4 className="text-sm font-bold text-amber-200">Investigator advisory</h4>
              <p className="text-xs text-amber-100 whitespace-pre-wrap">{advisoryText}</p>
            </div>
          )}

          <div className={`rounded-xl border p-3 flex items-start gap-2 text-xs ${
            hasDurablePersistence
              ? 'border-emerald-800/70 bg-emerald-950/20 text-emerald-200'
              : isMemoryOnly
                ? 'border-amber-800/70 bg-amber-950/20 text-amber-200'
                : 'border-slate-700 bg-slate-900/70 text-slate-300'
          }`}>
            {hasDurablePersistence ? <Database size={15} className="mt-0.5 shrink-0" /> : <HardDrive size={15} className="mt-0.5 shrink-0" />}
            <p><strong>Graph persistence:</strong> {persistenceMessage}</p>
          </div>

          <div className="rounded-xl border border-slate-700/70 bg-slate-900/70 p-4 space-y-2">
            <h4 className="text-sm font-bold text-slate-200">Analysis service report</h4>
            <p className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap">{reportText}</p>
          </div>

          <div className="flex items-start gap-2 rounded-xl border border-amber-800/60 bg-amber-950/20 p-3 text-xs text-amber-200">
            <Info size={15} className="mt-0.5 shrink-0" />
            <p>These outputs are research decision-support signals. They do not establish what physically occurred and must not be presented as forensic proof or a court-admissible conclusion without independent validation and human review.</p>
          </div>
        </div>
      )}
    </div>
  );
}
