import React, { useState } from 'react';
import { X, FileVideo, FileImage, Cpu, Copy, Check, AlertTriangle, CheckCircle2, HelpCircle, Info } from 'lucide-react';

const VALID_ASSESSMENTS = new Set(['MATCH', 'CONTRADICTION', 'INDETERMINATE']);

export default function EvidenceModal({ evidence, isOpen, onClose }) {
  const [copied, setCopied] = useState(false);
  if (!isOpen || !evidence) return null;

  // Legacy records have no calibrated assessment status; fail closed.
  const rawAssessment = String(evidence.assessment_status || '').toUpperCase();
  const assessmentStatus = VALID_ASSESSMENTS.has(rawAssessment)
    ? rawAssessment
    : 'INDETERMINATE';
  const isVideo = /\.(mp4|avi|mov|mkv)$/i.test(evidence.evidence_filename || '');
  const statusLabel = assessmentStatus;

  const copyDossier = async () => {
    const report = [
      'CRIMEVISION AI - RESEARCH ANALYSIS RECORD',
      `Case ID: ${evidence.case_id || 'Unassigned'}`,
      `Evidence: ${evidence.evidence_filename || 'N/A'}`,
      `Witness statement: ${evidence.statement_text || 'N/A'}`,
      `Classifier assessment: ${assessmentStatus}`,
      `Debiased Sinkhorn divergence (diagnostic): ${typeof evidence.sinkhorn_cost === 'number' ? evidence.sinkhorn_cost.toFixed(6) : 'N/A'}`,
      `Model epistemic uncertainty: ${typeof evidence.model_epistemic_uncertainty === 'number' ? evidence.model_epistemic_uncertainty.toFixed(6) : 'Unavailable'}`,
      `Timestamp: ${evidence.timestamp || 'Recorded'}`,
      '',
      'Analysis report:',
      evidence.audit_report || 'No report was stored.',
      '',
      'This is an automated research output, not an independently verified physical finding or legal conclusion.',
    ].join('\n');

    try {
      await navigator.clipboard.writeText(report);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch (error) {
      console.error('Could not copy evidence record:', error);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
      <div className="bg-slate-900 border border-slate-700 w-full max-w-2xl rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]" onClick={(event) => event.stopPropagation()}>
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-950/60">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-slate-800 border border-slate-700 text-slate-200">
              {assessmentStatus === 'CONTRADICTION' ? <AlertTriangle size={20} /> : assessmentStatus === 'MATCH' ? <CheckCircle2 size={20} /> : <HelpCircle size={20} />}
            </div>
            <div>
              <h3 className="text-base font-bold text-white">Evidence Analysis Record</h3>
              <span className="text-xs font-mono text-slate-300">{statusLabel}</span>
              <p className="text-xs text-slate-400 font-mono">Case: {evidence.case_id || 'Unassigned'}</p>
            </div>
          </div>
          <button type="button" onClick={onClose} className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800"><X size={18} /></button>
        </div>

        <div className="p-6 overflow-y-auto space-y-4 text-sm">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-slate-800/80 p-3.5 rounded-xl border border-slate-700/60 space-y-1.5">
              <span className="text-xs font-semibold text-slate-400 uppercase flex items-center gap-1.5">{isVideo ? <FileVideo size={14} /> : <FileImage size={14} />} Evidence media</span>
              <p className="font-mono text-sm text-slate-100 font-bold break-all">{evidence.evidence_filename || 'Unavailable'}</p>
            </div>
            <div className="bg-slate-800/80 p-3.5 rounded-xl border border-slate-700/60 space-y-1.5">
              <span className="text-xs font-semibold text-slate-400 uppercase flex items-center gap-1.5"><Cpu size={14} /> OT diagnostic</span>
              <p className="text-xs text-slate-400">Debiased Sinkhorn divergence (not a verdict threshold)</p>
              <p className="font-mono font-bold text-blue-300 text-lg">{typeof evidence.sinkhorn_cost === 'number' ? evidence.sinkhorn_cost.toFixed(6) : 'Unavailable'}</p>
            </div>
          </div>

          <div className="bg-slate-800/60 p-4 rounded-xl border border-slate-700/80">
            <span className="text-xs font-semibold text-slate-400 uppercase block mb-1.5">Witness statement</span>
            <blockquote className="text-slate-200 italic bg-slate-900/80 p-3 rounded-lg border-l-2 border-blue-500">{evidence.statement_text || 'No statement recorded.'}</blockquote>
          </div>

          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
            <span className="text-xs font-bold text-blue-300 uppercase">Research analysis report</span>
            <p className="text-xs text-slate-300 leading-relaxed whitespace-pre-wrap">{evidence.audit_report || 'No report is stored for this record. The interface does not infer a verdict from the divergence score.'}</p>
          </div>

          <div className="bg-amber-950/20 p-3 rounded-xl border border-amber-800/60 flex items-start gap-2 text-xs text-amber-200">
            <Info size={15} className="mt-0.5 shrink-0" />
            <p>Image-quality diagnostics and model epistemic uncertainty are separate measures. An automated research assessment is not proof of what physically occurred or a legal conclusion.</p>
          </div>
        </div>

        <div className="px-6 py-3.5 border-t border-slate-800 bg-slate-950/60 flex items-center justify-between">
          <button type="button" onClick={copyDossier} className="flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700">
            {copied ? <Check size={14} className="text-emerald-400" /> : <Copy size={14} />}{copied ? 'Copied' : 'Copy Analysis Record'}
          </button>
          <button type="button" onClick={onClose} className="text-xs font-semibold px-4 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white">Close</button>
        </div>
      </div>
    </div>
  );
}