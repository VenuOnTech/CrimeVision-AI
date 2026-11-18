import React, { useMemo, useState } from 'react';
import { Database, FileVideo, FileImage, AlertTriangle, CheckCircle2, HelpCircle, Search, Download, Eye } from 'lucide-react';
import EvidenceModal from './EvidenceModal';

// Legacy records were produced by the retired uncalibrated 1.15 rule.
// Do not reinterpret their old boolean flag as a validated classifier result.
// Unknown / lowercase / malformed status values fail closed to INDETERMINATE.
const VALID_STATUSES = new Set(['MATCH', 'CONTRADICTION', 'INDETERMINATE']);

const getStatus = (item) => {
  const status = String(item?.assessment_status || '').toUpperCase();
  return VALID_STATUSES.has(status) ? status : 'INDETERMINATE';
};

export default function EvidenceTable({ evidenceList = [], onSelectEvidence, selectedIndex }) {
  const [filter, setFilter] = useState('ALL');
  const [searchQuery, setSearchQuery] = useState('');
  const [modalEvidence, setModalEvidence] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const filteredList = useMemo(() => {
    return evidenceList.filter((item) => {
      const status = getStatus(item);
      if (filter !== 'ALL' && status !== filter) return false;
      if (!searchQuery.trim()) return true;
      const query = searchQuery.toLowerCase();
      return item.evidence_filename?.toLowerCase().includes(query) || item.statement_text?.toLowerCase().includes(query);
    });
  }, [evidenceList, filter, searchQuery]);

  const openDossier = (event, item) => {
    event.stopPropagation();
    setModalEvidence(item);
    setIsModalOpen(true);
  };

  const exportEvidenceJSON = () => {
    const blob = new Blob([JSON.stringify(evidenceList, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `crimevision_case_evidence_${Date.now()}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  };

  const countFor = (status) => evidenceList.filter((item) => getStatus(item) === status).length;

  return (
    <div className="bg-slate-800/95 backdrop-blur-md p-6 rounded-2xl border border-slate-700/80 shadow-2xl flex flex-col space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-4 pb-2 border-b border-slate-700/60">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-blue-500/10 border border-blue-500/30 rounded-xl text-blue-400"><Database size={22} /></div>
          <div>
            <h3 className="text-lg font-black text-slate-100">Ingested Evidence Repository</h3>
            <p className="text-xs text-slate-400">Analysis results associated with the current case</p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <div className="relative">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input type="text" value={searchQuery} onChange={(event) => setSearchQuery(event.target.value)} placeholder="Search file or statement…" className="bg-slate-900/90 border border-slate-700 rounded-lg pl-8 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-blue-500 w-44 md:w-56" />
          </div>
          <select value={filter} onChange={(event) => setFilter(event.target.value)} className="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200">
            <option value="ALL">All ({evidenceList.length})</option>
            <option value="MATCH">Matches ({countFor('MATCH')})</option>
            <option value="CONTRADICTION">Contradictions ({countFor('CONTRADICTION')})</option>
            <option value="INDETERMINATE">Indeterminate ({countFor('INDETERMINATE')})</option>
          </select>
          <button type="button" onClick={exportEvidenceJSON} className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg bg-slate-900 hover:bg-slate-700 text-slate-300 border border-slate-700">
            <Download size={13} /> Export JSON
          </button>
        </div>
      </div>

      {filteredList.length === 0 ? (
        <div className="py-12 text-center text-slate-500">
          <Database size={36} className="mx-auto mb-2 opacity-50" />
          <p className="text-sm font-medium">No matching evidence records found.</p>
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-slate-700/80 bg-slate-900/60">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-slate-950/90 uppercase font-mono text-[11px] text-slate-400 border-b border-slate-700">
              <tr>
                <th className="py-3 px-4">#</th>
                <th className="py-3 px-4">Case</th>
                <th className="py-3 px-4">Evidence</th>
                <th className="py-3 px-4">Witness Statement</th>
                <th className="py-3 px-4 text-center">Sinkhorn Divergence</th>
                <th className="py-3 px-4 text-center">Classifier Assessment</th>
                <th className="py-3 px-4">Timestamp</th>
                <th className="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {filteredList.map((item, idx) => {
                const status = getStatus(item);
                const isSelected = selectedIndex === evidenceList.indexOf(item);
                const isVideo = /\.(mp4|avi|mov|mkv)$/i.test(item.evidence_filename || '');
                return (
                  <tr key={`${item.case_id}-${item.evidence_filename}-${idx}`} onClick={() => onSelectEvidence?.(evidenceList.indexOf(item))} className={`transition-colors cursor-pointer ${isSelected ? 'bg-blue-950/50 border-l-4 border-l-blue-500' : 'hover:bg-slate-800/60'}`}>
                    <td className="py-3 px-4 font-mono text-slate-500">{idx + 1}</td>
                    <td className="py-3 px-4 font-mono"><span className="px-2 py-0.5 rounded-lg bg-blue-950/80 text-blue-300 border border-blue-800 text-[10px]">{item.case_id || 'Unassigned'}</span></td>
                    <td className="py-3 px-4 font-medium text-slate-100"><div className="flex items-center gap-2">{isVideo ? <FileVideo size={15} className="text-blue-400" /> : <FileImage size={15} className="text-purple-400" />}<span className="truncate max-w-[150px]" title={item.evidence_filename}>{item.evidence_filename || 'Unnamed evidence'}</span></div></td>
                    <td className="py-3 px-4 text-slate-300"><p className="truncate max-w-[280px]" title={item.statement_text}>{item.statement_text || 'No statement'}</p></td>
                    <td className="py-3 px-4 text-center font-mono font-bold text-blue-300">{typeof item.sinkhorn_cost === 'number' ? item.sinkhorn_cost.toFixed(6) : 'N/A'}</td>
                    <td className="py-3 px-4 text-center">
                      {status === 'CONTRADICTION' ? (
                        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] bg-red-950 text-red-300 border border-red-800"><AlertTriangle size={12} /> Contradiction</span>
                      ) : status === 'MATCH' ? (
                        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] bg-emerald-950 text-emerald-300 border border-emerald-800"><CheckCircle2 size={12} /> Match</span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] bg-slate-900 text-slate-300 border border-slate-700"><HelpCircle size={12} /> Indeterminate</span>)}
                    </td>
                    <td className="py-3 px-4 text-[11px] font-mono text-slate-400">{item.timestamp || 'Recorded'}</td>
                    <td className="py-3 px-4 text-right"><button type="button" onClick={(event) => openDossier(event, item)} className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300"><Eye size={13} /> Inspect</button></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <EvidenceModal evidence={modalEvidence} isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} />
    </div>
  );
}