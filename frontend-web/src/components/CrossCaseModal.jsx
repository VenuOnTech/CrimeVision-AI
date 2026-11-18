import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import {
  Share2,
  X,
  Link as LinkIcon,
  Sparkles,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
} from 'lucide-react';

import { RESEARCH_API_URL } from '../lib/config';

const API_BASE = RESEARCH_API_URL;

const formatStorage = (item) => {
  if (item?.persisted === true || item?.storage_backend === 'neo4j') return 'Persisted in Neo4j';
  return 'Volatile memory only';
};

export default function CrossCaseModal({
  isOpen,
  onClose,
  cases = [],
  currentCaseId,
  onLinkCreated,
  existingLinks = [],
}) {
  const [activeTab, setActiveTab] = useState('manual');
  const [sourceCaseId, setSourceCaseId] = useState('');
  const [targetCaseId, setTargetCaseId] = useState('');
  const [relationType, setRelationType] = useState('SHARED_SUSPECT');
  const [description, setDescription] = useState('');
  const [confidence, setConfidence] = useState(0.85);
  const [sharedEntity, setSharedEntity] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [applyingSuggestionId, setApplyingSuggestionId] = useState(null);
  const [suggestions, setSuggestions] = useState([]);
  const [loadingSuggestions, setLoadingSuggestions] = useState(false);
  const [requestError, setRequestError] = useState('');

  const availableTargets = useMemo(
    () => cases.filter((item) => item.id !== sourceCaseId),
    [cases, sourceCaseId],
  );

  useEffect(() => {
    if (!isOpen) return;
    const nextSource = currentCaseId && cases.some((item) => item.id === currentCaseId)
      ? currentCaseId
      : (cases[0]?.id || '');
    const nextTarget = cases.find((item) => item.id !== nextSource)?.id || '';
    setSourceCaseId(nextSource);
    setTargetCaseId(nextTarget);
    setRequestError('');
    setActiveTab('manual');
  }, [isOpen, currentCaseId, cases]);

  useEffect(() => {
    if (!isOpen) return undefined;
    const controller = new AbortController();
    let active = true;
    setLoadingSuggestions(true);
    setRequestError('');

    axios.get(`${API_BASE}/cross_case_suggestions`, { signal: controller.signal })
      .then((response) => {
        if (active) setSuggestions(Array.isArray(response.data) ? response.data : []);
      })
      .catch((error) => {
        if (!active || axios.isCancel(error) || error.code === 'ERR_CANCELED') return;
        console.error('Could not load cross-case suggestions:', error);
        setRequestError(error.response?.data?.detail || 'Could not load heuristic suggestions from the research API.');
      })
      .finally(() => {
        if (active) setLoadingSuggestions(false);
      });

    return () => {
      active = false;
      controller.abort();
    };
  }, [isOpen]);

  if (!isOpen) return null;

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!sourceCaseId || !targetCaseId || sourceCaseId === targetCaseId) {
      setRequestError('Select two different existing cases.');
      return;
    }
    setIsSubmitting(true);
    setRequestError('');
    try {
      const response = await axios.post(`${API_BASE}/cross_case_relation`, {
        source_case_id: sourceCaseId,
        target_case_id: targetCaseId,
        relation_type: relationType,
        description: description.trim() || `Investigator-recorded relationship: ${relationType.replace(/_/g, ' ').toLowerCase()}`,
        confidence: Number(confidence),
        shared_entity: sharedEntity.trim() || null,
      });
      await onLinkCreated?.(response.data);
      onClose?.();
    } catch (error) {
      console.error('Could not create cross-case relationship:', error);
      setRequestError(error.response?.data?.detail || error.message || 'Could not create the relationship.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleApplySuggestion = async (suggestion) => {
    const suggestionId = suggestion.id || `${suggestion.source_case_id}:${suggestion.target_case_id}:${suggestion.shared_entity}`;
    setApplyingSuggestionId(suggestionId);
    setRequestError('');
    try {
      const response = await axios.post(`${API_BASE}/cross_case_relation`, {
        source_case_id: suggestion.source_case_id,
        target_case_id: suggestion.target_case_id,
        relation_type: 'POTENTIAL_KEYWORD_OVERLAP',
        description: suggestion.description || 'Heuristic keyword overlap; investigator review required.',
        confidence: 0,
        shared_entity: suggestion.shared_entity || null,
      });
      await onLinkCreated?.(response.data);
      onClose?.();
    } catch (error) {
      console.error('Could not record heuristic overlap:', error);
      setRequestError(error.response?.data?.detail || error.message || 'Could not record the overlap.');
    } finally {
      setApplyingSuggestionId(null);
    }
  };

  return (
    <div className="fixed inset-0 z-[110] flex items-center justify-center bg-slate-950/80 backdrop-blur-md p-4 animate-fadeIn">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="cross-case-title"
        className="bg-slate-900 border border-slate-700/80 rounded-3xl p-6 w-full max-w-xl shadow-2xl space-y-5 animate-scaleUp"
      >
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-amber-500/10 border border-amber-500/30 rounded-2xl text-amber-400">
              <Share2 size={24} />
            </div>
            <div>
              <h2 id="cross-case-title" className="text-lg font-black text-white">Cross-Case Relationship Review</h2>
              <p className="text-xs text-slate-400">Record or review possible relationships between existing cases.</p>
            </div>
          </div>
          <button type="button" onClick={onClose} disabled={isSubmitting || Boolean(applyingSuggestionId)} aria-label="Close cross-case dialog" className="text-slate-400 hover:text-white p-1.5 rounded-xl hover:bg-slate-800 disabled:opacity-40">
            <X size={18} />
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-2 border-b border-slate-800 pb-2">
          {[
            ['manual', 'Connect Cases'],
            ['suggestions', `Keyword Overlaps (${suggestions.length})`],
            ['existing', `Recorded Links (${existingLinks.length})`],
          ].map(([tab, label]) => (
            <button
              key={tab}
              type="button"
              onClick={() => { setActiveTab(tab); setRequestError(''); }}
              className={`px-3 py-1.5 rounded-xl text-xs font-bold transition-all ${activeTab === tab ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40' : 'text-slate-400 hover:text-slate-200'}`}
            >
              {tab === 'suggestions' && <Sparkles size={12} className="inline mr-1 text-purple-400" />}
              {label}
            </button>
          ))}
        </div>

        {requestError && (
          <div role="alert" className="flex items-start gap-2 rounded-xl border border-red-800 bg-red-950/40 px-3 py-2 text-xs text-red-300">
            <AlertTriangle size={15} className="mt-0.5 shrink-0" />
            <span>{requestError}</span>
          </div>
        )}

        {activeTab === 'manual' && (
          <form onSubmit={handleSubmit} className="space-y-4">
            {cases.length < 2 ? (
              <div className="rounded-xl border border-slate-700 bg-slate-950/70 p-4 text-xs text-slate-400">
                Create at least two cases before recording a cross-case relationship.
              </div>
            ) : (
              <>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 bg-slate-950/80 p-3.5 rounded-2xl border border-slate-800">
                  <div>
                    <label htmlFor="cross-case-source" className="text-[10px] font-bold uppercase tracking-wider text-blue-400 block mb-1">Source Case</label>
                    <select id="cross-case-source" value={sourceCaseId} onChange={(event) => { setSourceCaseId(event.target.value); setTargetCaseId(cases.find((item) => item.id !== event.target.value)?.id || ''); }} className="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white">
                      {cases.map((item) => <option key={item.id} value={item.id}>{item.name || item.id} ({item.id})</option>)}
                    </select>
                  </div>
                  <div>
                    <label htmlFor="cross-case-target" className="text-[10px] font-bold uppercase tracking-wider text-amber-400 block mb-1">Target Case</label>
                    <select id="cross-case-target" value={targetCaseId} onChange={(event) => setTargetCaseId(event.target.value)} className="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white">
                      <option value="">Select another case</option>
                      {availableTargets.map((item) => <option key={item.id} value={item.id}>{item.name || item.id} ({item.id})</option>)}
                    </select>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <div>
                    <label htmlFor="cross-case-relation" className="text-[11px] font-bold uppercase tracking-wider text-slate-400 block mb-1">Relationship Type</label>
                    <select id="cross-case-relation" value={relationType} onChange={(event) => setRelationType(event.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3 py-2 text-xs text-slate-200">
                      <option value="SHARED_SUSPECT">Shared suspect (investigator assessed)</option>
                      <option value="SHARED_VEHICLE">Shared vehicle (investigator assessed)</option>
                      <option value="BALLISTICS_MATCH">Ballistics / weapon association</option>
                      <option value="PROXIMITY_LOCATION">Geographic proximity</option>
                      <option value="CHRONOLOGICAL_SERIES">Chronological relationship</option>
                      <option value="OVERLAPPING_EVIDENCE">Overlapping evidence item</option>
                    </select>
                  </div>
                  <div>
                    <label htmlFor="cross-case-confidence" className="text-[11px] font-bold uppercase tracking-wider text-slate-400 block mb-1">
                      Investigator-assigned strength: <span className="text-amber-400 font-mono">{Math.round(confidence * 100)}%</span>
                    </label>
                    <input id="cross-case-confidence" type="range" min="0.4" max="0.99" step="0.01" value={confidence} onChange={(event) => setConfidence(Number(event.target.value))} className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-500 mt-3" />
                    <p className="mt-1 text-[10px] text-slate-500">This is manually assigned metadata, not a calibrated model probability.</p>
                  </div>
                </div>

                <div>
                  <label htmlFor="cross-case-entity" className="text-[11px] font-bold uppercase tracking-wider text-slate-400 block mb-1">Shared Entity / Reference (optional)</label>
                  <input id="cross-case-entity" type="text" maxLength={300} value={sharedEntity} onChange={(event) => setSharedEntity(event.target.value)} placeholder="Describe the entity or reference being compared" className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3 py-2 text-xs text-slate-200" />
                </div>
                <div>
                  <label htmlFor="cross-case-description" className="text-[11px] font-bold uppercase tracking-wider text-slate-400 block mb-1">Investigator Notes / Rationale</label>
                  <textarea id="cross-case-description" rows={2} maxLength={2000} value={description} onChange={(event) => setDescription(event.target.value)} placeholder="Record why this relationship should be considered." className="w-full bg-slate-950 border border-slate-700 rounded-xl p-2.5 text-xs text-slate-300 resize-none" />
                </div>
                <div className="flex justify-end">
                  <button type="submit" disabled={isSubmitting || !sourceCaseId || !targetCaseId || sourceCaseId === targetCaseId} className="flex items-center gap-2 px-5 py-2 bg-amber-600 hover:bg-amber-500 disabled:bg-slate-800 disabled:text-slate-500 text-white rounded-xl text-xs font-bold">
                    <LinkIcon size={14} /> {isSubmitting ? 'Saving relationship...' : 'Record Relationship'}
                  </button>
                </div>
              </>
            )}
          </form>
        )}

        {activeTab === 'suggestions' && (
          <div className="space-y-3 max-h-80 overflow-y-auto pr-1">
            <p className="text-[11px] text-slate-500">These candidates are generated by shared-word heuristics only. They are not identity matches, verified correlations, or calibrated probabilities.</p>
            {loadingSuggestions ? (
              <div className="text-center py-8 text-xs text-slate-400">Checking case statements for shared keywords...</div>
            ) : suggestions.length === 0 ? (
              <div className="text-center py-8 bg-slate-950/60 rounded-2xl border border-slate-800 p-6 text-xs text-slate-400">
                <Sparkles size={24} className="text-purple-400 mx-auto mb-2" />
                No keyword-overlap candidates were returned.
              </div>
            ) : suggestions.map((suggestion, index) => {
              const suggestionId = suggestion.id || `${suggestion.source_case_id}:${suggestion.target_case_id}:${suggestion.shared_entity || index}`;
              return (
                <div key={suggestionId} className="bg-slate-950 p-3.5 rounded-2xl border border-purple-500/30 flex flex-col md:flex-row items-start md:items-center justify-between gap-3">
                  <div className="space-y-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-xs font-bold text-blue-400">{suggestion.source_case_name || suggestion.source_case_id}</span>
                      <ArrowRight size={12} className="text-purple-400" />
                      <span className="text-xs font-bold text-amber-400">{suggestion.target_case_name || suggestion.target_case_id}</span>
                      <span className="text-[10px] px-2 py-0.5 rounded-full bg-purple-950 text-purple-300 border border-purple-800">Keyword overlap</span>
                    </div>
                    <p className="text-xs text-slate-300">{suggestion.description}</p>
                    {suggestion.shared_entity && <p className="text-[10px] text-slate-500">Shared text: “{suggestion.shared_entity}”</p>}
                  </div>
                  <button type="button" onClick={() => handleApplySuggestion(suggestion)} disabled={Boolean(applyingSuggestionId)} className="flex items-center gap-1 px-3 py-1.5 bg-purple-700 hover:bg-purple-600 disabled:bg-slate-800 text-white rounded-xl text-xs font-bold shrink-0">
                    <CheckCircle2 size={13} /> {applyingSuggestionId === suggestionId ? 'Recording...' : 'Record for review'}
                  </button>
                </div>
              );
            })}
          </div>
        )}

        {activeTab === 'existing' && (
          <div className="space-y-2 max-h-80 overflow-y-auto pr-1">
            {existingLinks.length === 0 ? (
              <div className="text-center py-8 text-xs text-slate-400 bg-slate-950/60 rounded-2xl border border-slate-800">No cross-case relationships have been recorded.</div>
            ) : existingLinks.map((link) => (
              <div key={link.id || `${link.source_case_id}:${link.target_case_id}:${link.relation_type}`} className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex items-center justify-between gap-3 text-xs">
                <div className="min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-bold text-blue-400">{link.source_case_id}</span>
                    <span className="text-slate-500">↔</span>
                    <span className="font-bold text-amber-400">{link.target_case_id}</span>
                    <span className="text-[10px] font-mono text-slate-400 uppercase">[{link.relation_type}]</span>
                  </div>
                  <p className="text-[11px] text-slate-400 mt-0.5">{link.description}</p>
                  {link.confidence_interpretation === 'investigator_assigned_not_calibrated' && (
                    <p className="text-[10px] text-slate-500">Investigator-assigned metadata: {Math.round(Number(link.confidence || 0) * 100)}% (not calibrated)</p>
                  )}
                  {link.confidence_interpretation === 'heuristic_candidate_not_calibrated' && (
                    <p className="text-[10px] text-slate-500">Recorded from a shared-keyword heuristic; not a verified relationship or probability.</p>
                  )}
                </div>
                <span className={`text-[10px] px-2 py-1 rounded-full border shrink-0 ${link.persisted === true ? 'bg-emerald-950 text-emerald-300 border-emerald-800' : 'bg-amber-950 text-amber-300 border-amber-800'}`}>
                  {formatStorage(link)}
                </span>
              </div>
            ))}
          </div>
        )}

        <div className="flex justify-end border-t border-slate-800 pt-3">
          <button type="button" onClick={onClose} disabled={isSubmitting || Boolean(applyingSuggestionId)} className="px-4 py-2 text-xs font-semibold text-slate-400 hover:text-white rounded-xl hover:bg-slate-800 disabled:opacity-40">Close</button>
        </div>
      </div>
    </div>
  );
}
