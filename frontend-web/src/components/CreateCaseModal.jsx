import React, { useEffect, useState } from 'react';
import { Plus, X, FolderPlus, Sparkles } from 'lucide-react';

export default function CreateCaseModal({
  isOpen,
  onClose,
  onCreateCase,
  existingCount = 0,
}) {
  const [caseId, setCaseId] = useState('');
  const [caseName, setCaseName] = useState('');
  const [category, setCategory] = useState('Surveillance & Robbery');
  const [description, setDescription] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [formError, setFormError] = useState('');

  useEffect(() => {
    if (!isOpen) return;
    const nextNumber = existingCount + 1;
    const suffix = nextNumber <= 26
      ? String.fromCharCode(64 + nextNumber)
      : String(nextNumber);
    setCaseId(`CASE-2026-${String(nextNumber).padStart(3, '0')}`);
    setCaseName(`Case ${suffix} - Forensic Investigation`);
    setCategory('Surveillance & Robbery');
    setDescription('Evidential review and investigation scope.');
    setFormError('');
  }, [isOpen, existingCount]);

  if (!isOpen) return null;

  const applyPreset = (title, nextCategory, nextDescription) => {
    setCaseName(title);
    setCategory(nextCategory);
    setDescription(nextDescription);
    setFormError('');
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    const cleanCaseId = caseId.trim();
    const cleanCaseName = caseName.trim();
    if (!cleanCaseId || !cleanCaseName) {
      setFormError('Case ID and case title are required.');
      return;
    }
    if (!onCreateCase) {
      setFormError('Case creation is not connected to the research API.');
      return;
    }

    setIsSubmitting(true);
    setFormError('');
    try {
      const result = await onCreateCase({
        case_id: cleanCaseId,
        name: cleanCaseName,
        category: category.trim(),
        description: description.trim(),
      });
      if (!result || result.success !== true) {
        setFormError(result?.error || 'The case was not created. Check the API and try again.');
        return;
      }
      onClose?.();
    } catch (error) {
      console.error('Error creating case:', error);
      setFormError(error?.message || 'The case could not be created.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[110] flex items-center justify-center bg-slate-950/80 backdrop-blur-md p-4 animate-fadeIn">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="create-case-title"
        className="bg-slate-900 border border-slate-700/80 rounded-3xl p-6 w-full max-w-lg shadow-2xl space-y-5 animate-scaleUp"
      >
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-blue-500/10 border border-blue-500/30 rounded-2xl text-blue-400">
              <FolderPlus size={24} />
            </div>
            <div>
              <h2 id="create-case-title" className="text-lg font-black text-white">
                Create New Forensic Case
              </h2>
              <p className="text-xs text-slate-400">
                Create an independent case record and graph namespace.
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={isSubmitting}
            aria-label="Close create case dialog"
            className="text-slate-400 hover:text-white p-1.5 rounded-xl hover:bg-slate-800 transition-colors disabled:opacity-40"
          >
            <X size={18} />
          </button>
        </div>

        <div className="bg-slate-950/80 p-3 rounded-2xl border border-slate-800 space-y-2">
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
            <Sparkles size={12} className="text-amber-400" /> Case presets (editable)
          </span>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
            <button
              type="button"
              onClick={() => applyPreset('North Harbor Storage', 'Commercial Robbery', 'Surveillance review of a reported warehouse breach.')}
              className="p-2 bg-slate-900 hover:bg-slate-800 rounded-xl border border-slate-700 text-[11px] text-slate-200 text-left"
            >
              🏢 <strong className="block text-slate-100">Warehouse</strong>
              North Harbor
            </button>
            <button
              type="button"
              onClick={() => applyPreset('Bank Perimeter Review', 'Armed Robbery', 'Review of surveillance footage around a bank perimeter.')}
              className="p-2 bg-slate-900 hover:bg-slate-800 rounded-xl border border-slate-700 text-[11px] text-slate-200 text-left"
            >
              🏦 <strong className="block text-slate-100">Bank</strong>
              Perimeter
            </button>
            <button
              type="button"
              onClick={() => applyPreset('Highway Incident Review', 'Vehicular Forensics', 'Review of available vehicle-related visual evidence.')}
              className="p-2 bg-slate-900 hover:bg-slate-800 rounded-xl border border-slate-700 text-[11px] text-slate-200 text-left"
            >
              🚗 <strong className="block text-slate-100">Highway</strong>
              Vehicle incident
            </button>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <label htmlFor="new-case-id" className="text-[11px] font-bold uppercase tracking-wider text-slate-400 block mb-1">
                Case Identifier
              </label>
              <input
                id="new-case-id"
                type="text"
                required
                maxLength={128}
                value={caseId}
                onChange={(event) => setCaseId(event.target.value)}
                placeholder="e.g. CASE-2026-002"
                disabled={isSubmitting}
                className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3 py-2 text-xs font-mono text-blue-300 focus:outline-none focus:border-blue-500 disabled:opacity-50"
              />
            </div>
            <div>
              <label htmlFor="new-case-category" className="text-[11px] font-bold uppercase tracking-wider text-slate-400 block mb-1">
                Category
              </label>
              <select
                id="new-case-category"
                value={category}
                onChange={(event) => setCategory(event.target.value)}
                disabled={isSubmitting}
                className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-blue-500 disabled:opacity-50"
              >
                <option value="Surveillance & Robbery">Surveillance & Robbery</option>
                <option value="Commercial Robbery">Commercial Robbery</option>
                <option value="Armed Robbery">Armed Robbery</option>
                <option value="Homicide Forensics">Homicide Forensics</option>
                <option value="Vehicular Forensics">Vehicular Forensics</option>
                <option value="Ballistics & Weapons">Ballistics & Weapons</option>
                <option value="Cyber & Digital Forensics">Cyber & Digital Forensics</option>
              </select>
            </div>
          </div>

          <div>
            <label htmlFor="new-case-name" className="text-[11px] font-bold uppercase tracking-wider text-slate-400 block mb-1">
              Case Title
            </label>
            <input
              id="new-case-name"
              type="text"
              required
              maxLength={200}
              value={caseName}
              onChange={(event) => setCaseName(event.target.value)}
              placeholder="e.g. North Harbor Storage Investigation"
              disabled={isSubmitting}
              className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white font-semibold focus:outline-none focus:border-blue-500 disabled:opacity-50"
            />
          </div>

          <div>
            <label htmlFor="new-case-description" className="text-[11px] font-bold uppercase tracking-wider text-slate-400 block mb-1">
              Case Description / Scope
            </label>
            <textarea
              id="new-case-description"
              rows={3}
              maxLength={2000}
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              placeholder="Describe the investigation scope and evidence expected..."
              disabled={isSubmitting}
              className="w-full bg-slate-950 border border-slate-700 rounded-xl p-3 text-xs text-slate-300 focus:outline-none focus:border-blue-500 resize-none disabled:opacity-50"
            />
          </div>

          {formError && (
            <div role="alert" className="rounded-xl border border-red-800 bg-red-950/40 px-3 py-2 text-xs text-red-300">
              {formError}
            </div>
          )}

          <div className="flex items-center justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              disabled={isSubmitting}
              className="px-4 py-2 text-xs font-semibold text-slate-400 hover:text-white rounded-xl hover:bg-slate-800 transition-colors disabled:opacity-40"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting || !caseId.trim() || !caseName.trim()}
              className="flex items-center gap-2 px-5 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white rounded-xl text-xs font-bold transition-all shadow-lg shadow-blue-500/25"
            >
              <Plus size={14} />
              <span>{isSubmitting ? 'Creating Case...' : 'Create Case & Open Tab'}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
