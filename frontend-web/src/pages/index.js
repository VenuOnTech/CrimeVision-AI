import React, { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import LocalDemoBanner from '../components/LocalDemoBanner';
import {
  ShieldAlert,
  FileSearch,
  RefreshCw,
  Sparkles,
  Trash2,
  Share2,
  FolderPlus,
  ArrowRight,
  Plus,
} from 'lucide-react';
import Heatmap from '../components/Heatmap';
import GraphView from '../components/GraphView';
import IngestionZone from '../components/IngestionZone';
import EvidenceTable from '../components/EvidenceTable';
import LLMAnalysisPanel from '../components/LLMAnalysisPanel';
import CaseTabBar from '../components/CaseTabBar';
import CreateCaseModal from '../components/CreateCaseModal';
import EditCaseModal from '../components/EditCaseModal';
import CrossCaseModal from '../components/CrossCaseModal';
import { RESEARCH_API_URL as API_BASE } from '../lib/config';
import { useCaseGraph } from '../hooks/useCaseGraph';

export default function Dashboard() {
  const [cases, setCases] = useState([]);
  const [openTabs, setOpenTabs] = useState([]);
  const [activeCaseId, setActiveCaseId] = useState(null);
  const [isMultiCaseView, setIsMultiCaseView] = useState(false);
  const [crossCaseLinks, setCrossCaseLinks] = useState([]);

  const [isCreateCaseOpen, setIsCreateCaseOpen] = useState(false);
  const [isCrossCaseOpen, setIsCrossCaseOpen] = useState(false);
  const [editingCase, setEditingCase] = useState(null);

  const [toastMessage, setToastMessage] = useState(null);
  const [resetTrigger, setResetTrigger] = useState(0);
  const casesRequestSequence = useRef(0);
  const toastTimerRef = useRef(null);
  const activeCaseIdRef = useRef(activeCaseId);
  const isMultiCaseViewRef = useRef(isMultiCaseView);

  const graph = useCaseGraph(activeCaseId, isMultiCaseView);

  const showToast = useCallback((msg) => {
    setToastMessage(msg);
    if (toastTimerRef.current) window.clearTimeout(toastTimerRef.current);
    toastTimerRef.current = window.setTimeout(() => {
      setToastMessage(null);
      toastTimerRef.current = null;
    }, 3500);
  }, []);

  useEffect(
    () => () => {
      if (toastTimerRef.current) window.clearTimeout(toastTimerRef.current);
    },
    [],
  );

  const activeCaseObj = cases.find((c) => c.id === activeCaseId) || {
    id: activeCaseId || 'UNASSIGNED',
    name: activeCaseId ? `Case #${activeCaseId}` : 'No Active Case',
  };

  const fetchCasesList = useCallback(async () => {
    const requestId = ++casesRequestSequence.current;
    try {
      const [casesResponse, linksResponse] = await Promise.all([
        axios.get(`${API_BASE}/cases`),
        axios.get(`${API_BASE}/cross_case_relations`).catch(() => ({ data: [] })),
      ]);
      if (requestId !== casesRequestSequence.current) return;

      const receivedCases = Array.isArray(casesResponse.data) ? casesResponse.data : [];
      setCases(receivedCases);
      setOpenTabs((previousTabs) => {
        if (previousTabs.length === 0) {
          return receivedCases.length > 0 ? [receivedCases[0]] : [];
        }
        return previousTabs
          .map((tab) => receivedCases.find((item) => item.id === tab.id) || null)
          .filter(Boolean);
      });
      setActiveCaseId((currentId) => {
        if (currentId && receivedCases.some((item) => item.id === currentId)) return currentId;
        return receivedCases[0]?.id || null;
      });
      setCrossCaseLinks(Array.isArray(linksResponse.data) ? linksResponse.data : []);
    } catch (error) {
      console.error('Could not refresh cases:', error);
    }
  }, []);

  useEffect(() => {
    activeCaseIdRef.current = activeCaseId;
    isMultiCaseViewRef.current = isMultiCaseView;
  }, [activeCaseId, isMultiCaseView]);

  useEffect(() => {
    fetchCasesList();
  }, [fetchCasesList]);

  const handleSelectTab = (caseId) => {
    activeCaseIdRef.current = caseId;
    isMultiCaseViewRef.current = false;
    setIsMultiCaseView(false);
    setActiveCaseId(caseId);
    if (!openTabs.some((tab) => tab.id === caseId)) {
      const match = cases.find((item) => item.id === caseId) || {
        id: caseId,
        name: `Case #${caseId}`,
      };
      setOpenTabs((previous) => [...previous, match]);
    }
    showToast(`Switched to case: ${cases.find((item) => item.id === caseId)?.name || caseId}`);
  };

  const handleCloseTab = (caseIdToClose) => {
    const updatedTabs = openTabs.filter((tab) => tab.id !== caseIdToClose);
    setOpenTabs(updatedTabs);
    if (String(activeCaseIdRef.current) === String(caseIdToClose)) {
      const nextCaseId = updatedTabs.length > 0 ? updatedTabs[updatedTabs.length - 1].id : null;
      activeCaseIdRef.current = nextCaseId;
      isMultiCaseViewRef.current = false;
      setIsMultiCaseView(false);
      setActiveCaseId(nextCaseId);
    }
  };

  const handleCreateCase = async (casePayload) => {
    try {
      const response = await axios.post(`${API_BASE}/cases`, casePayload);
      const newCase = response.data?.case;
      if (!newCase?.id) {
        throw new Error('The research API did not return a valid case record.');
      }
      activeCaseIdRef.current = newCase.id;
      isMultiCaseViewRef.current = false;
      setCases((previous) => [...previous.filter((item) => item.id !== newCase.id), newCase]);
      setOpenTabs((previous) => [...previous.filter((item) => item.id !== newCase.id), newCase]);
      setIsMultiCaseView(false);
      setActiveCaseId(newCase.id);
      const storage = response.data?.storage_status;
      showToast(
        storage?.backend === 'neo4j' && storage?.durable === true
          ? `Created case: ${newCase.name || newCase.id}`
          : `Created ${newCase.name || newCase.id} in volatile memory only; Neo4j persistence is unavailable.`,
      );
      fetchCasesList();
      return { success: true, case: newCase, storage_status: storage || null };
    } catch (error) {
      const message = error.response?.data?.detail || error.message || 'Unknown API error';
      showToast(`Case was not created: ${message}`);
      return { success: false, error: message };
    }
  };

  const handleSaveCase = async (caseId, updatedFields) => {
    try {
      const response = await axios.put(
        `${API_BASE}/cases/${encodeURIComponent(caseId)}`,
        updatedFields,
      );
      const updated = response.data?.case;
      if (!updated?.id) throw new Error('The API did not return the updated case record.');
      setCases((previous) =>
        previous.map((item) => (item.id === caseId ? { ...item, ...updated } : item)),
      );
      setOpenTabs((previous) =>
        previous.map((item) => (item.id === caseId ? { ...item, ...updated } : item)),
      );
      const storage = response.data?.storage_status;
      showToast(
        storage?.backend === 'neo4j' && storage?.durable === true
          ? `Updated case: ${updated.name || caseId}`
          : `Updated ${updated.name || caseId} in volatile memory only; Neo4j persistence is unavailable.`,
      );
      return { success: true, case: updated, storage_status: storage || null };
    } catch (error) {
      const message = error.response?.data?.detail || error.message || 'Unknown API error';
      showToast(`Case update failed: ${message}`);
      return { success: false, error: message };
    }
  };

  const handleDeleteCase = async (caseIdToDelete) => {
    const deletingId = String(caseIdToDelete);
    try {
      const response = await axios.delete(`${API_BASE}/cases/${encodeURIComponent(deletingId)}`);
      const remaining = cases.filter((item) => String(item.id) !== deletingId);
      setCases((previous) => previous.filter((item) => String(item.id) !== deletingId));
      setOpenTabs((previous) => previous.filter((item) => String(item.id) !== deletingId));
      if (String(activeCaseIdRef.current) === deletingId) {
        const nextCaseId = remaining[0]?.id || null;
        activeCaseIdRef.current = nextCaseId;
        isMultiCaseViewRef.current = false;
        setIsMultiCaseView(false);
        setActiveCaseId(nextCaseId);
      }
      const deletion = response.data || {};
      showToast(
        deletion.database_deleted === true
          ? `Deleted case: ${deletingId}`
          : `Removed ${deletingId} from volatile memory; Neo4j deletion was not confirmed.`,
      );
      await fetchCasesList();
    } catch (error) {
      showToast(`Case was not deleted: ${error.response?.data?.detail || error.message}`);
    }
  };

  const handleToggleMultiCaseView = () => {
    const nextValue = !isMultiCaseViewRef.current;
    isMultiCaseViewRef.current = nextValue;
    setIsMultiCaseView(nextValue);
  };

  const handleUploadSuccess = (data, requestContext = {}) => {
    if (!data) return;
    const responseCaseId = data.case_id == null ? '' : String(data.case_id);
    const requestCaseId = requestContext.caseId == null ? '' : String(requestContext.caseId);
    const expectedCaseId = responseCaseId || requestCaseId;
    const selectedCaseId =
      activeCaseIdRef.current == null ? '' : String(activeCaseIdRef.current);

    if (!expectedCaseId || (responseCaseId && requestCaseId && responseCaseId !== requestCaseId)) {
      showToast(
        'Analysis response did not identify the requested case. Visible case data was left unchanged.',
      );
      return;
    }

    const assessmentStatus = ['MATCH', 'CONTRADICTION', 'INDETERMINATE'].includes(
      data.assessment_status,
    )
      ? data.assessment_status
      : 'INDETERMINATE';

    if (
      requestContext.stale === true ||
      isMultiCaseViewRef.current ||
      expectedCaseId !== selectedCaseId
    ) {
      fetchCasesList();
      if (expectedCaseId === selectedCaseId) {
        graph.refresh();
      }
      showToast(
        `Analysis for ${expectedCaseId} completed. The current case view was refreshed without applying a stale response.`,
      );
      return;
    }

    graph.refresh();
    fetchCasesList();

    const toast =
      assessmentStatus === 'CONTRADICTION'
        ? 'Research classifier assessment: CONTRADICTION'
        : assessmentStatus === 'MATCH'
          ? 'Research classifier assessment: MATCH'
          : 'Analysis completed; assessment is INDETERMINATE';
    showToast(toast);
  };

  const handleSelectEvidence = (index) => {
    graph.selectEvidence(index);
    const item = graph.graph.evidence[index];
    if (item) {
      showToast(`Inspecting evidence: ${item.evidence_filename || 'record'}`);
    }
  };

  const handleClearAll = async () => {
    const clearCaseId =
      activeCaseIdRef.current == null ? '' : String(activeCaseIdRef.current);
    if (!clearCaseId || isMultiCaseViewRef.current) {
      showToast('Create or select a single case before clearing evidence.');
      return;
    }
    let clearResponse;
    try {
      clearResponse = await axios.post(`${API_BASE}/clear/${encodeURIComponent(clearCaseId)}`);
    } catch (error) {
      showToast(`Evidence was not cleared: ${error.response?.data?.detail || error.message}`);
      return;
    }
    setResetTrigger((previous) => previous + 1);
    await graph.refresh();
    await fetchCasesList();
    const clearResult = clearResponse?.data || {};
    showToast(
      clearResult.database_cleared === true
        ? `Evidence cleared from ${clearCaseId}; case retained in Neo4j.`
        : `Evidence cleared from volatile memory only; Neo4j clearing was not confirmed.`,
    );
  };

  const handleCrossCaseCreated = async (linkData) => {
    showToast(`Established cross-case link: ${linkData.link?.relation_type || 'Correlated'}`);
    await fetchCasesList();
  };

  const evidenceList = graph.graph.evidence;
  const totalIngested = evidenceList.length;
  const totalContradictions = evidenceList.filter(
    (item) => item.assessment_status === 'CONTRADICTION',
  ).length;
  const totalMatches = evidenceList.filter((item) => item.assessment_status === 'MATCH').length;
  const totalIndeterminate = evidenceList.filter(
    (item) => !['MATCH', 'CONTRADICTION'].includes(item.assessment_status),
  ).length;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 md:p-8 font-sans selection:bg-blue-600 selection:text-white">
      {toastMessage && (
        <div className="fixed top-5 right-5 z-[120] bg-blue-600 text-white px-4 py-2.5 rounded-xl shadow-2xl flex items-center gap-2 text-xs font-bold animate-slideIn">
          <Sparkles size={16} />
          <span>{toastMessage}</span>
        </div>
      )}

      <header className="flex flex-wrap justify-between items-center mb-5 border-b border-slate-800 pb-5 gap-4">
        <div className="flex items-center gap-3.5">
          <div className="p-3 bg-blue-600/20 border border-blue-500/40 rounded-2xl shadow-lg shadow-blue-500/10">
            <ShieldAlert size={28} className="text-blue-400" />
          </div>
          <div>
            <h1 className="text-2xl md:text-3xl font-black tracking-tight text-white flex items-center gap-2.5">
              CrimeVision AI{' '}
              <span className="text-xs font-mono font-bold px-2.5 py-0.5 rounded-full bg-blue-950 text-blue-300 border border-blue-700">
                Investigator Workspace
              </span>
            </h1>
            <p className="text-xs text-slate-400 mt-0.5">
              Multi-case graph • CLIP + Optimal Transport + calibrated abstention • local-first decision support
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => setIsCrossCaseOpen(true)}
            className="flex items-center gap-1.5 text-xs font-bold bg-amber-950/70 hover:bg-amber-900/90 text-amber-200 border border-amber-700/90 px-3.5 py-2 rounded-xl transition-all shadow"
            title="Relate between Case A and Case B in Neo4j"
          >
            <Share2 size={14} className="text-amber-400" /> Relate Cases
          </button>
          <button
            type="button"
            onClick={() => setIsCreateCaseOpen(true)}
            className="flex items-center gap-1.5 text-xs font-bold bg-blue-600 hover:bg-blue-500 text-white border border-blue-400/50 px-3.5 py-2 rounded-xl transition-all shadow-lg shadow-blue-500/25"
            title="Create New Case Tab"
          >
            <FolderPlus size={14} /> + Create Case
          </button>
          {activeCaseId && (
            <button
              type="button"
              onClick={handleClearAll}
              className="flex items-center gap-1.5 text-xs font-semibold bg-slate-900 hover:bg-red-950/60 text-slate-400 hover:text-red-300 border border-slate-800 hover:border-red-800 px-3 py-2 rounded-xl transition-all shadow"
              title="Reset active case view"
            >
              <Trash2 size={13} /> Reset Case
            </button>
          )}
          <button
            type="button"
            onClick={graph.refresh}
            className="flex items-center gap-1.5 text-xs font-semibold bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-700 px-3.5 py-2 rounded-xl transition-all shadow hover:border-slate-600"
            title="Refresh from Neo4j"
          >
            <RefreshCw size={13} className="text-blue-400" /> Refresh
          </button>
        </div>
      </header>
      <LocalDemoBanner />

      <CaseTabBar
        cases={cases}
        openTabs={openTabs}
        activeTabId={activeCaseId}
        onSelectTab={handleSelectTab}
        onCloseTab={handleCloseTab}
        onOpenNewCaseModal={() => setIsCreateCaseOpen(true)}
        onOpenCrossCaseModal={() => setIsCrossCaseOpen(true)}
        onEditCase={(caseItem) => setEditingCase(caseItem)}
        onDeleteCase={handleDeleteCase}
        isMultiCaseView={isMultiCaseView}
        onToggleMultiCaseView={handleToggleMultiCaseView}
        crossCaseCount={crossCaseLinks.length}
      />

      {cases.length === 0 && (
        <div className="mb-6 p-8 rounded-3xl bg-gradient-to-r from-slate-900/90 via-blue-950/40 to-slate-900/90 border-2 border-dashed border-blue-500/40 flex flex-col md:flex-row items-center justify-between gap-6 shadow-2xl animate-fadeIn">
          <div className="space-y-2">
            <span className="text-xs font-mono font-bold text-blue-400 uppercase tracking-wider flex items-center gap-2">
              <Sparkles size={16} /> Forensic Investigation Setup
            </span>
            <h2 className="text-xl font-black text-white">No Active Forensic Case Initialized</h2>
            <p className="text-xs text-slate-300 max-w-xl leading-relaxed">
              Create a case file first. You can then upload an original image/video with a genuine
              witness statement. Analysis diagnostics and case-graph persistence are shown separately.
            </p>
          </div>
          <button
            type="button"
            onClick={() => setIsCreateCaseOpen(true)}
            className="flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-500 text-white rounded-2xl text-sm font-black transition-all shadow-xl shadow-blue-500/30 shrink-0"
          >
            <Plus size={16} />
            <span>Create Case 1 to Start</span>
            <ArrowRight size={16} />
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <div className="space-y-6">
          <IngestionZone
            onUploadSuccess={handleUploadSuccess}
            onClearAll={handleClearAll}
            resetTrigger={resetTrigger}
            caseId={activeCaseId || ''}
            caseName={isMultiCaseView ? 'Multi-case view' : activeCaseObj.name}
            disabled={isMultiCaseView || !activeCaseId}
          />

          <div className="bg-slate-800/95 backdrop-blur-md p-5 rounded-2xl border border-slate-700/80 shadow-2xl space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-black text-slate-100 flex items-center gap-2">
                <div className="p-1.5 bg-blue-500/10 border border-blue-500/30 rounded-lg text-blue-400">
                  <FileSearch size={18} />
                </div>
                {isMultiCaseView ? 'Global Multi-Case Metrics' : 'Case Investigation Metrics'}
              </h3>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-slate-900 border border-slate-700 text-slate-400">
                {isMultiCaseView
                  ? `${cases.length} Cases`
                  : activeCaseId
                    ? `#${activeCaseId}`
                    : 'Standby'}
              </span>
            </div>

            <div className="grid grid-cols-2 xl:grid-cols-5 gap-3">
              <div className="bg-slate-900/80 p-3 rounded-xl border border-slate-700/60 space-y-1">
                <span className="text-[10px] uppercase font-bold text-slate-400">Total Ingested</span>
                <p className="text-xl font-black text-white font-mono">{totalIngested}</p>
              </div>
              <div className="bg-slate-900/80 p-3 rounded-xl border border-slate-700/60 space-y-1">
                <span className="text-[10px] uppercase font-bold text-emerald-400">Classifier Matches</span>
                <p className="text-xl font-black text-emerald-400 font-mono">{totalMatches}</p>
              </div>
              <div className="bg-slate-900/80 p-3 rounded-xl border border-slate-700/60 space-y-1">
                <span className="text-[10px] uppercase font-bold text-red-400">Classifier Contradictions</span>
                <p className="text-xl font-black text-red-400 font-mono">{totalContradictions}</p>
              </div>
              <div className="bg-slate-900/80 p-3 rounded-xl border border-slate-700/60 space-y-1">
                <span className="text-[10px] uppercase font-bold text-slate-400">Indeterminate</span>
                <p className="text-xl font-black text-slate-300 font-mono">{totalIndeterminate}</p>
              </div>
              <div className="bg-slate-900/80 p-3 rounded-xl border border-slate-700/60 space-y-1">
                <span className="text-[10px] uppercase font-bold text-purple-400">Cross-Case Links</span>
                <p className="text-xs font-bold text-purple-300 font-mono mt-1 flex items-center gap-1">
                  <Share2 size={13} className="text-amber-400" /> {crossCaseLinks.length} Active
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="lg:col-span-2 space-y-6">
          <GraphView
            nodes={graph.graph.nodes}
            edges={graph.graph.edges}
            onRefresh={graph.refresh}
            isMultiCaseView={isMultiCaseView}
            cases={cases}
            onSelectCase={handleSelectTab}
            activeCaseId={activeCaseId}
          />

          <Heatmap
            data={graph.activeHeatmap.matrix}
            tokens={graph.activeHeatmap.tokens}
            patches={graph.activeHeatmap.patches}
            assessmentStatus={graph.activeHeatmap.assessmentStatus}
            modelEpistemicUncertainty={graph.activeHeatmap.modelEpistemicUncertainty}
            visualQualityMetrics={graph.activeHeatmap.visualQualityMetrics}
            evidenceName={graph.activeHeatmap.evidenceName}
            cost={graph.activeHeatmap.cost}
          />
        </div>
      </div>

      <div className="w-full mb-6">
        <LLMAnalysisPanel
          evidenceName={graph.activeAnalysis.evidenceName}
          statementText={graph.activeAnalysis.statementText}
          cost={graph.activeAnalysis.cost}
          assessmentStatus={graph.activeAnalysis.assessmentStatus}
          modelEpistemicUncertainty={graph.activeAnalysis.modelEpistemicUncertainty}
          visualQualityMetrics={graph.activeAnalysis.visualQualityMetrics}
          auditReport={graph.activeAnalysis.auditReport}
          investigatorAdvisory={graph.activeAnalysis.investigatorAdvisory}
          graphPersistenceStatus={graph.activeAnalysis.graphPersistenceStatus}
          caseId={activeCaseId || 'MULTI'}
          caseName={activeCaseObj.name}
          crossCaseLinks={crossCaseLinks}
          isMultiCaseView={isMultiCaseView}
        />
      </div>

      <div className="w-full">
        <EvidenceTable
          evidenceList={graph.graph.evidence}
          onSelectEvidence={handleSelectEvidence}
          selectedIndex={graph.selectedIndex}
        />
      </div>

      <CreateCaseModal
        isOpen={isCreateCaseOpen}
        onClose={() => setIsCreateCaseOpen(false)}
        onCreateCase={handleCreateCase}
        existingCount={cases.length}
      />
      <EditCaseModal
        isOpen={Boolean(editingCase)}
        onClose={() => setEditingCase(null)}
        caseItem={editingCase}
        onSaveCase={handleSaveCase}
      />
      <CrossCaseModal
        isOpen={isCrossCaseOpen}
        onClose={() => setIsCrossCaseOpen(false)}
        cases={cases}
        currentCaseId={activeCaseId}
        onLinkCreated={handleCrossCaseCreated}
        existingLinks={crossCaseLinks}
      />
    </div>
  );
}