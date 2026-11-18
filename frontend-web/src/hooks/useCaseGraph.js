// Single source of truth for one case's graph, evidence, and analysis panels.
// All panels read from this hook; when its state updates, every panel updates
// from the same response. This eliminates the "stale panel" class of bugs.

import { useCallback, useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { RESEARCH_API_URL } from '../lib/config';

const EMPTY = {
  nodes: [],
  edges: [],
  evidence: [],
  storageStatus: null,
  crossCaseLinks: [],
};

function buildActiveAnalysis(evidence, storageStatus) {
  if (!evidence) {
    return {
      evidenceName: null,
      statementText: null,
      cost: null,
      assessmentStatus: 'INDETERMINATE',
      modelEpistemicUncertainty: null,
      visualQualityMetrics: null,
      auditReport: null,
      investigatorAdvisory: null,
      graphPersistenceStatus: storageStatus,
    };
  }
  return {
    evidenceName: evidence.evidence_filename || null,
    statementText: evidence.statement_text || null,
    cost: evidence.sinkhorn_cost ?? null,
    assessmentStatus: evidence.assessment_status || 'INDETERMINATE',
    modelEpistemicUncertainty: evidence.model_epistemic_uncertainty ?? null,
    visualQualityMetrics: evidence.visual_quality_metrics || null,
    auditReport: evidence.audit_report || null,
    investigatorAdvisory: evidence.investigator_advisory || null,
    graphPersistenceStatus: storageStatus,
  };
}

function buildActiveHeatmap(evidence) {
  if (!evidence) {
    return {
      tokens: [],
      patches: [],
      matrix: [],
      assessmentStatus: 'INDETERMINATE',
      modelEpistemicUncertainty: null,
      visualQualityMetrics: null,
      evidenceName: null,
      cost: null,
    };
  }
  return {
    tokens: Array.isArray(evidence.tokens) ? evidence.tokens : [],
    patches: Array.isArray(evidence.patches) ? evidence.patches : [],
    matrix: Array.isArray(evidence.heatmap_data) ? evidence.heatmap_data : [],
    assessmentStatus: evidence.assessment_status || 'INDETERMINATE',
    modelEpistemicUncertainty: evidence.model_epistemic_uncertainty ?? null,
    visualQualityMetrics: evidence.visual_quality_metrics || null,
    evidenceName: evidence.evidence_filename || null,
    cost: evidence.sinkhorn_cost ?? null,
  };
}

export function useCaseGraph(caseId, isMultiCase = false) {
  const [graph, setGraph] = useState(EMPTY);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const requestSeq = useRef(0);

  const fetchGraph = useCallback(async (targetCaseId, forceMulti) => {
    const requestId = ++requestSeq.current;
    if (!targetCaseId && !forceMulti) {
      setGraph(EMPTY);
      setSelectedIndex(0);
      setError(null);
      setIsLoading(false);
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const endpoint = forceMulti
        ? `${RESEARCH_API_URL}/graph/all`
        : `${RESEARCH_API_URL}/graph/${encodeURIComponent(targetCaseId)}`;
      const response = await axios.get(endpoint);
      if (requestId !== requestSeq.current) return;
      const p = response.data || {};
      const evidence = Array.isArray(p.evidence_list) ? p.evidence_list : [];
      setGraph({
        nodes: Array.isArray(p.nodes) ? p.nodes : [],
        edges: Array.isArray(p.edges) ? p.edges : [],
        evidence,
        storageStatus: p.storage_status || null,
        crossCaseLinks: Array.isArray(p.cross_case_links) ? p.cross_case_links : [],
      });
      setSelectedIndex((cur) => (cur < evidence.length ? cur : 0));
    } catch (err) {
      if (requestId !== requestSeq.current) return;
      console.error('Graph fetch failed:', err);
      setGraph(EMPTY);
      setSelectedIndex(0);
      setError(err?.response?.data?.detail || err?.message || 'Could not load case data.');
    } finally {
      if (requestId === requestSeq.current) setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchGraph(caseId, isMultiCase);
  }, [caseId, isMultiCase, fetchGraph]);

  const refresh = useCallback(
    () => fetchGraph(caseId, isMultiCase),
    [caseId, isMultiCase, fetchGraph],
  );

  const selectEvidence = useCallback((index) => {
    if (Number.isInteger(index) && index >= 0) {
      setSelectedIndex(index);
    }
  }, []);

  const activeEvidence = graph.evidence[selectedIndex] || null;

  return {
    graph,
    isLoading,
    error,
    refresh,
    selectedIndex,
    selectEvidence,
    activeEvidence,
    activeAnalysis: buildActiveAnalysis(activeEvidence, graph.storageStatus),
    activeHeatmap: buildActiveHeatmap(activeEvidence),
  };
}