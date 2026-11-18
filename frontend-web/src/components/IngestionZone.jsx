import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import {
  UploadCloud,
  FileVideo,
  FileImage,
  FileText,
  Loader2,
  AlertCircle,
  Sparkles,
  RotateCcw,
  Database,
  AlertTriangle,
  CheckCircle2,
  X,
} from 'lucide-react';
import { RESEARCH_API_URL } from '../lib/config';

const MEDIA_EXTENSIONS = /\.(mp4|avi|mov|mkv|jpg|jpeg|png|bmp|webp|tif|tiff)$/i;
const TEXT_EXTENSION = /\.txt$/i;

let nextId = 1;

function makeQueueItem(file) {
  return {
    id: `q-${nextId++}`,
    file,
    status: 'pending',
    error: null,
    result: null,
  };
}

function fileIcon(file) {
  if (file.type?.startsWith('video/')) return <FileVideo size={15} className="text-blue-400" />;
  if (file.type?.startsWith('image/')) return <FileImage size={15} className="text-purple-400" />;
  return <FileText size={15} className="text-slate-400" />;
}

export default function IngestionZone({
  onUploadSuccess,
  onClearAll,
  resetTrigger,
  caseId = '',
  caseName = '',
  disabled = false,
}) {
  const [queue, setQueue] = useState([]);
  const [statement, setStatement] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);

  const lifecycleRef = useRef(0);
  const currentCaseIdRef = useRef(String(caseId || ''));
  const mountedRef = useRef(false);

  currentCaseIdRef.current = String(caseId || '');

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      lifecycleRef.current += 1;
    };
  }, []);

  useEffect(() => {
    lifecycleRef.current += 1;
    setQueue([]);
    setStatement('');
    setError(null);
    setIsProcessing(false);
  }, [resetTrigger, caseId]);

  const onDrop = useCallback(
    async (acceptedFiles, rejectedFiles) => {
      setError(null);
      const dropCaseId = String(currentCaseIdRef.current || '');

      if (disabled || !dropCaseId) {
        setError('Select one case before attaching evidence. Upload is disabled in multi-case view.');
        return;
      }
      if (rejectedFiles.length > 0) {
        setError('Unsupported file type. Provide image/video evidence and optionally one .txt statement.');
        return;
      }

      const mediaFiles = acceptedFiles.filter((file) => MEDIA_EXTENSIONS.test(file.name));
      const textFiles = acceptedFiles.filter(
        (file) => TEXT_EXTENSION.test(file.name) || file.type === 'text/plain',
      );
      const unknownFiles = acceptedFiles.filter(
        (file) => !mediaFiles.includes(file) && !textFiles.includes(file),
      );

      if (unknownFiles.length > 0) {
        setError('Some dropped files are not supported evidence or .txt statements.');
        return;
      }

      if (mediaFiles.length > 0) {
        setQueue((prev) => [...prev, ...mediaFiles.map(makeQueueItem)]);
      }

      if (textFiles.length > 1) {
        setError('Only one .txt statement may be attached at a time. Using the first one.');
      }
      if (textFiles.length >= 1) {
        try {
          const loadedText = await textFiles[0].text();
          if (!mountedRef.current || dropCaseId !== currentCaseIdRef.current) return;
          if (!loadedText.trim()) {
            setError('The witness statement file is empty.');
            return;
          }
          setStatement(loadedText.trim());
        } catch (err) {
          if (mountedRef.current && dropCaseId === currentCaseIdRef.current) {
            setError('Could not read the statement file as text.');
          }
        }
      }

      if (mediaFiles.length === 0 && textFiles.length === 0) {
        setError('No supported evidence or statement file was selected.');
      }
    },
    [disabled],
  );

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    noClick: true,
    multiple: true,
    disabled: disabled || isProcessing,
    accept: {
      'image/*': ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'],
      'video/*': ['.mp4', '.avi', '.mov', '.mkv'],
      'text/plain': ['.txt'],
    },
  });

  const uploadOne = async (item, generation) => {
    setQueue((prev) =>
      prev.map((x) =>
        x.id === item.id ? { ...x, status: 'uploading', error: null } : x,
      ),
    );
    try {
      const requestCaseId = String(currentCaseIdRef.current || '');
      if (!requestCaseId) throw new Error('No active case');
      const formData = new FormData();
      formData.append('evidence_file', item.file, item.file.name);
      formData.append('statement_text', statement.trim());
      formData.append('case_id', requestCaseId);
      const response = await axios.post(`${RESEARCH_API_URL}/analyze_evidence/`, formData);
      if (generation !== lifecycleRef.current) return;
      const data = response.data;
      if (!data || data.status !== 'success') {
        throw new Error(data?.detail || 'Research API did not confirm success.');
      }
      setQueue((prev) =>
        prev.map((x) =>
          x.id === item.id ? { ...x, status: 'success', result: data } : x,
        ),
      );
      onUploadSuccess?.(data, {
        caseId: requestCaseId,
        requestGeneration: generation,
        stale: false,
      });
    } catch (err) {
      if (generation !== lifecycleRef.current) return;
      setQueue((prev) =>
        prev.map((x) =>
          x.id === item.id
            ? {
                ...x,
                status: 'failed',
                error:
                  err.response?.data?.detail || err.message || 'Evidence analysis failed.',
              }
            : x,
        ),
      );
    }
  };

  const handleAnalyzeAll = async () => {
    setError(null);
    const requestCaseId = String(currentCaseIdRef.current || '');
    if (disabled || !requestCaseId) {
      setError('Create or select a case before analysing evidence.');
      return;
    }
    if (!statement.trim()) {
      setError('Enter or upload the genuine witness statement. No default statement will be inserted.');
      return;
    }
    const pending = queue.filter((x) => x.status === 'pending' || x.status === 'failed');
    if (pending.length === 0) {
      setError('No pending evidence files to analyse.');
      return;
    }
    if (isProcessing) return;

    const generation = ++lifecycleRef.current;
    setIsProcessing(true);
    for (const item of pending) {
      if (generation !== lifecycleRef.current) break;
      await uploadOne(item, generation);
    }
    if (mountedRef.current && generation === lifecycleRef.current) {
      setIsProcessing(false);
    }
  };

  const handleRemove = (id) => {
    setQueue((prev) => prev.filter((x) => x.id !== id));
  };

  const handleReset = () => {
    lifecycleRef.current += 1;
    setQueue([]);
    setStatement('');
    setError(null);
    setIsProcessing(false);
    onClearAll?.();
  };

  const pendingCount = queue.filter((x) => x.status === 'pending' || x.status === 'failed').length;
  const successCount = queue.filter((x) => x.status === 'success').length;
  const failedCount = queue.filter((x) => x.status === 'failed').length;
  const totalCount = queue.length;

  return (
    <div className="bg-slate-800/95 backdrop-blur-md p-5 rounded-2xl border border-slate-700/80 shadow-2xl flex flex-col space-y-4">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-base font-black text-slate-100 flex items-center gap-2">
          <span className="p-1.5 bg-blue-500/10 border border-blue-500/30 rounded-lg text-blue-400">
            <UploadCloud size={18} />
          </span>
          Evidence Ingestion Engine
        </h3>
        <span
          className="text-[11px] font-mono text-blue-300 bg-blue-950/80 px-2.5 py-0.5 rounded-full border border-blue-800 truncate max-w-[170px]"
          title={caseName || caseId}
        >
          {caseName || (caseId ? `Case: ${caseId}` : 'No case selected')}
        </span>
      </div>

      {disabled && (
        <div className="p-3 rounded-xl border border-amber-800/70 bg-amber-950/30 text-amber-200 text-xs">
          Select a single case to enable evidence ingestion. Multi-case view is read-only.
        </div>
      )}

      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-xl p-5 flex flex-col items-center justify-center transition-all text-center ${
          disabled
            ? 'border-slate-700 bg-slate-950/50 text-slate-600 opacity-60'
            : isDragActive
              ? 'border-blue-500 bg-blue-950/30 text-blue-400'
              : 'border-slate-600 bg-slate-900/50 text-slate-400'
        }`}
      >
        <input {...getInputProps()} />
        {isProcessing ? (
          <div className="flex flex-col items-center text-blue-400 py-3">
            <Loader2 className="animate-spin mb-2" size={34} />
            <p className="font-bold text-sm text-slate-100">
              Processing {successCount + failedCount} of {totalCount}...
            </p>
            <p className="text-[10px] text-slate-400 mt-1">Analysis runs sequentially per file.</p>
          </div>
        ) : (
          <>
            <UploadCloud size={32} className="mb-2 text-blue-400" />
            <p className="font-bold text-xs text-slate-200">
              {isDragActive
                ? 'Drop files to attach'
                : 'Drop image/video evidence and an optional .txt statement'}
            </p>
            <p className="text-[10px] text-slate-400 mt-1">
              MP4, AVI, MOV, MKV, JPG, PNG, BMP, WEBP, TIF, TIFF, TXT
            </p>
            <button
              type="button"
              onClick={open}
              disabled={disabled || isProcessing}
              className="mt-3 px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-xs text-slate-200 hover:bg-slate-700 disabled:opacity-40"
            >
              Browse files
            </button>
          </>
        )}
      </div>

      {queue.length > 0 && (
        <div className="space-y-1.5 max-h-64 overflow-y-auto">
          <div className="text-[10px] uppercase font-bold text-slate-400 tracking-wider flex items-center justify-between">
            <span>Queue ({queue.length})</span>
            <span className="font-mono text-slate-500">
              {pendingCount} pending · {successCount} done · {failedCount} failed
            </span>
          </div>
          {queue.map((item) => (
            <div
              key={item.id}
              className={`flex items-start gap-2 bg-slate-900/80 border rounded-lg p-2 text-xs ${
                item.status === 'failed'
                  ? 'border-red-800/70'
                  : item.status === 'success'
                    ? 'border-emerald-800/70'
                    : 'border-slate-700'
              }`}
            >
              <div className="pt-0.5">{fileIcon(item.file)}</div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="truncate text-slate-100" title={item.file.name}>
                    {item.file.name}
                  </span>
                  <span className="text-slate-500 shrink-0">
                    {(item.file.size / (1024 * 1024)).toFixed(2)} MB
                  </span>
                </div>
                {item.status === 'uploading' && (
                  <p className="text-[10px] text-blue-300 flex items-center gap-1 mt-1">
                    <Loader2 size={11} className="animate-spin" /> Uploading...
                  </p>
                )}
                {item.status === 'success' && (
                  <p className="text-[10px] text-emerald-300 flex items-center gap-1 mt-1">
                    <CheckCircle2 size={11} /> {item.result?.assessment_status || 'INDETERMINATE'}
                  </p>
                )}
                {item.status === 'failed' && (
                  <p className="text-[10px] text-red-300 flex items-center gap-1 mt-1">
                    <AlertCircle size={11} /> {item.error}
                  </p>
                )}
              </div>
              {!isProcessing && item.status !== 'uploading' && (
                <button
                  type="button"
                  onClick={() => handleRemove(item.id)}
                  className="text-slate-500 hover:text-red-300 p-0.5 rounded"
                  title="Remove"
                >
                  <X size={13} />
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      <div className="space-y-1.5">
        <label className="text-[11px] font-bold text-slate-300 flex items-center justify-between">
          <span>Witness statement / transcript</span>
          <span className="text-slate-500 font-normal">
            {queue.length > 1 ? 'Applies to all queued files' : 'Required for cross-modal analysis'}
          </span>
        </label>
        <textarea
          value={statement}
          onChange={(event) => setStatement(event.target.value)}
          placeholder="Paste the exact witness statement. Do not paraphrase it for the analysis."
          rows={3}
          disabled={disabled || isProcessing}
          className="w-full bg-slate-900 border border-slate-700 rounded-xl p-2.5 text-xs text-slate-100 focus:outline-none focus:border-blue-500 placeholder-slate-500 disabled:opacity-50"
        />
      </div>

      <div className="flex gap-2 pt-1">
        <button
          type="button"
          onClick={handleAnalyzeAll}
          disabled={disabled || isProcessing || pendingCount === 0 || !statement.trim()}
          className="flex-1 py-2.5 px-3 rounded-xl text-xs font-bold flex items-center justify-center gap-1.5 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white"
        >
          <Sparkles size={14} />
          {pendingCount > 1 ? `Run ${pendingCount} Analyses` : 'Run Evidential Analysis'}
        </button>
        <button
          type="button"
          onClick={handleReset}
          disabled={disabled || isProcessing}
          className="py-2.5 px-3 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 text-slate-300 rounded-xl text-xs font-medium flex items-center gap-1"
        >
          <RotateCcw size={13} /> Reset
        </button>
      </div>

      {error && (
        <div
          role="alert"
          className="p-3 bg-red-950/70 border border-red-800 rounded-xl text-red-300 text-xs flex items-center gap-2"
        >
          <AlertCircle size={16} className="shrink-0 text-red-400" />
          <span>{error}</span>
        </div>
      )}

      {successCount > 0 && !isProcessing && (
        <div className="p-3 bg-slate-900 border border-slate-700 rounded-xl space-y-2 text-xs">
          <p className="font-bold text-slate-100">
            {successCount} of {queue.length} analyses completed.
          </p>
          {queue
            .filter((x) => x.status === 'success')
            .slice(0, 1)
            .map((item) => {
              const status = ['MATCH', 'CONTRADICTION', 'INDETERMINATE'].includes(
                item.result?.assessment_status,
              )
                ? item.result.assessment_status
                : 'INDETERMINATE';
              const graphStatus =
                item.result?.graph_status && typeof item.result.graph_status === 'object'
                  ? item.result.graph_status
                  : null;
              return (
                <div key={item.id} className="space-y-1">
                  <p className="text-slate-400">
                    Latest: <span className="text-slate-200">{item.file.name}</span> →{' '}
                    <span className="text-blue-300">{status}</span>
                  </p>
                  {graphStatus && (
                    <div
                      className={`flex items-start gap-2 rounded-lg p-2 ${
                        graphStatus.persisted
                          ? 'bg-emerald-950/40 text-emerald-200'
                          : 'bg-amber-950/40 text-amber-200'
                      }`}
                    >
                      {graphStatus.persisted ? (
                        <Database size={15} className="mt-0.5 shrink-0" />
                      ) : (
                        <AlertTriangle size={15} className="mt-0.5 shrink-0" />
                      )}
                      <span>
                        <strong>
                          {graphStatus.persisted ? 'Durable graph storage' : 'Volatile memory fallback'}:
                        </strong>{' '}
                        {graphStatus.message || 'Persistence status unavailable.'}
                      </span>
                    </div>
                  )}
                </div>
              );
            })}
        </div>
      )}
    </div>
  );
}