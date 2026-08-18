import React from 'react';
import { UploadCloud, FileVideo, FileText, MapPin } from 'lucide-react';

/**
 * INGESTION ZONE COMPONENT (MODULE 3)
 * Drag-and-drop interface for detectives to upload case files.
 */
export default function IngestionZone() {
  return (
    <div className="bg-slate-800 p-4 rounded-xl border border-slate-700 shadow-lg">
      <h3 className="text-lg font-bold text-slate-100 mb-4">Evidence Ingestion</h3>
      
      <div className="border-2 border-dashed border-slate-600 rounded-xl p-8 flex flex-col items-center justify-center text-slate-400 hover:border-blue-500 hover:text-blue-400 transition-colors cursor-pointer">
        <UploadCloud size={48} className="mb-4" />
        <p className="font-semibold text-lg">Drag & Drop Evidence Files</p>
        <p className="text-sm mt-1">Supports MP4, JPG, WAV, PDF, & GPX</p>
      </div>

      <div className="flex gap-4 mt-6">
        <div className="flex-1 bg-slate-700 p-3 rounded-lg flex items-center gap-3">
          <FileVideo className="text-blue-400" />
          <span className="text-sm font-medium">3 CCTV Clips</span>
        </div>
        <div className="flex-1 bg-slate-700 p-3 rounded-lg flex items-center gap-3">
          <FileText className="text-green-400" />
          <span className="text-sm font-medium">2 Transcripts</span>
        </div>
        <div className="flex-1 bg-slate-700 p-3 rounded-lg flex items-center gap-3">
          <MapPin className="text-red-400" />
          <span className="text-sm font-medium">1 GPS Log</span>
        </div>
      </div>
    </div>
  );
}