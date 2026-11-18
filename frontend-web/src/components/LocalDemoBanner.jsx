import React from 'react';
import { ShieldAlert } from 'lucide-react';

export default function LocalDemoBanner() {
  return (
    <div
      role="status"
      className="mb-5 flex items-start gap-2 rounded-2xl border border-amber-700/70 bg-amber-950/30 px-4 py-3 text-xs text-amber-200"
    >
      <ShieldAlert size={16} className="mt-0.5 shrink-0 text-amber-400" />
      <p>
        <strong>Local demo mode — no authentication.</strong> All case data, evidence and
        analysis remain on this machine. Do not expose the research API to a network.
        See SRS §7.3 and §8.2.
      </p>
    </div>
  );
}