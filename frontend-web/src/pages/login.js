import React, { useState } from 'react';
import { useRouter } from 'next/router';
import { ShieldAlert, ArrowRight, Info } from 'lucide-react';

export const DEMO_SESSION_KEY = 'crimevision.demo.session';

export default function Login() {
  const router = useRouter();
  const [isEntering, setIsEntering] = useState(false);

  const handleEnterDemo = () => {
    setIsEntering(true);
    try {
      sessionStorage.setItem(
        DEMO_SESSION_KEY,
        JSON.stringify({
          enteredAt: new Date().toISOString(),
          mode: 'demo',
          note: 'UI-only demo gate. No server authentication is performed.',
        })
      );
    } catch (error) {
      // sessionStorage may be unavailable (private mode). The route guard
      // in _app.js will still redirect to /login, which is the honest
      // behaviour: we do not claim authentication succeeded.
    }
    router.push('/');
  };

  return (
    <div className="min-h-screen bg-slate-900 flex items-center justify-center p-4">
      <div className="w-full max-w-md bg-slate-800 rounded-2xl shadow-2xl border border-slate-700 overflow-hidden">
        <div className="bg-slate-900 p-8 text-center border-b border-slate-700">
          <ShieldAlert size={48} className="text-blue-500 mx-auto mb-4" />
          <h1 className="text-3xl font-bold text-white tracking-tight">CrimeVision AI</h1>
          <p className="text-slate-400 mt-2 text-sm">Local research prototype</p>
          <p className="text-amber-300 text-xs mt-3">
            This page is not authentication. It is a UI-only demo gate.
          </p>
        </div>

        <div className="p-8 space-y-5">
          <div className="rounded-xl border border-amber-700/70 bg-amber-950/30 p-4 text-xs text-amber-200 space-y-2">
            <div className="flex items-start gap-2">
              <Info size={14} className="mt-0.5 shrink-0 text-amber-400" />
              <p>
                The CrimeVision AI research API is currently unauthenticated.
                No username or password is checked. This button only sets a
                client-side flag so the dashboard can be reached for local
                demonstration. Keep every service bound to loopback.
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={handleEnterDemo}
            disabled={isEntering}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 text-white font-bold py-3 px-4 rounded-lg transition-colors flex justify-center items-center gap-2"
          >
            <span>{isEntering ? 'Opening demo…' : 'Enter Demo Workspace'}</span>
            <ArrowRight size={16} />
          </button>

          <p className="text-[11px] text-center text-slate-500">
            See SRS §7.3. Real identity management and per-case authorization
            are not implemented in this prototype.
          </p>
        </div>
      </div>
    </div>
  );
}