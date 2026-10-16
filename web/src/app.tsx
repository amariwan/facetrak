import React, { useEffect, useState } from 'react';

type Section = 'Overview' | 'Cameras' | 'Live' | 'Models' | 'Tracking' |
  'Hardware' | 'Automation' | 'Events' | 'Performance' | 'Settings' |
  'Security' | 'Diagnostics' | 'Backends' | 'Anonymize' | 'Licenses';
const SECTIONS: Section[] = ['Overview', 'Cameras', 'Live', 'Models',
  'Tracking', 'Hardware', 'Automation', 'Events', 'Performance',
  'Settings', 'Security', 'Diagnostics', 'Backends', 'Anonymize',
  'Licenses'];

const PATHS: Record<Section, string> = {
  Overview: '/status', Cameras: '/cameras', Live: '/metrics',
  Models: '/api/v1/models', Tracking: '/api/v1/tracking',
  Hardware: '/api/v1/hardware/discovery', Automation: '/api/v1/rules',
  Events: '/events', Performance: '/api/v1/profiler',
  Settings: '/system', Security: '/api/v1/devices',
  Diagnostics: '/api/v1/diagnostics',
  Backends: '/api/v1/face/backends', Anonymize: '/api/v1/anonymize/jobs',
  Licenses: '/api/v1/model-registry/license' };

function useApi(path: string) {
  const [data, setData] = useState<unknown>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let live = true;
    setLoading(true);
    fetch(path).then(async (r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      if (live) { setData(j); setError(null); }
    }).catch((e) => { if (live) setError(String(e)); })
      .finally(() => { if (live) setLoading(false); });
    return () => { live = false; };
  }, [path]);
  return { data, error, loading };
}

async function postJson(path: string, body: unknown): Promise<unknown> {
  const r = await fetch(path, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(`HTTP ${r.status}: ${t.slice(0, 300)}`);
  }
  return r.json();
}

function Panel({ section }: { section: Section }) {
  const { data, error, loading } = useApi(PATHS[section]);
  if (section === 'Backends') return <BackendsPanel />;
  if (section === 'Anonymize') return <AnonymizePanel />;
  if (loading) return <p role="status">Loading {section}…</p>;
  if (error) return (
    <div role="alert">
      <p>{section} unavailable (offline or camera stopped).</p>
      <pre>{error}</pre>
    </div>);
  return <pre aria-label={`${section} data`}>
    {JSON.stringify(data, null, 1)?.slice(0, 4000)}</pre>;
}

function BackendsPanel() {
  const { data, error, loading } = useApi('/api/v1/face/backends');
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  if (loading) return <p role="status">Loading Backends…</p>;
  if (error) return <div role="alert"><p>Backends unavailable.</p><pre>{error}</pre></div>;
  const d = data as {
    detector_backend: string; recognition_backend: string;
    liveness_backend: string;
    available: Record<string, boolean>;
    insightface: { enabled: boolean; available: boolean; provider: string | null;
      model_pack: string | null; init_error: string | null };
  };
  const switchBackend = async (patch: object) => {
    setBusy(true); setMsg(null);
    try {
      const res = await postJson('/api/v1/face/backends', patch);
      setMsg(`OK: ${JSON.stringify(res)}`);
    } catch (e) { setMsg(`Fehler: ${String(e)}`); }
    finally { setBusy(false); }
  };
  return (
    <div>
      <h2>Detektor / Erkennung / Liveness</h2>
      <p>Aktiv: Detektor <b>{d.detector_backend}</b> · Recognition{' '}
        <b>{d.recognition_backend}</b> · Liveness <b>{d.liveness_backend}</b></p>
      <p>Verfügbar: YuNet {d.available.yunet ? '✓' : '✗'} · SFace{' '}
        {d.available.sface ? '✓' : '✗'} · SCRFD{' '}
        {d.available.scrfd ? '✓' : '✗ (InsightFace-Modelle + Lizenz nötig)'} ·{' '}
        ArcFace {d.available.arcface ? '✓' : '✗ (InsightFace-Modelle + Lizenz nötig)'}</p>
      <p>InsightFace: {d.insightface.enabled ? 'aktiviert' : 'deaktiviert'} ·{' '}
        {d.insightface.available
          ? `bereit (${d.insightface.provider}, ${d.insightface.model_pack})`
          : `nicht bereit${d.insightface.init_error ? ` — ${d.insightface.init_error}` : ''}`}</p>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <button disabled={busy} onClick={() => switchBackend({ detector_backend: 'yunet' })}>
          YuNet aktivieren</button>
        <button disabled={busy} onClick={() => switchBackend({ detector_backend: 'scrfd' })}>
          SCRFD aktivieren</button>
        <button disabled={busy} onClick={() => switchBackend({ recognition_backend: 'sface' })}>
          SFace aktivieren</button>
        <button disabled={busy} onClick={() => switchBackend({ recognition_backend: 'arcface' })}>
          ArcFace aktivieren</button>
      </div>
      {msg && <pre role="status">{msg}</pre>}
      <h3>Registrierungs-Qualität</h3>
      <QualityMini />
      <h3>Modellstatus</h3>
      <ModelsMini />
    </div>);
}

function ModelsMini() {
  const { data, error, loading } = useApi('/api/v1/model-registry');
  if (loading) return <p role="status">Loading models…</p>;
  if (error) return <pre role="alert">{error}</pre>;
  return <pre aria-label="model registry">
    {JSON.stringify(data, null, 1)?.slice(0, 4000)}</pre>;
}

function QualityMini() {
  const { data, error, loading } = useApi('/api/v1/registration/quality');
  if (loading) return <p role="status">Loading quality…</p>;
  if (error) return <pre role="alert">{error}</pre>;
  const q = data as { capturing: boolean; buffered: number; required: number;
    threshold: number; recent: { quality: number; label: string;
      accepted: boolean; reasons: string[] }[] };
  return (
    <div>
      <p>Erfassung: {q.capturing ? 'läuft' : 'gestoppt'} · Samples{' '}
        {q.buffered}/{q.required} · Schwelle {q.threshold}</p>
      {q.recent.length === 0 && <p>Noch keine Samples bewertet.</p>}
      {q.recent.slice(-8).map((s, i) => (
        <div key={i}>{s.accepted ? '✓' : '✗'} {s.label} ({s.quality}){' '}
          {s.reasons.join(', ')}</div>))}
    </div>);
}

type Job = {
  id: string; status: string; progress: number;
  current_frame: number; total_frames: number;
  input: string; result: null | { output?: string; faces_redacted?: number;
    analysis_json?: string };
  error: string | null;
};

function AnonymizePanel() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [outputDir, setOutputDir] = useState('');
  const [method, setMethod] = useState('gaussian');
  const [busy, setBusy] = useState(false);
  const refresh = async () => {
    try {
      const r = await fetch('/api/v1/anonymize/jobs');
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setJobs(await r.json()); setError(null);
    } catch (e) { setError(String(e)); }
  };
  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 2000);
    return () => clearInterval(t);
  }, []);
  const start = async () => {
    setBusy(true); setError(null);
    try {
      await postJson('/api/v1/anonymize/jobs',
        { input, output_dir: outputDir, method });
      setInput(''); await refresh();
    } catch (e) { setError(String(e)); }
    finally { setBusy(false); }
  };
  const cancel = async (id: string) => {
    await fetch(`/api/v1/anonymize/jobs/${id}`, { method: 'DELETE' });
    await refresh();
  };
  return (
    <div>
      <h2>Video-Anonymisierung (Blur / Mosaic)</h2>
      <p>Alle erkannten Gesichter werden standardmäßig anonymisiert —
        keine Identifizierung nötig. Analyse-JSON ist editierbar,
        Rendern erfolgt ohne erneute Inferenz.</p>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <label>Videodatei (Server-Pfad){' '}
          <input aria-label="input video" value={input}
            onChange={(e) => setInput(e.target.value)} size={40} /></label>
        <label>Ausgabeverzeichnis{' '}
          <input aria-label="output dir" value={outputDir}
            onChange={(e) => setOutputDir(e.target.value)} size={30} /></label>
        <label>Methode{' '}
          <select aria-label="method" value={method}
            onChange={(e) => setMethod(e.target.value)}>
            <option value="gaussian">Gaussian Blur</option>
            <option value="mosaic">Mosaic</option>
          </select></label>
        <button disabled={busy || !input || !outputDir} onClick={start}>
          Job starten</button>
      </div>
      {error && <pre role="alert">{error}</pre>}
      <h3>Batch-Aufgaben</h3>
      {jobs.length === 0 && <p>Keine Aufgaben.</p>}
      {jobs.map((j) => (
        <div key={j.id} style={{ border: '1px solid #555', margin: 6, padding: 6 }}>
          <div><b>{j.id}</b> — {j.status} —{' '}
            {Math.round(j.progress * 100)}% ({j.current_frame}/{j.total_frames})</div>
          <progress value={j.progress} max={1} style={{ width: '100%' }} />
          {j.error && <pre role="alert">{j.error}</pre>}
          {j.result?.output && <div>Ausgabe: {j.result.output} ·
            Gesichter: {j.result.faces_redacted}</div>}
          {(j.status === 'analyzing' || j.status === 'rendering' ||
            j.status === 'queued') && (
            <button onClick={() => cancel(j.id)}>Abbrechen</button>)}
        </div>))}
    </div>);
}

export function App() {
  const [section, setSection] = useState<Section>('Overview');
  const [dark, setDark] = useState(true);
  return (
    <div data-theme={dark ? 'dark' : 'light'}
      style={{ background: dark ? '#111' : '#fff',
               color: dark ? '#eee' : '#111', minHeight: '100vh',
               fontFamily: 'system-ui, sans-serif' }}>
      <header style={{ display: 'flex', gap: 12, padding: 12 }}>
        <h1>FaceTrak 1.0</h1>
        <button onClick={() => setDark(!dark)}
          aria-label="toggle dark mode">{dark ? '☾' : '☀'}</button>
      </header>
      <nav aria-label="sections" style={{ display: 'flex', flexWrap: 'wrap',
        gap: 6, padding: 12 }}>
        {SECTIONS.map((s) => (
          <button key={s} onClick={() => setSection(s)}
            aria-current={s === section}>{s}</button>))}
      </nav>
      <main style={{ padding: 12 }}>
        {section === 'Live' && (
          <img src="/preview.mjpg" alt="live preview"
            style={{ maxWidth: '100%' }}
            onError={(e) => { (e.target as HTMLImageElement).style.display =
              'none'; }} />)}
        <Panel section={section} />
      </main>
    </div>);
}
