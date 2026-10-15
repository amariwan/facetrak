import React, { useEffect, useState } from 'react';

type Section = 'Overview' | 'Cameras' | 'Live' | 'Models' | 'Tracking' |
  'Hardware' | 'Automation' | 'Events' | 'Performance' | 'Settings' |
  'Security' | 'Diagnostics';
const SECTIONS: Section[] = ['Overview', 'Cameras', 'Live', 'Models',
  'Tracking', 'Hardware', 'Automation', 'Events', 'Performance',
  'Settings', 'Security', 'Diagnostics'];

const PATHS: Record<Section, string> = {
  Overview: '/status', Cameras: '/cameras', Live: '/metrics',
  Models: '/api/v1/models', Tracking: '/api/v1/tracking',
  Hardware: '/api/v1/hardware/discovery', Automation: '/api/v1/rules',
  Events: '/events', Performance: '/api/v1/profiler',
  Settings: '/system', Security: '/api/v1/devices',
  Diagnostics: '/api/v1/diagnostics' };

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

function Panel({ section }: { section: Section }) {
  const { data, error, loading } = useApi(PATHS[section]);
  if (loading) return <p role="status">Loading {section}…</p>;
  if (error) return (
    <div role="alert">
      <p>{section} unavailable (offline or camera stopped).</p>
      <pre>{error}</pre>
    </div>);
  return <pre aria-label={`${section} data`}>
    {JSON.stringify(data, null, 1)?.slice(0, 4000)}</pre>;
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

