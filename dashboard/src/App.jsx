import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis
} from "recharts";

const REFRESH_MS = 15000;

function fmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function Card({ title, children, subtitle }) {
  return (
    <section className="card">
      <div className="card-header">
        <h2>{title}</h2>
        {subtitle ? <span>{subtitle}</span> : null}
      </div>
      {children}
    </section>
  );
}

function App() {
  const [bundle, setBundle] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const res = await fetch(`/data/dashboard_bundle.json?ts=${Date.now()}`);
        if (!res.ok) throw new Error(`Failed to load dashboard data (${res.status})`);
        const data = await res.json();
        if (!cancelled) {
          setBundle(data);
          setError("");
        }
      } catch (err) {
        if (!cancelled) setError(err.message || "Failed to load dashboard data");
      }
    };

    load();
    const timer = setInterval(load, REFRESH_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const comparisonRows = bundle?.comparisonRows || [];
  const scatterRows = bundle?.scatterRows || [];
  const latencyRows = bundle?.latencyRows || [];
  const routingRows = bundle?.routingRows || [];
  const generatedAt = bundle?.generatedAt || "";

  const bestThroughput = useMemo(() => {
    if (!comparisonRows.length) return null;
    return [...comparisonRows].sort((a, b) => (b.tok_per_sec || 0) - (a.tok_per_sec || 0))[0];
  }, [comparisonRows]);

  return (
    <main className="container">
      <header className="top">
        <h1>LLM Inference Benchmark Dashboard</h1>
        <p>
          Results from Phase 5 benchmark artifacts and load tests.
          {generatedAt ? ` Last sync: ${generatedAt}` : ""}
        </p>
        {error ? <div className="error">{error}</div> : null}
      </header>

      <div className="grid">
        <Card
          title="Live Benchmark Comparison Table"
          subtitle={bestThroughput ? `Best throughput: ${bestThroughput.mode}` : ""}
        >
          <table>
            <thead>
              <tr>
                <th>Mode</th>
                <th>Avg Memory (GB)</th>
                <th>Avg TTFT (ms)</th>
                <th>Avg Tok/s</th>
                <th>Avg Quality</th>
              </tr>
            </thead>
            <tbody>
              {comparisonRows.map((row) => (
                <tr key={row.mode}>
                  <td>{row.mode}</td>
                  <td>{fmt(row.mem_gb)}</td>
                  <td>{fmt(row.ttft_ms, 1)}</td>
                  <td>{fmt(row.tok_per_sec, 1)}</td>
                  <td>{row.quality === null ? "-" : fmt(row.quality, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card title="Memory vs Quality Scatter Plot">
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={320}>
              <ScatterChart margin={{ top: 16, right: 16, bottom: 16, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" dataKey="mem_gb" name="Memory" unit=" GB" />
                <YAxis type="number" dataKey="quality" name="Quality" domain={[0, 5]} />
                <ZAxis type="number" dataKey="tok_per_sec" range={[100, 900]} name="Tok/s" />
                <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                <Legend />
                <Scatter name="Mode tradeoff" data={scatterRows} fill="#2f80ed" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Latency Distribution Chart">
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={latencyRows} margin={{ top: 16, right: 16, bottom: 16, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="users" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="generate_avg_ms" name="Avg Latency (ms)" stroke="#27ae60" strokeWidth={2} />
                <Line type="monotone" dataKey="generate_p95_ms" name="P95 Latency (ms)" stroke="#eb5757" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Routing Decision Log" subtitle={`Showing latest ${routingRows.length} entries`}>
          <div className="log">
            {routingRows.length === 0 ? (
              <p>No routing log entries found.</p>
            ) : (
              routingRows.map((row, idx) => (
                <div className="log-item" key={`${row.timestamp || "t"}-${idx}`}>
                  <span className="badge">{row.routing?.tier || "-"}</span>
                  <div>
                    <div className="log-top">
                      <strong>{row.routing?.precision || "unknown"}</strong>
                      <small>{row.timestamp || ""}</small>
                    </div>
                    <p>{row.prompt || "(no prompt captured)"}</p>
                    <small>
                      tok/s: {fmt(row.metrics?.tok_per_sec)} | ttft_ms: {fmt(row.metrics?.ttft_ms, 1)} | total_ms:{" "}
                      {fmt(row.metrics?.total_ms, 1)}
                    </small>
                  </div>
                </div>
              ))
            )}
          </div>
        </Card>
      </div>
    </main>
  );
}

export default App;

