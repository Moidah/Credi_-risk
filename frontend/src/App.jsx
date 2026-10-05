import React, { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

const palette = {
  ink: "#12181B",
  paper: "#F6F4EF",
  ledger: "#1F2E33",
  signal: "#B5482A",
  muted: "#5B6B70",
  line: "#D8D2C4",
};

function useApi(path) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API_URL}${path}`)
      .then((r) => r.json())
      .then(setData)
      .catch((e) => setError(e.message));
  }, [path]);

  return { data, error };
}

function MetricCard({ label, value, sub }) {
  return (
    <div style={styles.metricCard}>
      <div style={styles.metricLabel}>{label}</div>
      <div style={styles.metricValue}>{value}</div>
      {sub && <div style={styles.metricSub}>{sub}</div>}
    </div>
  );
}

export default function App() {
  const { data: metrics } = useApi("/api/metrics/latest");
  const { data: importance } = useApi("/api/feature-importance/latest");

  const best = metrics && metrics.length
    ? metrics.reduce((a, b) => (a.auc_roc > b.auc_roc ? a : b))
    : null;

  return (
    <div style={styles.page}>
      <header style={styles.header}>
        <div>
          <div style={styles.eyebrow}>Riesgo Crediticio</div>
          <h1 style={styles.title}>Panel de Scoring de Default</h1>
        </div>
      </header>

      <section style={styles.metricsRow}>
        <MetricCard
          label="Mejor modelo"
          value={best ? best.model_name.replace("_", " ") : "—"}
        />
        <MetricCard
          label="AUC-ROC"
          value={best ? best.auc_roc : "—"}
          sub="área bajo la curva ROC"
        />
        <MetricCard
          label="Tasa de default"
          value={best ? `${(best.default_rate * 100).toFixed(1)}%` : "—"}
        />
        <MetricCard
          label="Registros evaluados"
          value={best ? best.n_records?.toLocaleString("es") : "—"}
        />
      </section>

      <section style={styles.panel}>
        <h2 style={styles.panelTitle}>Variables más predictivas</h2>
        <div style={{ width: "100%", height: 360 }}>
          <ResponsiveContainer>
            <BarChart
              data={importance || []}
              layout="vertical"
              margin={{ left: 40, right: 24, top: 8, bottom: 8 }}
            >
              <CartesianGrid stroke={palette.line} horizontal={false} />
              <XAxis type="number" stroke={palette.muted} />
              <YAxis
                type="category"
                dataKey="feature_name"
                width={160}
                stroke={palette.muted}
                tick={{ fontSize: 12 }}
              />
              <Tooltip
                contentStyle={{
                  background: palette.ink,
                  border: "none",
                  color: palette.paper,
                }}
              />
              <Bar dataKey="importance" fill={palette.signal} radius={[0, 3, 3, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section style={styles.panel}>
        <h2 style={styles.panelTitle}>Historial de entrenamientos</h2>
        <table style={styles.table}>
          <thead>
            <tr>
              <th style={styles.th}>Modelo</th>
              <th style={styles.th}>AUC-ROC</th>
              <th style={styles.th}>Registros</th>
              <th style={styles.th}>Fecha</th>
            </tr>
          </thead>
          <tbody>
            {(metrics || []).map((m, i) => (
              <tr key={i}>
                <td style={styles.td}>{m.model_name}</td>
                <td style={styles.td}>{m.auc_roc}</td>
                <td style={styles.td}>{m.n_records?.toLocaleString("es")}</td>
                <td style={styles.td}>
                  {new Date(m.trained_at).toLocaleString("es")}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    background: palette.paper,
    color: palette.ink,
    fontFamily: "'Iowan Old Style', 'Georgia', serif",
    padding: "48px 64px",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "baseline",
    borderBottom: `1px solid ${palette.line}`,
    paddingBottom: 24,
    marginBottom: 40,
  },
  eyebrow: {
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: 12,
    letterSpacing: "0.04em",
    color: palette.signal,
    marginBottom: 8,
  },
  title: {
    fontSize: 34,
    fontWeight: 500,
    margin: 0,
    color: palette.ledger,
  },
  metricsRow: {
    display: "grid",
    gridTemplateColumns: "repeat(4, 1fr)",
    gap: 20,
    marginBottom: 48,
  },
  metricCard: {
    border: `1px solid ${palette.line}`,
    padding: "20px 24px",
    background: "#FFFFFF",
  },
  metricLabel: {
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: 11,
    color: palette.muted,
    marginBottom: 10,
  },
  metricValue: {
    fontSize: 28,
    color: palette.ledger,
    textTransform: "capitalize",
  },
  metricSub: {
    fontSize: 12,
    color: palette.muted,
    marginTop: 6,
  },
  panel: {
    marginBottom: 48,
  },
  panelTitle: {
    fontSize: 18,
    fontWeight: 500,
    color: palette.ledger,
    marginBottom: 20,
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
  },
  th: {
    textAlign: "left",
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: 11,
    color: palette.muted,
    borderBottom: `1px solid ${palette.line}`,
    padding: "8px 12px 8px 0",
  },
  td: {
    padding: "10px 12px 10px 0",
    borderBottom: `1px solid ${palette.line}`,
    fontSize: 14,
  },
};
