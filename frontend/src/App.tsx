import { useEffect, useState } from "react";

type DatasetStatus = {
  status: "ready" | "not_prepared";
  rows: number | null;
  source_kind: string | null;
  message: string;
};

const apiBase = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

function App() {
  const [dataset, setDataset] = useState<DatasetStatus | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    fetch(`${apiBase}/dataset/status`)
      .then((response) => {
        if (!response.ok) throw new Error("API request failed");
        return response.json() as Promise<DatasetStatus>;
      })
      .then(setDataset)
      .catch(() => setOffline(true));
  }, []);

  return (
    <main className="shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="Fraudetect AI home">
          <span className="brand-mark">F</span>
          <span>FRAUDETECT <b>AI</b></span>
        </a>
        <div className="phase"><span /> Foundation · Phase 1</div>
      </header>

      <section className="hero">
        <p className="eyebrow">PAYMENT RISK INTELLIGENCE</p>
        <h1>Investigate the evidence<br />around every transaction.</h1>
        <p className="lede">
          ML detects the risk. AI investigates the evidence. Humans retain control.
        </p>
        <div className="status-card">
          <div>
            <span className="status-label">DATA PIPELINE</span>
            <strong>{offline ? "API OFFLINE" : dataset?.status === "ready" ? "READY" : "AWAITING DATA"}</strong>
          </div>
          <p>{offline ? "Start the FastAPI service to check pipeline status." : dataset?.message ?? "Checking validated dataset status…"}</p>
          {dataset?.rows ? <span className="row-count">{dataset.rows.toLocaleString()} rows prepared</span> : null}
        </div>
      </section>

      <section className="architecture" aria-label="Product intelligence layers">
        <article><span>01</span><h2>ML risk engine</h2><p>Measured fraud probability and configurable risk thresholds.</p></article>
        <article><span>02</span><h2>Relationship context</h2><p>Behavior, velocity, shared entities, and bounded graph evidence.</p></article>
        <article><span>03</span><h2>AI investigation</h2><p>Tool-retrieved evidence, uncertainty, and policy-bound recommendations.</p></article>
      </section>

      <footer>Demo recommendations only · No autonomous payment actions</footer>
    </main>
  );
}

export default App;

