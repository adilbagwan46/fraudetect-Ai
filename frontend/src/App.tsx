import { useEffect, useState } from "react";

type DatasetStatus = {
  status: "ready" | "not_prepared";
  rows: number | null;
  source_kind: string | null;
  message: string;
};

type ReportSignal = {
  signal: string;
  importance: "HIGH" | "MEDIUM" | "LOW" | "INFO";
  explanation: string;
  evidence_ids: string[];
};

type CopilotResponse = {
  provider: string;
  mode: "real_llm" | "deterministic_fallback";
  ai_available: boolean;
  model: string | null;
  fallback_reason: string | null;
  report: {
    summary: string;
    risk_assessment: { level: "LOW" | "MEDIUM" | "HIGH"; assessment: string };
    key_signals: ReportSignal[];
    behavioral_analysis: { summary: string; history_limitation: string | null };
    uncertainties: string[];
    recommended_actions: { action: string; reason: string }[];
    analyst_note: string;
    disclaimer: string;
  };
};

const apiBase = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

function App() {
  const [dataset, setDataset] = useState<DatasetStatus | null>(null);
  const [offline, setOffline] = useState(false);
  const [transactionReference, setTransactionReference] = useState("TX-000000001");
  const [copilot, setCopilot] = useState<CopilotResponse | null>(null);
  const [copilotError, setCopilotError] = useState<string | null>(null);
  const [copilotLoading, setCopilotLoading] = useState(false);

  useEffect(() => {
    fetch(`${apiBase}/dataset/status`)
      .then((response) => {
        if (!response.ok) throw new Error("API request failed");
        return response.json() as Promise<DatasetStatus>;
      })
      .then(setDataset)
      .catch(() => setOffline(true));
  }, []);

  async function runCopilot(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setCopilotLoading(true);
    setCopilotError(null);
    try {
      const response = await fetch(`${apiBase}/risk/investigate/copilot`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transaction_reference: transactionReference.trim() }),
      });
      const payload = await response.json() as CopilotResponse | { detail?: string };
      if (!response.ok) {
        throw new Error("detail" in payload && payload.detail ? payload.detail : "Investigation failed");
      }
      setCopilot(payload as CopilotResponse);
    } catch (error) {
      setCopilot(null);
      setCopilotError(error instanceof Error ? error.message : "Investigation failed");
    } finally {
      setCopilotLoading(false);
    }
  }

  return (
    <main className="shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="Fraudetect AI home">
          <span className="brand-mark">F</span>
          <span>FRAUDETECT <b>AI</b></span>
        </a>
        <div className="phase"><span /> Investigation Copilot · Phase 4</div>
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
        <article><span>02</span><h2>Behavioral context</h2><p>Strictly prior activity, amount deviation, recency, and type novelty.</p></article>
        <article><span>03</span><h2>AI investigation</h2><p>Allowlisted evidence, explicit uncertainty, and advisory next steps.</p></article>
      </section>

      <section className="copilot-workspace" aria-labelledby="copilot-title">
        <div className="copilot-intro">
          <p className="eyebrow">EVIDENCE-GROUNDED REVIEW</p>
          <h2 id="copilot-title">AI Investigation Copilot</h2>
          <p>Generate a structured analyst brief from an internal demo transaction reference. Raw identities and history never enter the report.</p>
          <form onSubmit={runCopilot} className="reference-form">
            <label htmlFor="transaction-reference">Transaction reference</label>
            <div>
              <input
                id="transaction-reference"
                value={transactionReference}
                onChange={(event) => setTransactionReference(event.target.value)}
                pattern="[A-Za-z0-9][A-Za-z0-9._:-]*"
                maxLength={64}
                required
              />
              <button disabled={copilotLoading} type="submit">
                {copilotLoading ? "Investigating…" : "Run Copilot"}
              </button>
            </div>
          </form>
          {copilotError ? <p className="copilot-error" role="alert">{copilotError}</p> : null}
        </div>

        <div className="report-panel" aria-live="polite">
          {!copilot ? (
            <div className="report-empty">
              <span>READY</span>
              <p>The deterministic demo fallback works without an API key. Real LLM mode is labeled when enabled server-side.</p>
            </div>
          ) : (
            <>
              <div className="report-heading">
                <div>
                  <span className={`risk-chip risk-${copilot.report.risk_assessment.level.toLowerCase()}`}>
                    {copilot.report.risk_assessment.level} RISK
                  </span>
                  <h3>Investigation Summary</h3>
                </div>
                <span className={`mode-chip ${copilot.ai_available ? "mode-live" : ""}`}>
                  {copilot.mode === "real_llm" ? "Real LLM" : "Demo Fallback"}
                </span>
              </div>
              <p className="report-summary">{copilot.report.summary}</p>

              <div className="report-section">
                <h4>Key Signals</h4>
                <div className="signal-list">
                  {copilot.report.key_signals.map((signal) => (
                    <article key={`${signal.signal}-${signal.evidence_ids.join("-")}`}>
                      <span>{signal.importance}</span>
                      <div><strong>{signal.signal}</strong><p>{signal.explanation}</p></div>
                    </article>
                  ))}
                </div>
              </div>

              <div className="report-grid">
                <div className="report-section">
                  <h4>Behavioral Analysis</h4>
                  <p>{copilot.report.behavioral_analysis.summary}</p>
                  {copilot.report.behavioral_analysis.history_limitation ? <p className="limitation">{copilot.report.behavioral_analysis.history_limitation}</p> : null}
                </div>
                <div className="report-section">
                  <h4>Limitations</h4>
                  <ul>{copilot.report.uncertainties.map((item) => <li key={item}>{item}</li>)}</ul>
                </div>
              </div>

              <div className="report-section actions">
                <h4>Recommended Actions</h4>
                <ol>{copilot.report.recommended_actions.map((item) => <li key={item.action}><strong>{item.action}</strong><span>{item.reason}</span></li>)}</ol>
              </div>
              <p className="report-disclaimer">{copilot.report.disclaimer}</p>
            </>
          )}
        </div>
      </section>

      <footer>Demo recommendations only · No autonomous payment actions</footer>
    </main>
  );
}

export default App;
