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
  relationship_context: {
    context_available: boolean;
    history_available: boolean;
    relationship_seen_before: boolean | null;
    relationship_first_seen: boolean | null;
    prior_interaction_count: number;
    prior_total_amount: number;
    prior_amount: { average: number; median: number; maximum: number } | null;
    current_amount_context: {
      amount_vs_prior_average: number | null;
      amount_vs_prior_median: number | null;
      amount_vs_prior_maximum: number | null;
      prior_empirical_percentile: number;
      exceeds_prior_relationship_maximum: boolean;
    } | null;
    steps_since_previous_interaction: number | null;
    baseline_is_limited: boolean;
    origin_network: {
      prior_unique_counterparty_count: number;
      prior_transaction_count: number;
      current_destination_is_new: boolean | null;
    };
    destination_network: {
      prior_unique_origin_count: number;
      prior_transaction_count: number;
      current_origin_is_new_for_destination: boolean | null;
    };
  };
  report: {
    summary: string;
    risk_assessment: { level: "LOW" | "MEDIUM" | "HIGH"; assessment: string };
    key_signals: ReportSignal[];
    behavioral_analysis: { summary: string; history_limitation: string | null };
    relationship_analysis: {
      summary: string;
      history_limitation: string | null;
      evidence_ids: string[];
    };
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
        <div className="phase"><span /> Relationship Intelligence · Phase 5</div>
      </header>

      <section className="hero">
        <p className="eyebrow">PAYMENT RISK INTELLIGENCE</p>
        <h1>Investigate the evidence<br />around every transaction.</h1>
        <p className="lede">
          ML scores risk. Deterministic intelligence explains history and relationships. Humans retain control.
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
        <article><span>01</span><h2>ML risk</h2><p>Frozen probability, classification threshold, and simulated policy.</p></article>
        <article><span>02</span><h2>Evidence</h2><p>Typed, deterministic, and auditable transaction signals.</p></article>
        <article><span>03</span><h2>Behavior</h2><p>Strictly prior origin activity and amount deviation.</p></article>
        <article><span>04</span><h2>Relationships</h2><p>Causal pair history and aggregate network novelty.</p></article>
        <article><span>05</span><h2>AI analysis</h2><p>Allowlisted context, explicit uncertainty, and advisory steps.</p></article>
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
                <h4>Deterministic Evidence</h4>
                <div className="signal-list">
                  {copilot.report.key_signals.map((signal) => (
                    <article key={`${signal.signal}-${signal.evidence_ids.join("-")}`}>
                      <span>{signal.importance}</span>
                      <div><strong>{signal.signal}</strong><p>{signal.explanation}</p></div>
                    </article>
                  ))}
                </div>
              </div>

              <div className="report-section relationship-section">
                <div className="relationship-heading">
                  <h4>Relationship Intelligence</h4>
                  <span>{!copilot.relationship_context.context_available
                    ? "Unavailable"
                    : copilot.relationship_context.relationship_seen_before
                      ? "Previously observed"
                      : "New relationship"}</span>
                </div>
                <div className="relationship-metrics">
                  <div><span>Prior interactions</span><strong>{copilot.relationship_context.prior_interaction_count}</strong></div>
                  <div><span>Prior average</span><strong>{copilot.relationship_context.prior_amount ? copilot.relationship_context.prior_amount.average.toLocaleString(undefined, { maximumFractionDigits: 2 }) : "—"}</strong></div>
                  <div><span>Prior maximum</span><strong>{copilot.relationship_context.prior_amount ? copilot.relationship_context.prior_amount.maximum.toLocaleString(undefined, { maximumFractionDigits: 2 }) : "—"}</strong></div>
                  <div><span>Vs prior average</span><strong>{copilot.relationship_context.current_amount_context?.amount_vs_prior_average != null ? `${copilot.relationship_context.current_amount_context.amount_vs_prior_average.toFixed(2)}×` : "—"}</strong></div>
                  <div><span>Origin counterparties</span><strong>{copilot.relationship_context.context_available ? copilot.relationship_context.origin_network.prior_unique_counterparty_count : "—"}</strong></div>
                  <div><span>Destination origins</span><strong>{copilot.relationship_context.context_available ? copilot.relationship_context.destination_network.prior_unique_origin_count : "—"}</strong></div>
                </div>
                <p>{copilot.report.relationship_analysis.summary}</p>
                {copilot.report.relationship_analysis.history_limitation ? <p className="limitation">{copilot.report.relationship_analysis.history_limitation}</p> : null}
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
