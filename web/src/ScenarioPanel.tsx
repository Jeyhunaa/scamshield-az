import { ExternalLink, FlaskConical, RotateCcw } from "lucide-react";
import { DASHBOARD_URL } from "./data";
import type { AssessResponse, Scenario } from "./types";
import { ActionTag } from "./ui";

export type JourneyMode = "recorded" | "live";

export interface ScenarioPanelProps {
  scenarios: Scenario[];
  scenarioId: string | null;
  onSelect: (id: string) => void;
  mode: JourneyMode;
  onMode: (m: JourneyMode) => void;
  callActive: boolean;
  onCallActive: (v: boolean) => void;
  assessment: AssessResponse | null;
  apiStatus: "checking" | "ok" | "down";
  apiError: string | null;
  onReset: () => void;
}

const pct = (p: number) => (p < 0.1 ? `${(p * 100).toFixed(2)}%` : `${(p * 100).toFixed(1)}%`);
/** Position of a probability on a log scale from 0.01% to 100%. */
const logPos = (p: number) => Math.min(1, Math.max(0.02, (Math.log10(Math.max(p, 1e-4)) + 4) / 4));

function RiskMeter({ label, p, color }: { label: string; p: number; color: string }) {
  return (
    <div className="meter">
      <div className="row-l"><span>{label}</span><b>{pct(p)}</b></div>
      <div className="track"><div className="fill" style={{ width: `${logPos(p) * 100}%`, background: color }} /></div>
    </div>
  );
}

export function ScenarioPanel(p: ScenarioPanelProps) {
  const current = p.scenarios.find((s) => s.id === p.scenarioId);
  return (
    <aside className="panel" aria-label="Scenario controls">
      <div className="panel-head">
        <span className="mark">S</span>
        <div>
          <h1>ScamShield AZ</h1>
          <p className="lead">Customer experience</p>
        </div>
      </div>

      <div className="panel-block">
        <p className="section-label">Scenarios</p>
        <div className="scenario-list">
          {p.scenarios.map((s) => (
            <button key={s.id} className="scenario" aria-pressed={s.id === p.scenarioId} onClick={() => p.onSelect(s.id)}>
              <span className="kind" style={{ background: s.kind === "scam" ? "var(--danger)" : "var(--success)" }} title={s.kind} />
              <span>{s.title}</span>
              <ActionTag action={s.intended_action} small />
            </button>
          ))}
        </div>
        {current && <p className="muted small" style={{ margin: 0 }} key={current.id}>{current.story}</p>}
        <p className="xs muted" style={{ margin: 0 }}>Dot: legitimate (green) or scam (red). Tag: the outcome each scenario is designed to show.</p>
      </div>

      <div className="panel-block">
        <p className="section-label">In-app behaviour</p>
        <div className={`segmented ${p.mode === "live" ? "right" : ""}`}>
          <span className="thumb" />
          <button aria-pressed={p.mode === "recorded"} onClick={() => p.onMode("recorded")}>Recorded session</button>
          <button aria-pressed={p.mode === "live"} onClick={() => p.onMode("live")}>Live session</button>
        </div>
        <p className="muted small" style={{ margin: 0 }}>
          {p.mode === "recorded"
            ? "Replays the behaviour recorded for this scenario, so every run is repeatable."
            : "Scores how you actually use the app: pasting the IBAN, editing the amount, going back, or switching to another window."}
        </p>
        {p.mode === "live" && (
          <label className="switch">
            <span>Customer is on a phone call</span>
            <input type="checkbox" checked={p.callActive} onChange={(e) => p.onCallActive(e.target.checked)} />
          </label>
        )}
      </div>

      <div className="panel-block">
        <p className="section-label">ScamShield decision</p>
        {p.assessment ? (
          <div className="decision" key={`${p.assessment.enhanced_risk}-${p.assessment.recommended_action}`}>
            <div className="top">
              <span className="small muted">Recommended action</span>
              <ActionTag action={p.assessment.recommended_action} />
            </div>
            <RiskMeter label="ScamShield scam probability" p={p.assessment.enhanced_risk} color="var(--brand)" />
            <RiskMeter label="Existing transaction score" p={p.assessment.baseline_risk} color="var(--ink-400)" />
            <div className="meter"><div className="ticks"><span>0.01%</span><span>0.1%</span><span>1%</span><span>10%</span><span>100%</span></div></div>
            <p className="xs muted" style={{ margin: 0 }}>{p.assessment.decision_explanation}</p>
          </div>
        ) : (
          <p className="muted small" style={{ margin: 0 }}>Appears when the customer confirms a payment. Scores are never shown to the customer.</p>
        )}
      </div>

      <div className="panel-block">
        <div className="status-line">
          <span className={`led ${p.apiStatus === "ok" ? "ok" : ""}`} style={{ background: p.apiStatus === "ok" ? "var(--success)" : p.apiStatus === "down" ? "var(--danger)" : "var(--ink-400)" }} />
          {p.apiStatus === "ok" ? "Connected to ScamShield API" : p.apiStatus === "down" ? "ScamShield API unavailable" : "Connecting…"}
        </div>
        {p.apiError && <p className="error-text" style={{ margin: 0 }}>{p.apiError}</p>}
        <div className="link-row">
          <a href={DASHBOARD_URL} target="_blank" rel="noreferrer">Analyst dashboard <ExternalLink size={12} /></a>
          <a href="/docs" target="_blank" rel="noreferrer">API docs <ExternalLink size={12} /></a>
          <button className="link-btn" style={{ display: "inline-flex", alignItems: "center", gap: 4 }} onClick={p.onReset}>
            <RotateCcw size={12} /> Restart
          </button>
        </div>
        <div className="sim-note">
          <FlaskConical size={14} style={{ flex: "0 0 auto" }} />
          <span>Simulated: fictional bank, people and accounts. No real payments.</span>
        </div>
      </div>
    </aside>
  );
}
