import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeftRight, House, Info, List, SlidersHorizontal, Signal, Wifi, BatteryFull } from "lucide-react";
import { assessPayment, fetchHealth, fetchScenarios } from "./api";
import { ACCOUNT, RECENT_ACTIVITY, SAVED_RECIPIENTS } from "./data";
import { ScenarioPanel, type JourneyMode } from "./ScenarioPanel";
import { clockTime, reference as newReference } from "./format";
import { JourneyRecorder } from "./journey";
import { Activity, Home } from "./screens/Home";
import { Checking, Hold, Verify, Warn, type Signals } from "./screens/Interventions";
import { AmountForm, PayeeForm, PaymentsMenu, PinPad, Review, type SavedRecipient } from "./screens/Payment";
import { Result } from "./screens/Result";
import type { ActivityItem, AssessResponse, Draft, JourneyEvent, Outcome, Scenario } from "./types";
import { TopBar } from "./ui";

type Screen = "home" | "activity" | "payments" | "payee" | "amount" | "review" | "pin" | "checking" | "warn" | "verify" | "hold" | "result";
type Transition = "fwd" | "back" | "fade" | "rise";
const TAB_SCREENS: Screen[] = ["home", "payments", "activity"];
const FLOW_PROGRESS: Partial<Record<Screen, number>> = { payee: 0.25, amount: 0.5, review: 0.75, pin: 1 };

// Screen -> journey event recorded on arrival (live mode). Same vocabulary as the ML features.
const SCREEN_EVENT: Partial<Record<Screen, string>> = {
  home: "home",
  activity: "browse",
  payments: "payments_menu",
  payee: "payee_new_form",
  review: "review_screen",
};
const PURCHASE_WORDS = /order|purchase|buy|marketplace|iphone|phone|ticket|seller|shop/i;
const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

const emptyDraft: Draft = { recipient: "", iban: "", bank: "", purpose: "", amount: 0, saved: false };

function draftFor(sc: Scenario): Draft {
  const cv = sc.customer_view;
  return { recipient: cv.recipient, iban: cv.iban, bank: cv.recipient_bank, purpose: cv.purpose, amount: sc.transaction.amount, saved: !sc.transaction.is_new_beneficiary };
}

export default function App() {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [scenarioId, setScenarioId] = useState<string | null>(null);
  const [screen, setScreen] = useState<Screen>("home");
  const [draft, setDraft] = useState<Draft>(emptyDraft);
  const [assessment, setAssessment] = useState<AssessResponse | null>(null);
  const [signals, setSignals] = useState<Signals | null>(null);
  const [outcome, setOutcome] = useState<Outcome>("sent");
  const [ref, setRef] = useState("");
  const [balance, setBalance] = useState(ACCOUNT.openingBalance);
  const [activity, setActivity] = useState<ActivityItem[]>(RECENT_ACTIVITY);
  const [mode, setMode] = useState<JourneyMode>("recorded");
  const [callActive, setCallActive] = useState(false);
  const [apiStatus, setApiStatus] = useState<"checking" | "ok" | "down">("checking");
  const [apiError, setApiError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [transition, setTransition] = useState<Transition>("fade");
  const [freshId, setFreshId] = useState<string | null>(null);
  const [, setTick] = useState(0);
  const recorder = useRef(new JourneyRecorder());
  const committed = useRef({ payee: false, amount: false });

  const scenario = useMemo(() => scenarios.find((s) => s.id === scenarioId) ?? null, [scenarios, scenarioId]);

  // ---------------------------------------------------------------- bootstrapping
  useEffect(() => {
    Promise.all([fetchScenarios(), fetchHealth()])
      .then(([sc, health]) => {
        setScenarios(sc);
        setApiStatus(health.models_loaded ? "ok" : "down");
        if (!health.models_loaded) setApiError("The API is running but models are not loaded. Run scripts/run_pipeline.py.");
        if (sc.length) startScenario(sc[0]);
      })
      .catch((e: Error) => {
        setApiStatus("down");
        setApiError(`Cannot reach the ScamShield API (${e.message}). Start it with: uvicorn app.api.main:app`);
      });
    const clock = setInterval(() => setTick((t) => t + 1), 30_000);
    return () => clearInterval(clock);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Switching to another window/tab = app switch in the journey (live mode only).
  useEffect(() => {
    const away = () => mode === "live" && recorder.current.background();
    const back = () => mode === "live" && recorder.current.foreground();
    const vis = () => (document.hidden ? away() : back());
    window.addEventListener("blur", away);
    window.addEventListener("focus", back);
    document.addEventListener("visibilitychange", vis);
    return () => {
      window.removeEventListener("blur", away);
      window.removeEventListener("focus", back);
      document.removeEventListener("visibilitychange", vis);
    };
  }, [mode]);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 2200);
    return () => clearTimeout(t);
  }, [toast]);

  // ---------------------------------------------------------------- navigation
  const startScenario = useCallback((sc: Scenario) => {
    setScenarioId(sc.id);
    setDraft(draftFor(sc));
    setAssessment(null);
    setSignals(null);
    setScreen("home");
    setTransition("fade");
    recorder.current.reset();
    recorder.current.record("home");
    committed.current = { payee: false, amount: false };
  }, []);

  const go = (to: Screen, opts: { back?: boolean } = {}) => {
    if (opts.back) recorder.current.record("back_navigation");
    const ev = SCREEN_EVENT[to];
    if (ev) recorder.current.record(ev);
    setTransition(opts.back ? "back" : TAB_SCREENS.includes(to) && TAB_SCREENS.includes(screen) ? "fade" : "fwd");
    setScreen(to);
  };
  const show = (to: Screen, t: Transition) => {
    setTransition(t);
    setScreen(to);
  };

  const back = (): (() => void) | undefined => {
    switch (screen) {
      case "payee": return () => go("payments", { back: true });
      case "amount": return () => go(draft.saved ? "payments" : "payee", { back: true });
      case "review": return () => go("amount", { back: true });
      case "pin": return () => go("review", { back: true });
      default: return undefined;
    }
  };

  const savedList: SavedRecipient[] = useMemo(() => {
    const list = [...SAVED_RECIPIENTS];
    if (scenario && !scenario.transaction.is_new_beneficiary) {
      const cv = scenario.customer_view;
      list.unshift({ recipient: cv.recipient, iban: cv.iban, bank: cv.recipient_bank });
    }
    return list;
  }, [scenario]);

  const bankFor = useCallback(
    (iban: string): string | null => {
      const code = iban.slice(4, 8);
      if (code === "KRVN") return "Karvan";
      const known = [...scenarios.map((s) => ({ iban: s.customer_view.iban, bank: s.customer_view.recipient_bank })), ...SAVED_RECIPIENTS];
      return known.find((k) => k.iban.slice(4, 8) === code)?.bank ?? null;
    },
    [scenarios],
  );

  // ---------------------------------------------------------------- payment
  const submitPayment = async () => {
    if (!scenario) return;
    recorder.current.record("confirm");
    show("checking", "fade");
    const events: JourneyEvent[] = mode === "recorded" ? scenario.journey_events : recorder.current.snapshot(callActive);
    const has = (t: string) => events.some((e) => e.event_type === t);
    setSignals({
      newRecipient: !draft.saved,
      pasted: has("payee_pasted"),
      switchedApps: has("app_background"),
      onCall: has("call_active"),
      purchase: PURCHASE_WORDS.test(draft.purpose),
    });
    const tx = { ...scenario.transaction, amount: draft.amount, is_new_beneficiary: !draft.saved };
    try {
      const [res] = await Promise.all([assessPayment(tx, events), delay(1500)]);
      setAssessment(res);
      setApiError(null);
      setRef(newReference());
      if (res.recommended_action === "allow") finish("sent");
      else show(res.recommended_action, "rise");
    } catch (e) {
      setApiError((e as Error).message);
      setAssessment(null);
      finish("error");
    }
  };

  const finish = (o: Outcome) => {
    setOutcome(o);
    const status: ActivityItem["status"] =
      o === "on_hold" ? "on_hold" : o === "stopped" ? "stopped" : o.startsWith("sent") ? "completed" : "cancelled";
    if (o !== "error") {
      if (o.startsWith("sent")) setBalance((b) => Math.round((b - draft.amount) * 100) / 100);
      const id = `p${Date.now()}`;
      setFreshId(id);
      setActivity((a) => [
        { id, title: draft.recipient, subtitle: `Today, ${clockTime()} · Transfer`, amount: -draft.amount, kind: "transfer", status },
        ...a,
      ]);
    }
    show("result", "rise");
  };

  const goHome = () => {
    if (scenario) startScenario(scenario);
  };

  // ---------------------------------------------------------------- render
  const inTabs = screen === "home" || screen === "activity" || screen === "payments";
  const titles: Partial<Record<Screen, string>> = { payee: "Transfer", amount: "Amount", review: "Review", pin: "Confirm" };

  let body: React.ReactNode = null;
  switch (screen) {
    case "home":
      body = <Home balance={balance} activity={activity} freshId={freshId} onTransfer={() => go("payments")} onUnavailable={() => setToast("This feature isn't available in this simulation.")} onSeeAll={() => go("activity")} />;
      break;
    case "activity":
      body = <Activity activity={activity} freshId={freshId} />;
      break;
    case "payments":
      body = (
        <PaymentsMenu
          saved={savedList}
          onNew={() => {
            const fresh = scenario && scenario.transaction.is_new_beneficiary ? draftFor(scenario) : { ...emptyDraft, amount: scenario?.transaction.amount ?? 0, purpose: "" };
            setDraft({ ...fresh, saved: false });
            go("payee");
          }}
          onSaved={(r) => {
            const sameAsScenario = scenario && r.iban === scenario.customer_view.iban;
            setDraft({ recipient: r.recipient, iban: r.iban, bank: r.bank, saved: true, purpose: sameAsScenario ? scenario.customer_view.purpose : "", amount: scenario?.transaction.amount ?? 0 });
            recorder.current.record("payee_select_saved");
            show("amount", "fwd");
          }}
        />
      );
      break;
    case "payee":
      body = (
        <PayeeForm
          draft={draft}
          bankFor={bankFor}
          onPaste={() => recorder.current.record("payee_pasted")}
          onContinue={(recipient, iban, bank, typed) => {
            const changed = recipient !== draft.recipient || iban !== draft.iban;
            if (committed.current.payee && changed) recorder.current.record("payee_edit");
            else if (typed && !recorder.current.has("payee_pasted")) recorder.current.record("payee_typed");
            committed.current.payee = true;
            setDraft({ ...draft, recipient, iban, bank });
            show("amount", "fwd");
          }}
        />
      );
      break;
    case "amount":
      body = (
        <AmountForm
          draft={draft}
          balance={balance}
          onContinue={(amount, purpose) => {
            if (!committed.current.amount) recorder.current.record("amount_entered");
            else if (amount !== draft.amount) recorder.current.record("amount_edit");
            committed.current.amount = true;
            setDraft({ ...draft, amount, purpose });
            go("review");
          }}
        />
      );
      break;
    case "review":
      body = <Review draft={draft} onConfirm={() => show("pin", "fwd")} />;
      break;
    case "pin":
      body = <PinPad amount={draft.amount} onComplete={submitPayment} />;
      break;
    case "checking":
      body = <Checking />;
      break;
    case "warn":
      body = <Warn draft={draft} signals={signals!} onCancel={() => finish("cancelled")} onContinue={() => finish("sent_after_warning")} />;
      break;
    case "verify":
      body = <Verify draft={draft} onStopped={() => finish("stopped")} onCancel={() => finish("cancelled")} onVerified={() => finish("sent_after_verification")} />;
      break;
    case "hold":
      body = <Hold draft={draft} reference={ref} onWait={() => finish("on_hold")} onCancel={() => finish("hold_cancelled")} />;
      break;
    case "result":
      body = <Result outcome={outcome} draft={draft} reference={ref} balance={balance} onDone={goHome} />;
      break;
  }

  const panel = (
    <ScenarioPanel
      scenarios={scenarios}
      scenarioId={scenarioId}
      onSelect={(id) => { const sc = scenarios.find((s) => s.id === id); if (sc) startScenario(sc); setSheetOpen(false); }}
      mode={mode}
      onMode={(m) => { setMode(m); if (scenario) startScenario(scenario); }}
      callActive={callActive}
      onCallActive={setCallActive}
      assessment={assessment}
      apiStatus={apiStatus}
      apiError={apiError}
      onReset={() => { setBalance(ACCOUNT.openingBalance); setActivity(RECENT_ACTIVITY); if (scenario) startScenario(scenario); setSheetOpen(false); }}
    />
  );

  return (
    <div className="stage">
      <div className="device">
        <div className="device-screen">
          <div className="statusbar" aria-hidden>
            <span className="num">{clockTime()}</span>
            <span className={`island ${screen === "checking" ? "active" : ""}`} />
            <span className="icons"><Signal size={15} /><Wifi size={15} /><BatteryFull size={20} /></span>
          </div>
          <div className="app">
            {!inTabs && screen !== "checking" && screen !== "result" && <TopBar title={titles[screen]} onBack={back()} progress={FLOW_PROGRESS[screen]} />}
            <main className="app-body" key={screen}>
              <div className={`page t-${transition}`}>{scenarios.length || apiStatus !== "down" ? body : <ApiDown message={apiError} />}</div>
            </main>
            {inTabs && (
              <nav className="tabbar" aria-label="Main">
                <button className="tab" aria-current={screen === "home" ? "page" : undefined} onClick={() => go("home")}><span className="ti"><House size={21} /></span>Home</button>
                <button className="tab" aria-current={screen === "payments" ? "page" : undefined} onClick={() => go("payments")}><span className="ti"><ArrowLeftRight size={21} /></span>Payments</button>
                <button className="tab" aria-current={screen === "activity" ? "page" : undefined} onClick={() => go("activity")}><span className="ti"><List size={21} /></span>Activity</button>
              </nav>
            )}
          </div>
          <div id="device-overlay" />
          {toast && <div className="toast" role="status"><Info size={16} style={{ flex: "0 0 auto" }} />{toast}</div>}
        </div>
      </div>
      <div className="panel-desktop">{panel}</div>

      <button className="demo-fab" onClick={() => setSheetOpen(true)}><SlidersHorizontal size={14} /> Scenarios</button>
      {sheetOpen && (
        <div className="sheet-backdrop" onClick={() => setSheetOpen(false)}>
          <div className="sheet" onClick={(e) => e.stopPropagation()}><div className="grabber" />{panel}</div>
        </div>
      )}
    </div>
  );
}

function ApiDown({ message }: { message: string | null }) {
  return (
    <div className="screen" style={{ justifyContent: "center", textAlign: "center" }}>
      <h1 className="h2">Service unavailable</h1>
      <p className="muted" style={{ margin: 0 }}>The app can't reach the ScamShield API.</p>
      {message && <p className="xs muted" style={{ margin: 0, overflowWrap: "anywhere" }}>{message}</p>}
    </div>
  );
}
