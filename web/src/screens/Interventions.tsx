import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Check, Clock, Info, MessageSquare, Phone, Send, ShieldCheck, ShoppingCart, TriangleAlert, UserPlus } from "lucide-react";
import { BANK_NAME, CUSTOMER } from "../data";
import { money } from "../format";
import type { Draft } from "../types";
import { BrandMark, Checkbox, Choice, KV, Notice, StatusHero } from "../ui";

/** Observable facts about this payment, used to choose relevant, plain-language guidance. */
export interface Signals {
  newRecipient: boolean;
  pasted: boolean;
  switchedApps: boolean;
  onCall: boolean;
  purchase: boolean;
}

const CHECK_STEPS = ["Confirming the recipient's account", "Reviewing the payment details", "Checking for signs of fraud"];

/** Animated security check shown while ScamShield assesses the payment. */
export function Checking() {
  const [stage, setStage] = useState(0);
  useEffect(() => {
    const timers = [setTimeout(() => setStage(1), 480), setTimeout(() => setStage(2), 960)];
    return () => timers.forEach(clearTimeout);
  }, []);
  return (
    <div className="screen" style={{ justifyContent: "center", alignItems: "center", textAlign: "center", gap: 20 }}>
      <div className="scan" aria-hidden>
        <svg className="ring" width="112" height="112" viewBox="0 0 112 112">
          <circle className="track" cx="56" cy="56" r="52" fill="none" strokeWidth="4" />
          <circle className="arc" cx="56" cy="56" r="52" fill="none" strokeWidth="4" />
        </svg>
        <div className="core"><ShieldCheck size={36} /></div>
      </div>
      <div>
        <h1 className="h2">Securing your payment</h1>
        <p className="muted" style={{ margin: "6px 0 0" }}>This only takes a moment.</p>
      </div>
      <ul className="steps" aria-live="polite">
        {CHECK_STEPS.map((s, i) => (
          <li key={s} className={i < stage ? "done" : i === stage ? "active" : ""}>
            <span className="si">{i < stage && <Check size={13} strokeWidth={3} />}</span>
            {s}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function Warn({ draft, signals, onCancel, onContinue }: { draft: Draft; signals: Signals; onCancel: () => void; onContinue: () => void }) {
  const [ack, setAck] = useState(false);
  const tips: { key: string; icon: React.ReactNode; text: React.ReactNode }[] = [];
  if (signals.onCall)
    tips.push({ key: "call", icon: <Phone size={16} />, text: <><b>Is someone on the phone with you?</b> We will never call and ask you to move money, not even to a "safe account".</> });
  if (signals.purchase)
    tips.push({ key: "buy", icon: <ShoppingCart size={16} />, text: <><b>Buying something online?</b> Use the marketplace's secure checkout. Sellers who insist on a bank transfer are a common scam, and the money is rarely recovered.</> });
  if (signals.pasted || signals.switchedApps)
    tips.push({ key: "msg", icon: <MessageSquare size={16} />, text: <><b>Did these details arrive in a chat or message?</b> Fraudsters send convincing account details. Confirm them another way before you pay.</> });
  if (signals.newRecipient)
    tips.push({ key: "new", icon: <UserPlus size={16} />, text: <><b>You haven't paid {draft.recipient} before.</b> Make sure you know and trust who you're paying.</> });
  tips.push({ key: "stop", icon: <Info size={16} />, text: <><b>If in doubt, stop.</b> Once sent, a transfer can be very hard to get back.</> });

  return (
    <div className="screen">
      <StatusHero icon={<TriangleAlert size={30} />} tone="warning" title="Check before you send" effect="attention">
        This payment has some features we often see in scams.
      </StatusHero>
      <div className="card" style={{ padding: "4px 16px", animation: "fadeUp 380ms 140ms var(--ease-out) both" }}>
        <KV k="To" v={draft.recipient} />
        <KV k="Amount" v={<span className="num">{money(draft.amount)} AZN</span>} />
      </div>
      <div className="stagger" style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {tips.slice(0, 3).map((t) => (
          <Notice key={t.key} icon={t.icon} tone="warning">{t.text}</Notice>
        ))}
      </div>
      <Checkbox checked={ack} onChange={setAck}>I've checked the details and I trust this recipient</Checkbox>
      <div className="actions">
        <button className="btn btn-primary" onClick={onCancel}>Cancel payment</button>
        <button className="btn btn-secondary" disabled={!ack} onClick={onContinue}>Send anyway</button>
      </div>
    </div>
  );
}

const PURPOSES = [
  "Paying for goods or services",
  "Family or friends",
  "Rent, deposit or bills",
  "Investment or savings",
  "Moving my money to a safe account",
  "Something else",
];

function Stepper({ step }: { step: 1 | 2 }) {
  return (
    <div className="stepper" aria-label={`Step ${step} of 2`}>
      <span>Step {step} of 2</span>
      <span className="bars">
        <span className="on" />
        <span className={step === 2 ? "on" : ""} />
      </span>
    </div>
  );
}

export function Verify({ draft, onStopped, onCancel, onVerified }: { draft: Draft; onStopped: () => void; onCancel: () => void; onVerified: () => void }) {
  const [stage, setStage] = useState<"questions" | "code">("questions");
  const [purpose, setPurpose] = useState<string | null>(null);
  const [contacted, setContacted] = useState<boolean | null>(null);
  const [secret, setSecret] = useState<boolean | null>(null);

  if (stage === "questions") {
    const complete = purpose !== null && contacted !== null && secret !== null;
    const next = () => {
      if (secret || purpose === "Moving my money to a safe account") onStopped();
      else setStage("code");
    };
    return (
      <div className="screen">
        <Stepper step={1} />
        <StatusHero icon={<ShieldCheck size={30} />} tone="info" title="A quick security check" effect="pulse-once">
          To keep your money safe, please answer three questions. It takes less than a minute.
        </StatusHero>
        <div className="question stagger">
          <div className="q">What is this payment for?</div>
          <div className="choice-list" role="radiogroup">
            {PURPOSES.map((p) => <Choice key={p} checked={purpose === p} onSelect={() => setPurpose(p)}>{p}</Choice>)}
          </div>
        </div>
        <YesNo q="Has anyone contacted you by phone, SMS or messenger and asked you to make this payment?" value={contacted} onChange={setContacted} />
        <YesNo q="Has anyone told you to ignore bank warnings, or to keep this payment secret?" value={secret} onChange={setSecret} />
        <div className="actions">
          <button className="btn btn-primary" disabled={!complete} onClick={next}>Continue</button>
          <button className="btn btn-secondary" onClick={onCancel}>Cancel payment</button>
        </div>
      </div>
    );
  }
  return <CodeStep draft={draft} contacted={!!contacted} onCancel={onCancel} onVerified={onVerified} />;
}

function YesNo({ q, value, onChange }: { q: string; value: boolean | null; onChange: (v: boolean) => void }) {
  return (
    <div className="question">
      <div className="q">{q}</div>
      <div className="choice-grid" role="radiogroup">
        <Choice checked={value === false} onSelect={() => onChange(false)}>No</Choice>
        <Choice checked={value === true} onSelect={() => onChange(true)}>Yes</Choice>
      </div>
    </div>
  );
}

function CodeStep({ draft, contacted, onCancel, onVerified }: { draft: Draft; contacted: boolean; onCancel: () => void; onVerified: () => void }) {
  const [code] = useState(() => String(Math.floor(Math.random() * 1_000_000)).padStart(6, "0"));
  const [digits, setDigits] = useState<string[]>(Array(6).fill(""));
  const [calledBack, setCalledBack] = useState(false);
  const [showSms, setShowSms] = useState(false);
  const [error, setError] = useState(0);
  const refs = useRef<(HTMLInputElement | null)[]>([]);
  // The notification renders at device level (outside the animated page) so it sits like a system banner.
  const overlay = document.getElementById("device-overlay");

  useEffect(() => {
    const t = setTimeout(() => setShowSms(true), 800);
    return () => clearTimeout(t);
  }, []);

  const setDigit = (i: number, v: string) => {
    const d = v.replace(/\D/g, "");
    if (d.length > 1) {
      const next = d.slice(0, 6).split("");
      setDigits([...next, ...Array(6 - next.length).fill("")]);
      refs.current[Math.min(next.length, 5)]?.focus();
    } else {
      const next = [...digits];
      next[i] = d;
      setDigits(next);
      if (d && i < 5) refs.current[i + 1]?.focus();
    }
    setError(0);
  };
  const autofill = () => {
    setDigits(code.split(""));
    refs.current[5]?.focus();
    setError(0);
  };
  const entered = digits.join("");
  const submit = () => (entered === code ? onVerified() : setError((n) => n + 1));

  return (
    <div className={`screen ${showSms ? "with-sms" : ""}`}>
      {showSms && overlay && createPortal(
        <div className="sms" role="status">
          <div className="from">
            <BrandMark size={20} />
            <span>MESSAGES · {BANK_NAME.toUpperCase()}</span>
            <span className="when">now</span>
          </div>
          <button className="code-btn" onClick={autofill} title="Tap to fill the code">{code}</button> is your code to confirm a transfer of{" "}
          {money(draft.amount)} AZN to {draft.recipient}. Never share this code with anyone, including bank staff.
        </div>,
        overlay,
      )}
      <Stepper step={2} />
      <StatusHero icon={<MessageSquare size={28} />} tone="info" title="Enter the code we sent">
        We've sent a 6-digit code to <span className="nowrap">{CUSTOMER.phoneMasked}</span>.
      </StatusHero>
      {contacted && (
        <>
          <Notice icon={<Phone size={16} />} tone="warning">
            <b>You told us someone asked you to make this payment.</b> Before continuing, call them on a number you already have, not one from the message, and make sure it's really them.
          </Notice>
          <Checkbox checked={calledBack} onChange={setCalledBack}>I've called them on a number I already know and confirmed the request</Checkbox>
        </>
      )}
      <div className={`otp ${error ? "shake" : ""}`} key={error}>
        {digits.map((d, i) => (
          <input
            key={i}
            ref={(el) => { refs.current[i] = el; }}
            value={d}
            className={d ? "filled" : ""}
            inputMode="numeric"
            autoComplete={i === 0 ? "one-time-code" : "off"}
            aria-label={`Digit ${i + 1}`}
            maxLength={6}
            onChange={(e) => setDigit(i, e.target.value)}
            onKeyDown={(e) => { if (e.key === "Backspace" && !d && i > 0) refs.current[i - 1]?.focus(); if (e.key === "Enter") submit(); }}
          />
        ))}
      </div>
      {error > 0 && <span className="error-text">That code doesn't match. Check the message and try again.</span>}
      {showSms && error === 0 && entered.length === 0 && <span className="hint" style={{ textAlign: "center" }}>Tip: tap the code in the message to fill it in.</span>}
      <div className="actions">
        <button className="btn btn-primary" disabled={entered.length < 6 || (contacted && !calledBack)} onClick={submit}>Verify and send</button>
        <button className="btn btn-secondary" onClick={onCancel}>Cancel payment</button>
      </div>
    </div>
  );
}

export function Hold({ draft, reference, onWait, onCancel }: { draft: Draft; reference: string; onWait: () => void; onCancel: () => void }) {
  const now = new Date().toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
  return (
    <div className="screen">
      <StatusHero icon={<ShieldCheck size={30} />} tone="info" title="Your payment is on hold" effect="radar">
        We've paused this payment for a quick safety review. Your money hasn't left your account.
      </StatusHero>
      <div className="card" style={{ padding: "4px 16px", animation: "fadeUp 380ms 140ms var(--ease-out) both" }}>
        <KV k="To" v={draft.recipient} />
        <KV k="Amount" v={<span className="num">{money(draft.amount)} AZN</span>} />
        <KV k="Reference" v={<span className="num">{reference}</span>} />
      </div>
      <div className="card card-pad" style={{ animation: "fadeUp 380ms 200ms var(--ease-out) both" }}>
        <p className="h3" style={{ marginBottom: 14 }}>What happens next</p>
        <div className="timeline">
          <div className="tl"><span className="dot tone-success"><Check size={13} strokeWidth={3} /></span><div><div className="t">Payment requested</div><div className="d">Today, {now}</div></div></div>
          <div className="tl"><span className="dot tone-info live"><Clock size={13} /></span><div><div className="t">Security review</div><div className="d">A specialist will call you within 30 minutes</div></div></div>
          <div className="tl"><span className="dot tone-neutral"><Send size={12} /></span><div><div className="t" style={{ color: "var(--ink-500)" }}>Released or cancelled</div><div className="d">We'll let you know as soon as it's done</div></div></div>
        </div>
      </div>
      <Notice icon={<Phone size={16} />} tone="info">
        We'll only call from the number on the back of your card. <b>We will never ask for your PIN, passwords or codes.</b>
      </Notice>
      <div className="actions">
        <button className="btn btn-primary" onClick={onWait}>OK, I'll wait for the call</button>
        <button className="btn btn-secondary" onClick={onCancel}>Cancel this payment</button>
      </div>
    </div>
  );
}
