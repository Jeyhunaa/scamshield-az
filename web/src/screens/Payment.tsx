import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { Check, ChevronRight, ChevronsRight, Delete, Landmark, Lock, UserPlus } from "lucide-react";
import { ACCOUNT } from "../data";
import { compactIban, initials, isValidAzIban, maskedIban, money, prettyIban } from "../format";
import type { Draft } from "../types";
import { KV } from "../ui";

export interface SavedRecipient {
  recipient: string;
  iban: string;
  bank: string;
}

export function PaymentsMenu({ saved, onNew, onSaved }: { saved: SavedRecipient[]; onNew: () => void; onSaved: (r: SavedRecipient) => void }) {
  return (
    <div className="screen">
      <h1 className="h1" style={{ paddingTop: 8 }}>Send money</h1>
      <div className="list">
        <button className="row" onClick={onNew}>
          <span className="avatar"><UserPlus size={18} /></span>
          <div className="main">
            <div className="title">New recipient</div>
            <div className="sub">Pay any IBAN in Azerbaijan</div>
          </div>
          <ChevronRight className="chev" size={18} color="var(--ink-400)" />
        </button>
      </div>
      <div>
        <p className="section-label">Saved recipients</p>
        <div className="list stagger">
          {saved.map((r) => (
            <button className="row" key={r.iban} onClick={() => onSaved(r)}>
              <span className="avatar">{initials(r.recipient)}</span>
              <div className="main">
                <div className="title">{r.recipient}</div>
                <div className="sub num">{r.bank} · {maskedIban(r.iban)}</div>
              </div>
              <ChevronRight className="chev" size={18} color="var(--ink-400)" />
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

export function PayeeForm({
  draft,
  bankFor,
  onContinue,
  onPaste,
}: {
  draft: Draft;
  bankFor: (iban: string) => string | null;
  onContinue: (recipient: string, iban: string, bank: string, typed: boolean) => void;
  onPaste: () => void;
}) {
  const [name, setName] = useState(draft.recipient);
  const [iban, setIban] = useState(prettyIban(draft.iban));
  const [typed, setTyped] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const valid = isValidAzIban(iban);
  const bank = valid ? bankFor(compactIban(iban)) ?? "Other bank in Azerbaijan" : null;

  const submit = () => {
    if (!name.trim()) return setError("Enter the recipient's full name.");
    if (!valid) return setError("Enter a valid IBAN: AZ followed by 26 characters.");
    onContinue(name.trim(), compactIban(iban), bank ?? "", typed);
  };

  return (
    <div className="screen">
      <div>
        <h1 className="h1">New recipient</h1>
        <p className="muted" style={{ margin: "6px 0 0" }}>Enter the details exactly as they appear on the invoice or account.</p>
      </div>
      <div className="field">
        <label htmlFor="name">Full name or company</label>
        <input id="name" className="input" value={name} autoComplete="off" onChange={(e) => { setName(e.target.value); setError(null); }} />
      </div>
      <div className="field">
        <label htmlFor="iban">IBAN</label>
        <div className="input-wrap">
          <input
            id="iban"
            className="input mono"
            value={iban}
            autoComplete="off"
            spellCheck={false}
            placeholder="AZ00 XXXX 0000 0000 0000 0000 0000"
            onPaste={onPaste}
            onChange={(e) => { setIban(e.target.value.toUpperCase()); setTyped(true); setError(null); }}
          />
          {valid && <span className="badge-ok" key={iban}><Check size={13} strokeWidth={3} /></span>}
        </div>
        {bank ? (
          <span className="bank-chip" key={bank}><Landmark size={13} /> {bank}</span>
        ) : (
          <span className="hint">We'll identify the bank once the IBAN is complete.</span>
        )}
      </div>
      {error && <span className="error-text" key={error}>{error}</span>}
      <div className="actions">
        <button className="btn btn-primary" onClick={submit}>Continue</button>
      </div>
    </div>
  );
}

export function AmountForm({ draft, balance, onContinue }: { draft: Draft; balance: number; onContinue: (amount: number, purpose: string) => void }) {
  const [value, setValue] = useState(draft.amount ? draft.amount.toFixed(2) : "");
  const [purpose, setPurpose] = useState(draft.purpose);
  const amount = Number(value.replace(/,/g, ""));
  const valid = Number.isFinite(amount) && amount >= 1 && amount <= balance;
  return (
    <div className="screen">
      <div className="row card" style={{ padding: "12px 14px" }}>
        <span className="avatar">{initials(draft.recipient)}</span>
        <div className="main">
          <div className="title">{draft.recipient}</div>
          <div className="sub num">{draft.bank} · {maskedIban(draft.iban)}</div>
        </div>
      </div>
      <div className="amount-entry">
        <input
          aria-label="Amount"
          inputMode="decimal"
          value={value}
          placeholder="0.00"
          onChange={(e) => setValue(e.target.value.replace(/[^\d.,]/g, ""))}
        />
        <span className="ccy">AZN</span>
      </div>
      <p className="muted small num" style={{ textAlign: "center", margin: 0 }}>
        Available: {money(balance)} AZN
      </p>
      <div className="field">
        <label htmlFor="purpose">Payment purpose</label>
        <input id="purpose" className="input" value={purpose} maxLength={60} onChange={(e) => setPurpose(e.target.value)} placeholder="e.g. Rent for December" />
      </div>
      {!valid && value !== "" && <span className="error-text">{amount > balance ? "This is more than your available balance." : "Enter an amount of at least 1.00 AZN."}</span>}
      <div className="actions">
        <button className="btn btn-primary" disabled={!valid} onClick={() => onContinue(Math.round(amount * 100) / 100, purpose.trim())}>
          Continue
        </button>
      </div>
    </div>
  );
}

/** Slide-to-confirm: deliberate gesture for sending money. Enter/Space also confirms (keyboard users). */
export function SlideToConfirm({ label, onConfirm }: { label: string; onConfirm: () => void }) {
  const track = useRef<HTMLDivElement>(null);
  const startX = useRef(0);
  const [x, setX] = useState(0);
  const [dragging, setDragging] = useState(false);
  const [done, setDone] = useState(false);
  const max = () => Math.max(1, (track.current?.clientWidth ?? 320) - 60);

  const complete = () => {
    if (done) return;
    setDone(true);
    setX(max());
    setTimeout(onConfirm, 420);
  };
  const down = (e: PointerEvent<HTMLButtonElement>) => {
    if (done) return;
    setDragging(true);
    startX.current = e.clientX - x;
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const move = (e: PointerEvent<HTMLButtonElement>) => {
    if (dragging) setX(Math.min(max(), Math.max(0, e.clientX - startX.current)));
  };
  const up = () => {
    if (!dragging) return;
    setDragging(false);
    if (x > max() * 0.8) complete();
    else setX(0);
  };
  const key = (e: KeyboardEvent<HTMLButtonElement>) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      complete();
    }
  };
  const progress = x / max();
  return (
    <div ref={track} className={`slider ${dragging ? "" : "snap"} ${done ? "done" : ""}`}>
      <div className="fill" style={{ width: x + 54 }} />
      <div className="label" style={{ opacity: done ? 1 : 1 - progress * 0.9 }}>
        <span>{done ? "Confirmed" : label}</span>
      </div>
      <button
        className="handle"
        style={{ transform: `translateX(${x}px)` }}
        aria-label={`${label}. Press Enter to confirm.`}
        onPointerDown={down}
        onPointerMove={move}
        onPointerUp={up}
        onPointerCancel={up}
        onKeyDown={key}
      >
        {done ? <Check size={22} strokeWidth={3} color="var(--success)" /> : <ChevronsRight size={22} />}
      </button>
    </div>
  );
}

export function Review({ draft, onConfirm }: { draft: Draft; onConfirm: () => void }) {
  return (
    <div className="screen">
      <div style={{ textAlign: "center", paddingTop: 6 }}>
        <span className="avatar lg" style={{ animation: "pop 420ms var(--ease-spring) both" }}>{initials(draft.recipient)}</span>
        <div className="muted small" style={{ marginTop: 10 }}>You're sending {draft.recipient}</div>
        <div className="num" style={{ fontSize: 38, fontWeight: 680, letterSpacing: "-0.03em", marginTop: 2 }}>
          {money(draft.amount)} <span style={{ fontSize: 18, color: "var(--ink-500)", fontWeight: 600, letterSpacing: 0 }}>AZN</span>
        </div>
        <span className="pill tone-neutral" style={{ marginTop: 6 }}>{draft.saved ? "Saved recipient" : "New recipient"}</span>
      </div>
      <div className="card" style={{ padding: "4px 16px", animation: "fadeUp 400ms 120ms var(--ease-out) both" }}>
        <KV k="IBAN" v={<span className="num">{prettyIban(draft.iban)}</span>} />
        <KV k="Bank" v={draft.bank} />
        <KV k="Purpose" v={draft.purpose || "—"} />
        <KV k="From" v={<span className="num">{ACCOUNT.name} · {ACCOUNT.iban.slice(-4)}</span>} />
        <KV k="Fee" v={<span className="num">0.00 AZN</span>} />
        <KV k="Arrives" v="Instantly" />
      </div>
      <p className="muted small" style={{ margin: 0 }}>
        Please check the recipient carefully. Money sent to the wrong account can be difficult to recover.
      </p>
      <div className="actions">
        <SlideToConfirm label={`Slide to send ${money(draft.amount)} AZN`} onConfirm={onConfirm} />
      </div>
    </div>
  );
}

const KEY_LETTERS: Record<string, string> = { "2": "ABC", "3": "DEF", "4": "GHI", "5": "JKL", "6": "MNO", "7": "PQRS", "8": "TUV", "9": "WXYZ" };

export function PinPad({ amount, onComplete }: { amount: number; onComplete: () => void }) {
  const [pin, setPin] = useState("");
  const done = useRef(false);

  useEffect(() => {
    if (pin.length === 4 && !done.current) {
      done.current = true;
      const t = setTimeout(onComplete, 420);
      return () => clearTimeout(t);
    }
  }, [pin, onComplete]);

  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (done.current) return;
      if (/^\d$/.test(e.key)) setPin((p) => (p.length < 4 ? p + e.key : p));
      if (e.key === "Backspace") setPin((p) => p.slice(0, -1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const press = (d: string) => !done.current && setPin((p) => (p.length < 4 ? p + d : p));
  return (
    <div className="screen">
      <div style={{ textAlign: "center", paddingTop: 22 }}>
        <span className="status-icon tone-info" style={{ width: 54, height: 54 }}><Lock size={22} /></span>
        <h1 className="h2" style={{ marginTop: 14 }}>Enter your PIN</h1>
        <p className="muted" style={{ margin: "4px 0 0" }}>to confirm <span className="num">{money(amount)}</span> AZN</p>
      </div>
      <div className={`pin-dots ${pin.length === 4 ? "ok" : ""}`} aria-label={`${pin.length} of 4 digits entered`}>
        {[0, 1, 2, 3].map((i) => <span key={i} className={i < pin.length ? "on" : ""} />)}
      </div>
      <p className="hint" style={{ textAlign: "center", margin: 0 }}>Any 4 digits · simulated sign-in</p>
      <div className="keypad">
        {["1", "2", "3", "4", "5", "6", "7", "8", "9"].map((d) => (
          <button key={d} className="key" onClick={() => press(d)}>
            {d}
            <small>{KEY_LETTERS[d] ?? " "}</small>
          </button>
        ))}
        <span className="key blank" />
        <button className="key" onClick={() => press("0")}>0</button>
        <button className="key fn" aria-label="Delete" onClick={() => setPin((p) => p.slice(0, -1))}><Delete size={22} /></button>
      </div>
    </div>
  );
}
