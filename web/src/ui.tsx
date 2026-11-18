import { useEffect, useRef, useState, type ReactNode } from "react";
import { Check, ChevronLeft } from "lucide-react";
import type { Action } from "./types";

export type Tone = "success" | "warning" | "danger" | "info" | "neutral" | "sand";

/** Animates a number towards `value` (ease-out). Respects reduced-motion. */
export function useCountUp(value: number, duration = 700): number {
  const [shown, setShown] = useState(value);
  const from = useRef(value);
  useEffect(() => {
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    const start = from.current;
    if (reduce || start === value) {
      from.current = value;
      setShown(value);
      return;
    }
    let raf = 0;
    const t0 = performance.now();
    const tick = (t: number) => {
      const k = Math.min(1, (t - t0) / duration);
      const eased = 1 - Math.pow(1 - k, 3);
      setShown(start + (value - start) * eased);
      if (k < 1) raf = requestAnimationFrame(tick);
      else from.current = value;
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      from.current = value;
    };
  }, [value, duration]);
  return shown;
}

/** Karvan logomark: a caravan path over dunes. */
export function BrandMark({ size = 30 }: { size?: number }) {
  return (
    <span className="brand-mark" style={{ width: size, height: size, borderRadius: size * 0.3 }} aria-hidden>
      <svg width={size * 0.62} height={size * 0.62} viewBox="0 0 24 24" fill="none">
        <path d="M3 16.5c3-3.2 6-3.2 9 0s6 3.2 9 0" stroke="#fff" strokeWidth="2.2" strokeLinecap="round" />
        <path d="M3 11c3-3.2 6-3.2 9 0s6 3.2 9 0" stroke="#c9a867" strokeWidth="2.2" strokeLinecap="round" />
      </svg>
    </span>
  );
}

/** Dune line pattern for the account card (decorative, low opacity). */
export function DunePattern() {
  return (
    <svg className="pattern" width="260" height="200" viewBox="0 0 260 200" fill="none" aria-hidden>
      {Array.from({ length: 9 }).map((_, i) => (
        <path key={i} d={`M0 ${40 + i * 18} C 60 ${10 + i * 18}, 120 ${70 + i * 18}, 260 ${30 + i * 18}`} stroke={i % 3 === 1 ? "#c9a867" : "#ffffff"} strokeWidth="1.4" />
      ))}
    </svg>
  );
}

export function TopBar({ title, onBack, progress }: { title?: string; onBack?: () => void; progress?: number }) {
  return (
    <div className="topbar">
      {onBack ? (
        <button className="icon-btn" onClick={onBack} aria-label="Back">
          <ChevronLeft size={22} />
        </button>
      ) : (
        <span style={{ width: 38 }} />
      )}
      <div className="title">{title}</div>
      {progress !== undefined && (
        <div className="progress" aria-hidden>
          <span style={{ width: `${progress * 100}%` }} />
        </div>
      )}
    </div>
  );
}

export function StatusHero({ icon, tone, title, children, effect }: { icon: ReactNode; tone: Tone; title: string; children?: ReactNode; effect?: "radar" | "pulse-once" | "attention" }) {
  return (
    <div className="status-hero">
      <span className={`status-icon tone-${tone} ${effect ?? ""}`}>{icon}</span>
      <h1 className="h2">{title}</h1>
      {children && <p className="muted" style={{ margin: 0, fontSize: 15 }}>{children}</p>}
    </div>
  );
}

export function CheckDraw({ size = 40 }: { size?: number }) {
  return (
    <svg className="check-draw" width={size} height={size} viewBox="0 0 56 56" fill="none" aria-hidden>
      <circle cx="28" cy="28" r="26" stroke="currentColor" strokeWidth="2.5" opacity="0.35" />
      <path d="M17 29 l7.5 7.5 L39 21.5" stroke="currentColor" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function Notice({ icon, tone, children }: { icon: ReactNode; tone: Tone; children: ReactNode }) {
  return (
    <div className="notice">
      <span className={`ni tone-${tone}`}>{icon}</span>
      <div>{children}</div>
    </div>
  );
}

export function Checkbox({ checked, onChange, children }: { checked: boolean; onChange: (v: boolean) => void; children: ReactNode }) {
  return (
    <label className="check">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="box"><Check size={14} strokeWidth={3} /></span>
      <span>{children}</span>
    </label>
  );
}

export function Choice({ checked, onSelect, children }: { checked: boolean; onSelect: () => void; children: ReactNode }) {
  return (
    <button type="button" role="radio" aria-checked={checked} className="choice" onClick={onSelect}>
      <span className="dot" />
      <span>{children}</span>
    </button>
  );
}

export function KV({ k, v }: { k: string; v: ReactNode }) {
  return (
    <div className="kv">
      <span className="k">{k}</span>
      <span className="v">{v}</span>
    </div>
  );
}

const ACTION_TEXT: Record<Action, string> = { allow: "Allow", warn: "Warn", verify: "Verify", hold: "Hold" };

export function ActionTag({ action, small }: { action: Action; small?: boolean }) {
  return small ? (
    <span className={`intent-chip action-${action}`}>{ACTION_TEXT[action].toUpperCase()}</span>
  ) : (
    <span className={`action-tag action-${action}`}>{ACTION_TEXT[action].toUpperCase()}</span>
  );
}
