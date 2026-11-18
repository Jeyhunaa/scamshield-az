import { useState } from "react";
import { ArrowLeftRight, Banknote, Bell, Coffee, Eye, EyeOff, Landmark, MoreHorizontal, Plus, Receipt, ShoppingBag, ShoppingCart } from "lucide-react";
import { ACCOUNT, BANK_NAME, CUSTOMER } from "../data";
import { greeting, maskedIban, money } from "../format";
import type { ActivityItem } from "../types";
import { BrandMark, DunePattern, useCountUp } from "../ui";

const KIND_ICON = {
  transfer: ArrowLeftRight,
  grocery: ShoppingCart,
  salary: Banknote,
  bill: Receipt,
  coffee: Coffee,
  shopping: ShoppingBag,
} as const;

const STATUS_TEXT: Record<NonNullable<ActivityItem["status"]>, { text: string; color: string }> = {
  completed: { text: "Completed", color: "var(--ink-500)" },
  on_hold: { text: "On hold", color: "var(--warning)" },
  cancelled: { text: "Cancelled", color: "var(--ink-500)" },
  stopped: { text: "Stopped for your protection", color: "var(--danger)" },
};

export function ActivityList({ items, freshId }: { items: ActivityItem[]; freshId?: string | null }) {
  return (
    <div className="list stagger">
      {items.map((it) => {
        const Icon = KIND_ICON[it.kind];
        const struck = it.status === "cancelled" || it.status === "stopped";
        const inactive = struck || it.status === "on_hold";
        return (
          <div className={`row ${it.id === freshId ? "fresh" : ""}`} key={it.id}>
            <span className="avatar neutral sm">
              <Icon size={16} />
            </span>
            <div className="main">
              <div className="title">{it.title}</div>
              <div className="sub">
                {it.status ? <span className="status-tag" style={{ color: STATUS_TEXT[it.status].color }}>{STATUS_TEXT[it.status].text} · </span> : null}
                {it.subtitle}
              </div>
            </div>
            <div
              className={`num ${it.amount > 0 ? "amount-pos" : ""}`}
              style={{ fontWeight: 620, textDecoration: struck ? "line-through" : undefined, color: inactive ? "var(--ink-400)" : undefined }}
            >
              {it.amount > 0 ? "+" : "−"}
              {money(Math.abs(it.amount))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function Home({
  balance,
  activity,
  freshId,
  onTransfer,
  onUnavailable,
  onSeeAll,
}: {
  balance: number;
  activity: ActivityItem[];
  freshId: string | null;
  onTransfer: () => void;
  onUnavailable: () => void;
  onSeeAll: () => void;
}) {
  const shown = useCountUp(balance, 900);
  const [hidden, setHidden] = useState(false);
  return (
    <div className="screen">
      <div className="home-head stagger">
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <BrandMark size={34} />
          <div>
            <div className="muted small">{greeting()},</div>
            <div className="h2">{CUSTOMER.firstName}</div>
          </div>
        </div>
        <button className="icon-btn" aria-label="Notifications" onClick={onUnavailable}>
          <Bell size={20} />
        </button>
      </div>

      <div className="balance-card" style={{ animation: "riseIn 520ms var(--ease-out) both" }}>
        <DunePattern />
        <div className="top">
          <span className="label">{ACCOUNT.name} · {BANK_NAME}</span>
          <button className="eye" onClick={() => setHidden((h) => !h)} aria-label={hidden ? "Show balance" : "Hide balance"}>
            {hidden ? <Eye size={15} /> : <EyeOff size={15} />}
          </button>
        </div>
        <div className={`value num ${hidden ? "hidden-bal" : ""}`} aria-live="polite">
          {money(shown)}
          <small>AZN</small>
        </div>
        <div className="meta">
          <span className="num">{maskedIban(ACCOUNT.iban)}</span>
          <span>Available balance</span>
        </div>
      </div>

      <div className="quick stagger">
        <button className="primary" onClick={onTransfer}>
          <span className="qi"><ArrowLeftRight size={20} /></span>
          Transfer
        </button>
        <button onClick={onUnavailable}>
          <span className="qi"><Plus size={20} /></span>
          Top up
        </button>
        <button onClick={onUnavailable}>
          <span className="qi"><Landmark size={20} /></span>
          Pay bills
        </button>
        <button onClick={onUnavailable}>
          <span className="qi"><MoreHorizontal size={20} /></span>
          More
        </button>
      </div>

      <div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
          <p className="section-label">Recent activity</p>
          <button className="link-btn" onClick={onSeeAll}>See all</button>
        </div>
        <ActivityList items={activity.slice(0, 4)} freshId={freshId} />
      </div>
    </div>
  );
}

export function Activity({ activity, freshId }: { activity: ActivityItem[]; freshId: string | null }) {
  return (
    <div className="screen">
      <h1 className="h1" style={{ paddingTop: 8 }}>Activity</h1>
      <ActivityList items={activity} freshId={freshId} />
    </div>
  );
}
