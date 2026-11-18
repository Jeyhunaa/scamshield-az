import { CircleAlert, Clock, Info, PhoneOff, ShieldAlert, X } from "lucide-react";
import { longDate, money, prettyIban } from "../format";
import type { Draft, Outcome } from "../types";
import { CheckDraw, KV, Notice, StatusHero, useCountUp } from "../ui";

function SentAmount({ amount }: { amount: number }) {
  const shown = useCountUp(amount, 800);
  return <span className="num">{money(shown)} AZN</span>;
}

export function Result({ outcome, draft, reference, balance, onDone }: { outcome: Outcome; draft: Draft; reference: string; balance: number; onDone: () => void }) {
  const amount = `${money(draft.amount)} AZN`;
  let body: React.ReactNode;

  if (outcome === "sent" || outcome === "sent_after_warning" || outcome === "sent_after_verification") {
    body = (
      <>
        <StatusHero icon={<CheckDraw size={44} />} tone="success" title="Money sent">
          <SentAmount amount={draft.amount} /> is on its way to {draft.recipient}.
        </StatusHero>
        <div className="card receipt" style={{ padding: "4px 16px 10px" }}>
          <KV k="Status" v={<span className="pill tone-success">Completed</span>} />
          <KV k="Date" v={longDate()} />
          <KV k="IBAN" v={<span className="num">{prettyIban(draft.iban)}</span>} />
          <KV k="Purpose" v={draft.purpose || "—"} />
          <KV k="Reference" v={<span className="num">{reference}</span>} />
        </div>
        {outcome === "sent_after_verification" && (
          <div style={{ marginTop: 6 }}>
            <Notice icon={<Info size={16} />} tone="info">Thanks for completing the security check. It helps us keep your account safe.</Notice>
          </div>
        )}
      </>
    );
  } else if (outcome === "on_hold") {
    body = (
      <>
        <StatusHero icon={<Clock size={30} />} tone="info" title="Payment under review" effect="radar">
          Your transfer of <span className="num">{amount}</span> to {draft.recipient} is safely on hold. We'll call you shortly.
        </StatusHero>
        <div className="card receipt" style={{ padding: "4px 16px 10px" }}>
          <KV k="Status" v={<span className="pill tone-info">On hold · in review</span>} />
          <KV k="Reference" v={<span className="num">{reference}</span>} />
          <KV k="Balance" v={<span className="num">{money(balance)} AZN (unchanged)</span>} />
        </div>
        <div style={{ marginTop: 6 }}>
          <Notice icon={<PhoneOff size={16} />} tone="info">If someone is pressuring you to make this payment, hang up. A genuine person or company will understand the wait.</Notice>
        </div>
      </>
    );
  } else if (outcome === "stopped") {
    body = (
      <>
        <StatusHero icon={<ShieldAlert size={30} />} tone="danger" title="We've stopped this payment" effect="pulse-once">
          Your answers suggest someone may be trying to trick you. No money has left your account.
        </StatusHero>
        <div className="stagger" style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <Notice icon={<ShieldAlert size={16} />} tone="danger"><b>We will never ask you to move money to a "safe account"</b> or to keep a payment secret.</Notice>
          <Notice icon={<PhoneOff size={16} />} tone="danger">If you're on a call about this payment, hang up now.</Notice>
          <Notice icon={<Info size={16} />} tone="danger">Call us on the number on the back of your card. We're available 24/7.</Notice>
        </div>
      </>
    );
  } else if (outcome === "error") {
    body = (
      <StatusHero icon={<CircleAlert size={30} />} tone="neutral" title="We couldn't complete this payment">
        Something went wrong on our side. No money has left your account. Please try again in a moment.
      </StatusHero>
    );
  } else {
    body = (
      <>
        <StatusHero icon={<X size={30} />} tone="neutral" title="Payment cancelled">
          Nothing was sent to {draft.recipient}. No money has left your account.
        </StatusHero>
        <Notice icon={<Info size={16} />} tone="info">Checking first is always the right call. You can make the payment later if you're sure.</Notice>
      </>
    );
  }

  return (
    <div className="screen">
      {body}
      <div className="actions">
        <button className="btn btn-primary" onClick={onDone}>Done</button>
      </div>
    </div>
  );
}
