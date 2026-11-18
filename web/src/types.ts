// Mirrors the Pydantic schemas in app/api/schemas.py.

export type Action = "allow" | "warn" | "verify" | "hold";

export interface JourneyEvent {
  event_type: string;
  t_offset_s: number;
}

export interface TransactionContext {
  amount: number;
  hour: number;
  is_new_beneficiary: boolean;
  prior_txn_count: number;
  prior_avg_amount: number | null;
  prior_max_amount: number | null;
  txn_last_24h: number;
  txn_last_7d: number;
  amount_last_24h: number;
  hours_since_last_txn: number | null;
  n_distinct_beneficiaries: number;
  tenure_days: number;
}

export interface Scenario {
  id: string;
  title: string;
  kind: "legitimate" | "scam";
  intended_action: Action;
  story: string;
  customer_view: { recipient: string; iban: string; recipient_bank: string; purpose: string };
  transaction: TransactionContext;
  journey_events: JourneyEvent[];
}

export interface AssessResponse {
  baseline_risk: number;
  enhanced_risk: number;
  recommended_action: Action;
  baseline_policy_action: Action;
  decision_explanation: string;
  journey_features: Record<string, number | null>;
  model_version: string;
}

export interface Draft {
  recipient: string;
  iban: string;
  bank: string;
  purpose: string;
  amount: number;
  saved: boolean; // paying a saved (previously paid) recipient
}

export type Outcome =
  | "sent"
  | "sent_after_warning"
  | "sent_after_verification"
  | "cancelled"
  | "stopped"
  | "on_hold"
  | "hold_cancelled"
  | "error";

export interface ActivityItem {
  id: string;
  title: string;
  subtitle: string;
  amount: number; // negative = outgoing
  kind: "transfer" | "grocery" | "salary" | "bill" | "coffee" | "shopping";
  status?: "completed" | "on_hold" | "cancelled" | "stopped";
}
