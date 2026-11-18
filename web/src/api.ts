import type { AssessResponse, JourneyEvent, Scenario, TransactionContext } from "./types";

// Same origin as the FastAPI server (production) or proxied by Vite (development).
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body.slice(0, 300)}`);
  }
  return res.json() as Promise<T>;
}

export const fetchScenarios = () => request<Scenario[]>("/scenarios");

export const fetchHealth = () => request<{ status: string; models_loaded: boolean; model_version: string | null }>("/health");

export const assessPayment = (transaction: TransactionContext, journey_events: JourneyEvent[]) =>
  request<AssessResponse>("/assess", {
    method: "POST",
    body: JSON.stringify({ transaction, journey_events, hold_allowed: true }),
  });
