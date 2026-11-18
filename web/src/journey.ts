import type { JourneyEvent } from "./types";

/**
 * Records the in-app payment journey of the current session using the same event
 * vocabulary as src/scamshield/features/journey.py. Used in "Live session" mode, where
 * ScamShield scores how the presenter actually moved through the app.
 *
 * Switching to another window or tab is recorded as app_background / app_foreground.
 */
export class JourneyRecorder {
  private t0 = performance.now();
  private events: JourneyEvent[] = [];
  private backgroundSince: number | null = null;

  constructor() {
    this.reset();
  }

  reset(): void {
    this.t0 = performance.now();
    this.events = [{ event_type: "login", t_offset_s: 0 }];
    this.backgroundSince = null;
  }

  record(event_type: string): void {
    const t = Math.round((performance.now() - this.t0) / 100) / 10;
    this.events.push({ event_type, t_offset_s: Math.max(t, 0) });
  }

  background(): void {
    if (this.backgroundSince === null) {
      this.backgroundSince = performance.now();
      this.record("app_background");
    }
  }

  foreground(): void {
    if (this.backgroundSince !== null) {
      this.backgroundSince = null;
      this.record("app_foreground");
    }
  }

  has(event_type: string): boolean {
    return this.events.some((e) => e.event_type === event_type);
  }

  snapshot(callActive: boolean): JourneyEvent[] {
    const events = [...this.events];
    if (callActive) events.splice(1, 0, { event_type: "call_active", t_offset_s: 0.5 });
    return events;
  }
}
