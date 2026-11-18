"""Payment-journey behavioral feature extraction.

A payment journey is the ordered list of in-app events of the session in which a
transfer was created. The SAME function (`extract_journey_features`) is used for
training data, API requests and the dashboard, so feature definitions cannot drift
between training and serving.

Event schema: {"event_type": str, "t_offset_s": float}  (seconds since session start)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

# --- Event vocabulary -------------------------------------------------------------
LOGIN = "login"
HOME = "home"
BROWSE = "browse"                  # non-payment screens (accounts, cards, offers, ...)
PAYMENTS_MENU = "payments_menu"
PAYEE_SELECT_SAVED = "payee_select_saved"
PAYEE_NEW_FORM = "payee_new_form"
PAYEE_TYPED = "payee_typed"
PAYEE_PASTED = "payee_pasted"      # beneficiary IBAN/card pasted from clipboard
PAYEE_EDIT = "payee_edit"
AMOUNT_ENTERED = "amount_entered"
AMOUNT_EDIT = "amount_edit"
BACK_NAVIGATION = "back_navigation"
APP_BACKGROUND = "app_background"  # user switched away from the banking app
APP_FOREGROUND = "app_foreground"
CALL_ACTIVE = "call_active"        # OS reports an ongoing phone/VoIP call during the session
REVIEW_SCREEN = "review_screen"
CONFIRM = "confirm"

EVENT_TYPES: tuple[str, ...] = (
    LOGIN, HOME, BROWSE, PAYMENTS_MENU, PAYEE_SELECT_SAVED, PAYEE_NEW_FORM, PAYEE_TYPED,
    PAYEE_PASTED, PAYEE_EDIT, AMOUNT_ENTERED, AMOUNT_EDIT, BACK_NAVIGATION, APP_BACKGROUND,
    APP_FOREGROUND, CALL_ACTIVE, REVIEW_SCREEN, CONFIRM,
)

JOURNEY_FEATURES: list[str] = [
    "session_duration_s",
    "time_to_payment_s",
    "n_payment_screen_visits",
    "n_amount_edits",
    "n_payee_edits",
    "n_back_navigations",
    "n_app_switches",
    "background_time_s",
    "background_share",
    "confirm_time_s",
    "n_browse_events",
    "payee_pasted",
    "call_active",
]

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "session_duration_s": "Total session duration",
    "time_to_payment_s": "Time from login to the payments screen",
    "n_payment_screen_visits": "Visits to the payments screen",
    "n_amount_edits": "Amount edited after first entry",
    "n_payee_edits": "Beneficiary details edited",
    "n_back_navigations": "Back-navigations during the payment flow",
    "n_app_switches": "Switches away from the banking app",
    "background_time_s": "Time spent outside the banking app",
    "background_share": "Share of the session spent outside the app",
    "confirm_time_s": "Time spent on the review screen before confirming",
    "n_browse_events": "Non-payment screens browsed (normal browsing)",
    "payee_pasted": "Beneficiary pasted from clipboard",
    "call_active": "Phone call active during the session",
}


def extract_journey_features(events: Iterable[Mapping[str, object]]) -> dict[str, float]:
    """Summarise one session's event sequence into behavioral features.

    Robust to unsorted input, unknown event types (ignored) and missing events:
    time-based features that cannot be computed are returned as NaN (CatBoost
    handles missing values natively); counts default to 0.
    """
    rows = sorted(
        ((str(e["event_type"]), float(e["t_offset_s"])) for e in events),
        key=lambda x: x[1],
    )
    if any(t < 0 for _, t in rows):
        raise ValueError("t_offset_s must be non-negative")

    types = [r[0] for r in rows]
    times = [r[1] for r in rows]
    nan = float("nan")

    def count(kind: str) -> int:
        return sum(1 for t in types if t == kind)

    def first_time(kind: str) -> float:
        return next((t for k, t in rows if k == kind), nan)

    session_duration = (times[-1] - times[0]) if len(times) >= 2 else 0.0

    # Time outside the app: pair each background with the next foreground.
    # An unmatched background is closed at the end of the session.
    background_time = 0.0
    bg_start: float | None = None
    for kind, t in rows:
        if kind == APP_BACKGROUND and bg_start is None:
            bg_start = t
        elif kind == APP_FOREGROUND and bg_start is not None:
            background_time += t - bg_start
            bg_start = None
    if bg_start is not None and times:
        background_time += times[-1] - bg_start

    # Confirmation time: from the last review-screen view to the confirm event.
    confirm_t = first_time(CONFIRM)
    review_times = [t for k, t in rows if k == REVIEW_SCREEN and (np.isnan(confirm_t) or t <= confirm_t)]
    confirm_time = (confirm_t - review_times[-1]) if review_times and not np.isnan(confirm_t) else nan

    start_t = first_time(LOGIN)
    if np.isnan(start_t):
        start_t = times[0] if times else nan
    pay_t = first_time(PAYMENTS_MENU)
    time_to_payment = pay_t - start_t if not (np.isnan(pay_t) or np.isnan(start_t)) else nan

    return {
        "session_duration_s": session_duration,
        "time_to_payment_s": time_to_payment,
        "n_payment_screen_visits": float(count(PAYMENTS_MENU)),
        "n_amount_edits": float(count(AMOUNT_EDIT)),
        "n_payee_edits": float(count(PAYEE_EDIT)),
        "n_back_navigations": float(count(BACK_NAVIGATION)),
        "n_app_switches": float(count(APP_BACKGROUND)),
        "background_time_s": background_time,
        "background_share": background_time / session_duration if session_duration > 0 else 0.0,
        "confirm_time_s": confirm_time,
        "n_browse_events": float(count(BROWSE)),
        "payee_pasted": float(count(PAYEE_PASTED) > 0),
        "call_active": float(count(CALL_ACTIVE) > 0),
    }


def extract_journey_features_frame(events: pd.DataFrame) -> pd.DataFrame:
    """Batch version: long-format events (txn_id, event_type, t_offset_s) -> one row per txn_id.

    Deliberately calls `extract_journey_features` per session so that batch (training)
    and single-request (serving) features are computed by identical code.
    """
    required = {"txn_id", "event_type", "t_offset_s"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events frame missing columns: {sorted(missing)}")

    ev = events.sort_values(["txn_id", "t_offset_s"], kind="stable")
    txn_ids = ev["txn_id"].to_numpy()
    types = ev["event_type"].to_numpy()
    times = ev["t_offset_s"].to_numpy(dtype=float)
    boundaries = np.flatnonzero(txn_ids[1:] != txn_ids[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(txn_ids)]

    records = []
    for s, e in zip(starts, ends):
        session = [{"event_type": types[i], "t_offset_s": times[i]} for i in range(s, e)]
        feats = extract_journey_features(session)
        feats["txn_id"] = txn_ids[s]
        records.append(feats)
    out = pd.DataFrame.from_records(records, columns=["txn_id", *JOURNEY_FEATURES])
    return out
