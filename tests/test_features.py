"""Journey feature extraction, leakage-safe history features, generator and split."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from scamshield.data.generator import generate_dataset
from scamshield.data.preprocessing import (
    CONTEXT_COLUMNS,
    TRANSACTION_FEATURES,
    assign_split,
    compute_customer_context,
    derive_transaction_features,
)
from scamshield.features.journey import JOURNEY_FEATURES, extract_journey_features, extract_journey_features_frame

SESSION = [
    ("login", 0.0),
    ("call_active", 0.5),
    ("home", 2.0),
    ("browse", 5.0),
    ("payments_menu", 10.0),
    ("payee_new_form", 12.0),
    ("app_background", 14.0),
    ("app_foreground", 44.0),
    ("payee_pasted", 46.0),
    ("amount_entered", 50.0),
    ("amount_edit", 55.0),
    ("back_navigation", 58.0),
    ("payments_menu", 60.0),
    ("review_screen", 65.0),
    ("confirm", 80.0),
]


def _events(seq):
    return [{"event_type": e, "t_offset_s": t} for e, t in seq]


def test_journey_features_exact_values():
    f = extract_journey_features(_events(SESSION))
    assert set(f) == set(JOURNEY_FEATURES)
    assert f["session_duration_s"] == 80.0
    assert f["time_to_payment_s"] == 10.0
    assert f["n_payment_screen_visits"] == 2
    assert f["n_amount_edits"] == 1
    assert f["n_back_navigations"] == 1
    assert f["n_app_switches"] == 1
    assert f["background_time_s"] == 30.0
    assert f["background_share"] == pytest.approx(30 / 80)
    assert f["confirm_time_s"] == 15.0
    assert f["n_browse_events"] == 1
    assert f["payee_pasted"] == 1.0 and f["call_active"] == 1.0


def test_journey_order_invariant_and_unknown_events_ignored():
    shuffled = list(reversed(_events(SESSION))) + [{"event_type": "something_new", "t_offset_s": 30.0}]
    assert extract_journey_features(shuffled) == extract_journey_features(_events(SESSION))


def test_unmatched_background_closed_at_session_end():
    f = extract_journey_features(_events([("login", 0), ("app_background", 10), ("confirm", 25)]))
    assert f["background_time_s"] == 15.0


def test_confirm_time_uses_last_review_before_confirm():
    seq = [("login", 0), ("review_screen", 10), ("back_navigation", 12), ("review_screen", 20), ("confirm", 26)]
    assert extract_journey_features(_events(seq))["confirm_time_s"] == 6.0


def test_missing_events_give_nan_not_crash():
    f = extract_journey_features(_events([("login", 0), ("home", 3)]))
    assert math.isnan(f["confirm_time_s"]) and math.isnan(f["time_to_payment_s"])
    empty = extract_journey_features([])
    assert empty["session_duration_s"] == 0.0 and empty["n_app_switches"] == 0


def test_negative_offsets_rejected():
    with pytest.raises(ValueError):
        extract_journey_features(_events([("login", -1.0)]))


def test_batch_extraction_matches_single():
    rows = [("A", e, t) for e, t in SESSION] + [("B", "login", 0.0), ("B", "confirm", 9.0)]
    df = pd.DataFrame(rows, columns=["txn_id", "event_type", "t_offset_s"]).sample(frac=1, random_state=0)
    batch = extract_journey_features_frame(df).set_index("txn_id")
    single = extract_journey_features(_events(SESSION))
    for k in JOURNEY_FEATURES:
        assert batch.loc["A", k] == pytest.approx(single[k], nan_ok=True)
    assert batch.loc["B", "session_duration_s"] == 9.0


# ----------------------------------------------------------------- history features
def _tx_log() -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.to_datetime(["2025-01-01 10:00", "2025-01-01 20:00", "2025-01-05 09:00", "2025-01-05 10:00", "2025-01-01 12:00"])
    tx = pd.DataFrame(
        {
            "txn_id": ["t1", "t2", "t3", "t4", "u1"],
            "customer_id": ["c1", "c1", "c1", "c1", "c2"],
            "timestamp": ts,
            "amount": [100.0, 300.0, 50.0, 1000.0, 70.0],
            "beneficiary_id": ["b1", "b2", "b1", "b3", "b9"],
        }
    )
    cust = pd.DataFrame({"customer_id": ["c1", "c2"], "signup_date": pd.to_datetime(["2024-01-01", "2024-12-01"])})
    return tx, cust


def test_history_features_use_only_prior_transactions():
    tx, cust = _tx_log()
    ctx = compute_customer_context(tx, cust).set_index("txn_id")
    assert ctx.loc["t1", "prior_txn_count"] == 0 and pd.isna(ctx.loc["t1", "prior_avg_amount"])
    assert ctx.loc["t3", "prior_avg_amount"] == pytest.approx(200.0)
    assert ctx.loc["t4", "prior_max_amount"] == 300.0
    assert ctx.loc["t2", "txn_last_24h"] == 1           # t1 only; the current row is excluded
    assert ctx.loc["t4", "txn_last_24h"] == 1           # t3 within 24h, t1/t2 older
    assert ctx.loc["t4", "amount_last_24h"] == 50.0
    assert list(ctx.loc[["t1", "t2", "t3", "t4"], "is_new_beneficiary"]) == [1, 1, 0, 1]
    assert ctx.loc["t4", "n_distinct_beneficiaries"] == 2
    assert ctx.loc["u1", "prior_txn_count"] == 0        # customers are isolated


def test_future_transactions_do_not_change_past_features():
    tx, cust = _tx_log()
    before = compute_customer_context(tx, cust).set_index("txn_id")
    tx2 = tx.copy()
    tx2.loc[tx2["txn_id"] == "t4", "amount"] = 99999.0   # change the LAST transaction only
    after = compute_customer_context(tx2, cust).set_index("txn_id")
    cols = [c for c in CONTEXT_COLUMNS if c != "amount"]
    pd.testing.assert_frame_equal(before.loc[["t1", "t2", "t3"], cols], after.loc[["t1", "t2", "t3"], cols])


def test_derive_features_no_history_defaults():
    ctx = pd.DataFrame([{c: 0 for c in CONTEXT_COLUMNS} | {"amount": 500.0, "hour": 2, "prior_avg_amount": None, "prior_max_amount": None, "hours_since_last_txn": None}])
    f = derive_transaction_features(ctx).iloc[0]
    assert list(f.index) == TRANSACTION_FEATURES
    assert f["has_history"] == 0 and f["log_amount_to_avg"] == 0 and f["amount_to_max"] == 1.0
    assert f["is_night"] == 1.0
    assert not f.isna().any()


def test_derive_features_missing_column_raises():
    with pytest.raises(ValueError):
        derive_transaction_features(pd.DataFrame({"amount": [1.0]}))


# ----------------------------------------------------------------- generator + split
def test_generator_reproducible_and_no_latent_leak(small_cfg):
    a = generate_dataset(small_cfg, seed=7)
    b = generate_dataset(small_cfg, seed=7)
    c = generate_dataset(small_cfg, seed=8)
    pd.testing.assert_frame_equal(a.transactions, b.transactions)
    pd.testing.assert_frame_equal(a.events, b.events)
    assert not a.transactions.equals(c.transactions)
    assert not any(col.startswith("lat_") for df in (a.transactions, a.customers) for col in df.columns)
    assert set(a.events["txn_id"]) == set(a.transactions["txn_id"])
    assert 0.0 < a.transactions["is_scam"].mean() < 0.2


def test_signal_strength_only_changes_journeys(small_cfg):
    strong = generate_dataset(small_cfg, seed=3, signal_strength=1.0)
    none = generate_dataset(small_cfg, seed=3, signal_strength=0.0)
    pd.testing.assert_frame_equal(strong.transactions, none.transactions)  # controlled experiment
    assert not strong.events.equals(none.events)


def test_chronological_split(small_cfg):
    ts = pd.Series(pd.to_datetime(["2024-12-31 00:00", "2025-01-01 00:00", "2025-07-31 23:59", "2025-08-01", "2025-09-30", "2025-10-01", "2025-12-31"], format="ISO8601"))
    assert list(assign_split(ts, small_cfg)) == ["warmup", "train", "train", "val", "val", "test", "test"]


def test_split_dates_validated(small_cfg):
    bad = {**small_cfg, "split": {"train_end": "2025-10-01", "val_end": "2025-08-01"}}
    with pytest.raises(ValueError):
        assign_split(pd.Series(pd.to_datetime(["2025-02-01"])), bad)


def test_journey_features_separate_classes_on_average(small_cfg):
    d = generate_dataset(small_cfg, seed=11)
    jf = extract_journey_features_frame(d.events).merge(d.transactions[["txn_id", "is_scam"]], on="txn_id")
    means = jf.groupby("is_scam")[["n_app_switches", "call_active"]].mean()
    assert np.all(means.loc[1] > means.loc[0])
