"""Leakage-safe transaction history features, model-feature derivation and chronological split.

Two layers keep training and serving consistent:

* `compute_customer_context` turns a raw transaction log into per-transaction *context*
  (counts, averages, velocities) using ONLY strictly earlier transactions of the same
  customer. In production this context would come from the bank's core systems.
* `derive_transaction_features` maps context -> model features. It is the single
  function used by the training pipeline, the API and the dashboard.

Labels of earlier transactions are never used as features: in reality scam reports arrive
days or weeks later, so "previously scammed" would leak future information.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scamshield.data.generator import SyntheticData
from scamshield.features.journey import JOURNEY_FEATURES, extract_journey_features_frame

# Raw context a caller must supply for one transaction (see API schema).
CONTEXT_COLUMNS: list[str] = [
    "amount",
    "hour",
    "is_new_beneficiary",
    "prior_txn_count",
    "prior_avg_amount",
    "prior_max_amount",
    "txn_last_24h",
    "txn_last_7d",
    "amount_last_24h",
    "hours_since_last_txn",
    "n_distinct_beneficiaries",
    "tenure_days",
]

# Features of the simulated "existing" transaction-based fraud model.
TRANSACTION_FEATURES: list[str] = [
    "log_amount",
    "log_amount_to_avg",
    "amount_to_max",
    "is_new_beneficiary",
    "has_history",
    "log_prior_txn_count",
    "txn_last_24h",
    "txn_last_7d",
    "log_amount_last_24h",
    "log_hours_since_last",
    "is_night",
    "log_n_distinct_beneficiaries",
    "log_tenure_days",
]

NO_HISTORY_HOURS = 24 * 90  # value used when a customer has no previous transfer


def compute_customer_context(transactions: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Per-transaction customer context computed from strictly earlier transactions."""
    df = transactions.merge(customers[["customer_id", "signup_date"]], on="customer_id", how="left", validate="m:1")
    df = df.sort_values(["customer_id", "timestamp", "txn_id"], kind="stable").reset_index(drop=True)
    g = df.groupby("customer_id", sort=False)

    df["prior_txn_count"] = g.cumcount()
    prior_sum = g["amount"].cumsum() - df["amount"]
    df["prior_avg_amount"] = (prior_sum / df["prior_txn_count"]).where(df["prior_txn_count"] > 0)
    df["prior_max_amount"] = g["amount"].transform(lambda s: s.cummax().shift(1))
    df["is_new_beneficiary"] = (~df.duplicated(["customer_id", "beneficiary_id"])).astype(int)
    df["n_distinct_beneficiaries"] = df.groupby("customer_id", sort=False)["is_new_beneficiary"].cumsum() - df["is_new_beneficiary"]
    df["hours_since_last_txn"] = g["timestamp"].diff().dt.total_seconds() / 3600.0

    # Rolling windows over *previous* transactions only (closed="left" excludes the current row).
    indexed = df.set_index("timestamp")
    roll = indexed.groupby("customer_id", sort=False)["amount"]
    df["txn_last_24h"] = roll.rolling("24h", closed="left").count().fillna(0).to_numpy()
    df["txn_last_7d"] = roll.rolling("7D", closed="left").count().fillna(0).to_numpy()
    df["amount_last_24h"] = roll.rolling("24h", closed="left").sum().fillna(0).to_numpy()

    df["hour"] = df["timestamp"].dt.hour
    df["tenure_days"] = (df["timestamp"] - df["signup_date"]).dt.days.clip(lower=0)
    return df.drop(columns=["signup_date"])


def derive_transaction_features(ctx: pd.DataFrame) -> pd.DataFrame:
    """Context columns -> model features. Shared by training and serving."""
    missing = [c for c in CONTEXT_COLUMNS if c not in ctx.columns]
    if missing:
        raise ValueError(f"missing context columns: {missing}")
    amount = ctx["amount"].astype(float)
    count = ctx["prior_txn_count"].astype(float)
    has_hist = (count > 0).astype(float)
    avg = ctx["prior_avg_amount"].astype(float)
    mx = ctx["prior_max_amount"].astype(float)
    hours_since = ctx["hours_since_last_txn"].astype(float).fillna(NO_HISTORY_HOURS).clip(0, NO_HISTORY_HOURS)
    hour = ctx["hour"].astype(int)

    out = pd.DataFrame(index=ctx.index)
    out["log_amount"] = np.log1p(amount)
    out["log_amount_to_avg"] = np.where(has_hist > 0, np.log((amount + 1) / (avg.fillna(0) + 1)), 0.0)
    out["amount_to_max"] = np.where(has_hist > 0, (amount / mx.fillna(1).clip(lower=1)).clip(upper=50), 1.0)
    out["is_new_beneficiary"] = ctx["is_new_beneficiary"].astype(float)
    out["has_history"] = has_hist
    out["log_prior_txn_count"] = np.log1p(count)
    out["txn_last_24h"] = ctx["txn_last_24h"].astype(float)
    out["txn_last_7d"] = ctx["txn_last_7d"].astype(float)
    out["log_amount_last_24h"] = np.log1p(ctx["amount_last_24h"].astype(float))
    out["log_hours_since_last"] = np.log1p(hours_since)
    out["is_night"] = ((hour >= 23) | (hour <= 5)).astype(float)
    out["log_n_distinct_beneficiaries"] = np.log1p(ctx["n_distinct_beneficiaries"].astype(float))
    out["log_tenure_days"] = np.log1p(ctx["tenure_days"].astype(float).clip(lower=0))
    return out[TRANSACTION_FEATURES]


def assign_split(timestamps: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """Chronological split label per transaction: warmup / train / val / test."""
    d, s = cfg["data"], cfg["split"]
    start, train_end, val_end, end = (pd.Timestamp(x) for x in (d["start"], s["train_end"], s["val_end"], d["end"]))
    if not start < train_end < val_end < end:
        raise ValueError("split dates must satisfy start < train_end < val_end < end")
    split = np.select(
        [timestamps < start, timestamps < train_end, timestamps < val_end, timestamps < end],
        ["warmup", "train", "val", "test"],
        default="out_of_range",
    )
    return pd.Series(split, index=timestamps.index)


def build_modeling_table(data: SyntheticData, cfg: dict[str, Any]) -> pd.DataFrame:
    """Full table: identifiers, label, context, transaction features, journey features, split.

    History is computed over warm-up + modeling period; warm-up rows are then dropped so
    every modeled transaction has up to 3 months of genuine prior history.
    """
    ctx = compute_customer_context(data.transactions, data.customers)
    tx_feats = derive_transaction_features(ctx)
    journey = extract_journey_features_frame(data.events)
    table = pd.concat([ctx, tx_feats.drop(columns=[c for c in tx_feats.columns if c in ctx.columns])], axis=1)
    table = table.merge(journey, on="txn_id", how="left", validate="1:1")
    if table[JOURNEY_FEATURES[0]].isna().any() and len(journey):
        n_missing = int(table["n_payment_screen_visits"].isna().sum())
        raise ValueError(f"{n_missing} transactions have no journey events")
    table["split"] = assign_split(table["timestamp"], cfg)
    table = table[table["split"].isin(["train", "val", "test"])]
    return table.sort_values(["timestamp", "txn_id"], kind="stable").reset_index(drop=True)
