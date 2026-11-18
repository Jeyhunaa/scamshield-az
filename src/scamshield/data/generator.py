"""Synthetic APP-scam dataset generator.

Generative story (all parameters are simulation assumptions):

1. Customers get latent traits: age, digital literacy, income, spending scale, activity
   rate, interaction tempo, and habits (pasting IBANs, switching apps, being on calls).
   Vulnerability to scams rises with age and lower digital literacy, plus noise.
2. Legitimate transfers follow a per-customer Poisson process. Beneficiaries are chosen
   from the customer's own history (preferential to frequent payees) or are new; ~4% of
   transfers are large one-off purchases (cars, deposits), often to new beneficiaries.
3. Scam *episodes* arrive with a hazard proportional to vulnerability and a seasonal
   campaign intensity. Each episode has a type (impersonation, investment, purchase,
   romance) that determines the number, timing, amount and beneficiary pattern of the
   fraudulent transfers, and a latent "coaching intensity" (how strongly a scammer is
   directing the victim in real time).
4. Every transfer gets an in-app event sequence. Legitimate journeys depend on the
   customer's tempo and habits; scam journeys are shifted by coaching intensity scaled by
   `journey_signal_strength`. Coaching is latent and never stored, so labels are not a
   deterministic function of any observed feature, and legitimate customers frequently
   show "scam-like" behavior (slow elderly users, IBANs pasted from messengers, calls).

The label is the episode membership — it is decided *before* any features are generated,
and features are then sampled conditionally. Nothing downstream sees latent variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from scamshield.features import journey as J

SCAM_TYPES = ("impersonation", "investment", "purchase", "romance")
# Coaching intensity distribution per scam type: Beta(a, b)
COACHING_BETA = {"impersonation": (5.0, 2.0), "investment": (3.0, 3.0), "purchase": (1.2, 5.0), "romance": (2.0, 4.0)}


@dataclass
class SyntheticData:
    customers: pd.DataFrame
    transactions: pd.DataFrame
    events: pd.DataFrame


def _seasonality(ts: pd.DatetimeIndex | pd.Series) -> np.ndarray:
    """Relative scam-campaign intensity by calendar date (mild seasonal wave + Q4 bump)."""
    month = pd.DatetimeIndex(ts).month.to_numpy()
    wave = 1.0 + 0.25 * np.sin(2 * np.pi * (month - 3) / 12.0)
    q4 = np.where(month >= 11, 1.25, 1.0)
    return wave * q4


def _generate_customers(rng: np.random.Generator, n: int, warmup_start: pd.Timestamp) -> pd.DataFrame:
    age = np.clip(rng.normal(42, 15, n), 18, 88).round()
    literacy = np.clip(rng.beta(5, 2, n) - 0.007 * (age - 35) + rng.normal(0, 0.08, n), 0.02, 0.99)
    income = rng.lognormal(np.log(1100), 0.55, n)
    vulnerability = np.exp(1.1 * (1 - literacy) + 0.02 * (age - 45) + rng.normal(0, 0.5, n))
    vulnerability /= vulnerability.mean()
    tenure_days = rng.exponential(900, n).round() + 30
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "age": age.astype(int),
            "digital_literacy": literacy.round(3),
            "monthly_income": income.round(2),
            "signup_date": warmup_start - pd.to_timedelta(tenure_days, unit="D"),
            # lat_* = latent traits used only by the simulator; dropped before saving
            "lat_vulnerability": vulnerability,
            "lat_spend_scale": income * rng.lognormal(np.log(0.12), 0.5, n),
            "lat_txn_rate": rng.gamma(2.0, 0.5, n),  # multiplied by the configured mean rate
            "lat_p_new_payee": np.clip(rng.beta(2, 10, n) + 0.08 * (age < 30), 0.02, 0.6),
            "lat_tempo": rng.lognormal(0.012 * (age - 40) - 0.6 * (literacy - 0.65), 0.2),
            "lat_call_habit": rng.beta(1.0, 22.0, n),
            "lat_paste_habit": rng.beta(2.0, 3.0, n),
            "lat_switch_habit": rng.gamma(1.5, 0.25, n),
            "lat_browse_habit": rng.gamma(2.0, 0.6, n),
        }
    )


def _sample_hours(rng: np.random.Generator, n: int, profile: str) -> np.ndarray:
    hours = np.arange(24)
    if profile == "day":
        w = np.exp(-0.5 * ((hours - 13) / 3.5) ** 2) + 0.03
    elif profile == "evening":
        w = np.exp(-0.5 * ((hours - 21) / 3.0) ** 2) + np.exp(-0.5 * ((hours - 1) / 1.5) ** 2) * 0.4 + 0.05
    else:  # "normal" legitimate activity
        w = np.exp(-0.5 * ((hours - 12) / 4.0) ** 2) + 0.6 * np.exp(-0.5 * ((hours - 19) / 2.0) ** 2) + 0.02
    return rng.choice(hours, size=n, p=w / w.sum())


def _random_times(rng, n, start, end, hour_profile) -> pd.DatetimeIndex:
    days = rng.integers(0, (end - start).days, n)
    hours = _sample_hours(rng, n, hour_profile)
    secs = rng.integers(0, 3600, n)
    return start + pd.to_timedelta(days, unit="D") + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(secs, unit="s")


def _legit_transactions(rng, customers, start, end, rate_mean) -> pd.DataFrame:
    months = (end - start).days / 30.44
    rows = []
    for c in customers.itertuples(index=False):
        n = rng.poisson(c.lat_txn_rate * rate_mean * months)
        if n == 0:
            continue
        ts = _random_times(rng, n, start, end, "normal")
        big = rng.random(n) < 0.04
        amount = rng.lognormal(np.log(c.lat_spend_scale), 0.7, n)
        amount = np.where(big, amount * rng.lognormal(np.log(8.0), 0.5, n), amount)
        for t, a, b in zip(ts, amount, big):
            rows.append((c.customer_id, t, max(1.0, round(float(a), 2)), bool(b)))
    return pd.DataFrame(rows, columns=["customer_id", "timestamp", "amount", "lat_big_purchase"])


def _scam_transactions(rng, customers, start, end, base_rate, prevalence_multiplier) -> pd.DataFrame:
    months = (end - start).days / 30.44
    rows = []
    episode_counter = 0
    mule_counter = 0
    for c in customers.itertuples(index=False):
        lam = base_rate * prevalence_multiplier * c.lat_vulnerability * months
        n_ep = min(rng.poisson(lam), 2)
        for _ in range(n_ep):
            # Episode start drawn from seasonal intensity by rejection sampling
            while True:
                t0 = _random_times(rng, 1, start, end, "normal")[0]
                if rng.random() < _seasonality(pd.DatetimeIndex([t0]))[0] / 1.6:
                    break
            age = c.age
            w = np.array([0.35 + 0.30 * (age > 55), 0.30 + 0.20 * (age < 40), 0.25 + 0.10 * (age < 35), 0.10 + 0.10 * (age > 45)])
            stype = SCAM_TYPES[rng.choice(4, p=w / w.sum())]
            coaching = rng.beta(*COACHING_BETA[stype])
            episode_counter += 1
            mule_counter += 1
            mule = f"M{mule_counter:05d}"

            if stype == "impersonation":   # "safe account" / bank or police impersonation
                n_pay = 1 + rng.binomial(2, 0.3)
                gaps_h = rng.exponential(1.5, n_pay)
                amounts = c.monthly_income * rng.lognormal(np.log(1.8), 0.6, n_pay)
                hour_profile = "day"
                same_mule_p = 0.8
            elif stype == "investment":
                n_pay = min(1 + rng.poisson(1.5), 5)
                gaps_h = rng.exponential(10 * 24, n_pay)
                first = c.lat_spend_scale * rng.lognormal(np.log(2.0), 0.5)
                amounts = first * 1.6 ** np.arange(n_pay) * rng.lognormal(0, 0.2, n_pay)
                hour_profile = "evening"
                same_mule_p = 0.5
            elif stype == "purchase":       # fake marketplace seller
                n_pay = 1 + rng.binomial(1, 0.2)
                gaps_h = rng.exponential(6, n_pay)
                amounts = rng.lognormal(np.log(220), 0.7, n_pay)
                hour_profile = "normal"
                same_mule_p = 0.9
            else:                           # romance
                n_pay = min(2 + rng.poisson(1.5), 6)
                gaps_h = rng.uniform(7, 30, n_pay) * 24
                amounts = c.monthly_income * rng.lognormal(np.log(0.6), 0.6, n_pay)
                hour_profile = "evening"
                same_mule_p = 0.95

            t = t0
            for k in range(n_pay):
                if k > 0:
                    t = t + pd.Timedelta(hours=float(gaps_h[k]))
                    if rng.random() > same_mule_p:
                        mule_counter += 1
                        mule = f"M{mule_counter:05d}"
                if hour_profile != "normal" and k > 0 and gaps_h[k] > 24:
                    t = t.normalize() + pd.Timedelta(hours=int(_sample_hours(rng, 1, hour_profile)[0]))
                if t >= end:
                    break
                # victims become habituated: later payments in long scams are less coached
                ck = float(np.clip(coaching * rng.uniform(0.75, 1.1) * (0.85**k), 0, 1))
                rows.append((c.customer_id, t, max(5.0, round(float(amounts[k]), 2)), mule, stype, f"E{episode_counter:05d}", ck))
    return pd.DataFrame(rows, columns=["customer_id", "timestamp", "amount", "beneficiary_id", "scam_type", "episode_id", "lat_coaching"])


def _assign_beneficiaries(rng, txns: pd.DataFrame, customers: pd.DataFrame) -> pd.Series:
    """Legit beneficiaries chosen sequentially from each customer's own payee history."""
    p_new = customers.set_index("customer_id")["lat_p_new_payee"]
    out = pd.Series(index=txns.index, dtype=object)
    legit = txns[txns["is_scam"] == 0]
    for cust, grp in legit.groupby("customer_id", sort=False):
        payees: list[str] = []
        counts: list[int] = []
        pn = p_new[cust]
        for idx, big in zip(grp.index, grp["lat_big_purchase"]):
            new_p = 0.7 if big else pn
            if not payees or rng.random() < new_p:
                payees.append(f"{cust}-B{len(payees):03d}")
                counts.append(1)
                out[idx] = payees[-1]
            else:
                w = np.asarray(counts, dtype=float)
                j = rng.choice(len(payees), p=w / w.sum())
                counts[j] += 1
                out[idx] = payees[j]
    return out


def _journey(rng, *, coaching: float, signal: float, new_payee: bool, big: bool, cust) -> list[tuple[str, float]]:
    """Simulate one session's event sequence. `k` = effective scam coaching visible in behavior."""
    k = signal * coaching
    T = cust.lat_tempo
    ev: list[tuple[str, float]] = [(J.LOGIN, 0.0)]
    t = 0.0

    def step(mean_s: float, sigma: float = 0.45) -> float:
        return float(rng.lognormal(np.log(mean_s * T), sigma))

    if rng.random() < min(0.95, cust.lat_call_habit + 0.6 * k):
        ev.append((J.CALL_ACTIVE, 0.5))
    t += step(2.0)
    ev.append((J.HOME, t))
    for _ in range(rng.poisson(cust.lat_browse_habit * max(0.05, 1 - 0.85 * k))):
        t += step(6.0, 0.6)
        ev.append((J.BROWSE, t))
    t += step(3.0)
    ev.append((J.PAYMENTS_MENU, t))

    if new_payee:
        t += step(2.0)
        ev.append((J.PAYEE_NEW_FORM, t))
        if rng.random() < min(0.97, 0.6 * cust.lat_paste_habit + 0.45 * k):
            if rng.random() < 0.6:  # copied from another app (messenger, e-mail, marketplace)
                t += step(1.5)
                ev.append((J.APP_BACKGROUND, t))
                t += float(rng.lognormal(np.log(12 + 30 * k), 0.7))
                ev.append((J.APP_FOREGROUND, t))
            t += step(5.0)
            ev.append((J.PAYEE_PASTED, t))
        else:
            t += step(25.0, 0.4)
            ev.append((J.PAYEE_TYPED, t))
    else:
        t += step(3.0)
        ev.append((J.PAYEE_SELECT_SAVED, t))

    t += step(6.0, 0.5)
    ev.append((J.AMOUNT_ENTERED, t))

    # Hesitation / coaching blocks, interleaved in random order
    blocks: list[str] = []
    blocks += ["amount_edit"] * rng.poisson(0.15 + 0.25 * big + 1.3 * k)
    blocks += ["payee_edit"] * rng.poisson((0.08 if new_payee else 0.02) + 0.5 * k)
    blocks += ["switch"] * rng.poisson(cust.lat_switch_habit + 2.2 * k)
    blocks += ["back"] * rng.poisson(0.15 + 1.0 * k)
    rng.shuffle(blocks)
    for b in blocks:
        if b == "amount_edit":
            t += step(5.0)
            ev.append((J.AMOUNT_EDIT, t))
        elif b == "payee_edit":
            t += step(8.0)
            ev.append((J.PAYEE_EDIT, t))
        elif b == "switch":
            t += step(2.0)
            ev.append((J.APP_BACKGROUND, t))
            t += float(rng.lognormal(np.log(15 + 45 * k), 0.8))
            ev.append((J.APP_FOREGROUND, t))
        else:
            t += step(2.0)
            ev.append((J.BACK_NAVIGATION, t))
            if rng.random() < 0.7:
                t += step(2.5)
                ev.append((J.PAYMENTS_MENU, t))

    t += step(2.0)
    ev.append((J.REVIEW_SCREEN, t))
    confirm_log_mean = np.log(4.0 * T * (1.5 if big else 1.0)) + k * rng.normal(0.9, 0.5)
    t += float(rng.lognormal(confirm_log_mean, 0.5))
    ev.append((J.CONFIRM, t))
    return ev


def generate_dataset(
    cfg: dict[str, Any],
    seed: int,
    signal_strength: float | None = None,
    prevalence_multiplier: float = 1.0,
) -> SyntheticData:
    """Generate customers, transfers (with labels) and journey events. Fully seeded."""
    dcfg = cfg["data"]
    rng = np.random.default_rng(seed)
    signal = float(dcfg["journey_signal_strength"] if signal_strength is None else signal_strength)
    warmup = pd.Timestamp(dcfg["warmup_start"])
    end = pd.Timestamp(dcfg["end"])

    customers = _generate_customers(rng, int(dcfg["n_customers"]), warmup)
    legit = _legit_transactions(rng, customers, warmup, end, float(dcfg["monthly_txn_rate_mean"]))
    legit["is_scam"] = 0
    legit["scam_type"] = None
    legit["episode_id"] = None
    legit["lat_coaching"] = 0.0
    scams = _scam_transactions(rng, customers, warmup, end, float(dcfg["scam_episode_rate"]), prevalence_multiplier)
    scams["is_scam"] = 1
    scams["lat_big_purchase"] = False

    txns = pd.concat([legit, scams], ignore_index=True)
    txns = txns.sort_values(["timestamp", "customer_id"], kind="stable").reset_index(drop=True)
    txns["txn_id"] = [f"T{i:06d}" for i in range(len(txns))]
    legit_benef = _assign_beneficiaries(rng, txns, customers)
    txns["beneficiary_id"] = txns["beneficiary_id"].where(txns["is_scam"] == 1, legit_benef)

    # A payee is "new" for journey simulation if the customer never paid it before.
    txns["lat_new_payee"] = ~txns.duplicated(["customer_id", "beneficiary_id"])

    cust_by_id = {c.customer_id: c for c in customers.itertuples(index=False)}
    event_rows = []
    for row in txns.itertuples(index=False):
        cust = cust_by_id[row.customer_id]
        seq = _journey(rng, coaching=row.lat_coaching, signal=signal, new_payee=bool(row.lat_new_payee), big=bool(row.lat_big_purchase), cust=cust)
        event_rows.extend((row.txn_id, i, e, round(ts, 2)) for i, (e, ts) in enumerate(seq))
    events = pd.DataFrame(event_rows, columns=["txn_id", "seq", "event_type", "t_offset_s"])

    public_txn_cols = ["txn_id", "customer_id", "timestamp", "amount", "beneficiary_id", "is_scam", "scam_type", "episode_id"]
    public_cust_cols = ["customer_id", "age", "digital_literacy", "monthly_income", "signup_date"]
    return SyntheticData(customers[public_cust_cols].copy(), txns[public_txn_cols].copy(), events)


def save_raw(data: SyntheticData, out_dir) -> None:
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data.customers.to_parquet(out / "customers.parquet", index=False)
    data.transactions.to_parquet(out / "transactions.parquet", index=False)
    data.events.to_parquet(out / "journey_events.parquet", index=False)
