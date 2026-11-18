"""Cost-aware intervention optimizer.

For every permitted action a, the expected cost is

    EC(a) = p * L * (1 - e_a)  +  C_a  +  (1 - p) * F_a
            \\_ residual loss _/  \\op/   \\_ friction on legit _/

and the engine picks the permitted action with the lowest EC. Ties resolve to the LEAST
intrusive action (actions are ordered allow < warn < verify < hold).

All parameters (effectiveness, costs, guardrails) come from configs/intervention.yaml and
are simulation assumptions, not validated banking figures.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any

import numpy as np

ACTIONS: tuple[str, ...] = ("allow", "warn", "verify", "hold")
ALLOW, WARN, VERIFY, HOLD = range(4)


@dataclass(frozen=True)
class ActionParams:
    effectiveness: float
    operational_cost: float
    friction_cost: float


@dataclass(frozen=True)
class Guardrails:
    min_amount_verify: float = 50.0
    min_amount_hold: float = 200.0
    high_risk_floor: float = 0.5
    max_amount_without_warning: float = float("inf")


@dataclass(frozen=True)
class InterventionConfig:
    actions: dict[str, ActionParams]
    guardrails: Guardrails
    loss_given_scam: float = 1.0

    def __post_init__(self) -> None:
        if set(self.actions) != set(ACTIONS):
            raise ValueError(f"config must define exactly the actions {ACTIONS}")
        for name, a in self.actions.items():
            if not 0.0 <= a.effectiveness <= 1.0:
                raise ValueError(f"{name}: effectiveness must be in [0, 1]")
            if a.operational_cost < 0 or a.friction_cost < 0:
                raise ValueError(f"{name}: costs must be non-negative")
        if not 0.0 <= self.loss_given_scam <= 1.0:
            raise ValueError("loss_given_scam must be in [0, 1]")
        if not 0.0 < self.guardrails.high_risk_floor <= 1.0:
            raise ValueError("high_risk_floor must be in (0, 1]")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InterventionConfig:
        actions = {name: ActionParams(**{k: float(v) for k, v in params.items()}) for name, params in d["actions"].items()}
        guard = Guardrails(**{k: float(v) for k, v in d.get("guardrails", {}).items()})
        return cls(actions=actions, guardrails=guard, loss_given_scam=float(d.get("loss_given_scam", 1.0)))

    def vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(effectiveness, operational, friction) arrays in ACTIONS order."""
        e = np.array([self.actions[a].effectiveness for a in ACTIONS])
        c = np.array([self.actions[a].operational_cost for a in ACTIONS])
        f = np.array([self.actions[a].friction_cost for a in ACTIONS])
        return e, c, f

    def scale_effectiveness(self, multiplier: float) -> InterventionConfig:
        """Copy with every effectiveness multiplied (capped at 0.99) — for sensitivity analysis."""
        acts = {n: replace(a, effectiveness=float(min(0.99, a.effectiveness * multiplier))) for n, a in self.actions.items()}
        return replace(self, actions=acts)


def load_intervention_config(path=None) -> InterventionConfig:
    from scamshield.config import load_intervention_yaml

    return InterventionConfig.from_dict(load_intervention_yaml(path))


def _validate_inputs(p, amount) -> tuple[np.ndarray, np.ndarray]:
    p = np.atleast_1d(np.asarray(p, dtype=float))
    amount = np.atleast_1d(np.asarray(amount, dtype=float))
    if p.shape != amount.shape:
        raise ValueError("p and amount must have the same shape")
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if np.any(~np.isfinite(amount)) or np.any(amount < 0):
        raise ValueError("amounts must be finite and non-negative")
    return p, amount


def cost_components(p, amount, cfg: InterventionConfig) -> dict[str, np.ndarray]:
    """Expected-cost components, each of shape (n, 4) in ACTIONS order."""
    p, amount = _validate_inputs(p, amount)
    e, c, f = cfg.vectors()
    loss = (amount * cfg.loss_given_scam)[:, None]
    residual = p[:, None] * loss * (1 - e)[None, :]
    operational = np.broadcast_to(c, residual.shape).copy()
    friction = (1 - p)[:, None] * f[None, :]
    return {"residual_loss": residual, "operational": operational, "friction": friction, "total": residual + operational + friction}


def eligibility(p, amount, cfg: InterventionConfig, hold_allowed=True) -> np.ndarray:
    """Boolean (n, 4) mask of permitted actions under policy guardrails."""
    p, amount = _validate_inputs(p, amount)
    g = cfg.guardrails
    hold_ok = np.broadcast_to(np.asarray(hold_allowed, dtype=bool), p.shape)
    mask = np.ones((len(p), 4), dtype=bool)
    mask[:, ALLOW] = (p < g.high_risk_floor) & (amount <= g.max_amount_without_warning)
    mask[:, VERIFY] = amount >= g.min_amount_verify
    mask[:, HOLD] = hold_ok & (amount >= g.min_amount_hold)
    return mask  # WARN is always permitted, so every row has at least one option


def decide_batch(p, amount, cfg: InterventionConfig, hold_allowed=True) -> np.ndarray:
    """Index (0..3) of the minimum-expected-cost permitted action for each transaction."""
    total = cost_components(p, amount, cfg)["total"]
    masked = np.where(eligibility(p, amount, cfg, hold_allowed), total, np.inf)
    return np.argmin(masked, axis=1)  # argmin returns the first minimum -> least intrusive on ties


def fixed_threshold_actions(p, amount, thresholds: tuple[float, float, float], cfg: InterventionConfig, hold_allowed=True) -> np.ndarray:
    """Baseline policy: escalate by fixed probability thresholds (warn, verify, hold).

    The same guardrails apply: an ineligible action is downgraded to the strongest
    permitted action below it (and "allow" is upgraded to "warn" when not permitted).
    """
    t_warn, t_verify, t_hold = thresholds
    if not t_warn <= t_verify <= t_hold:
        raise ValueError("thresholds must satisfy warn <= verify <= hold")
    p, amount = _validate_inputs(p, amount)
    desired = (p >= t_warn).astype(int) + (p >= t_verify) + (p >= t_hold)
    elig = eligibility(p, amount, cfg, hold_allowed)
    levels = np.arange(4)[None, :]
    allowed_below = elig & (levels <= desired[:, None])
    chosen = np.where(allowed_below.any(axis=1), 3 - np.argmax(allowed_below[:, ::-1], axis=1), WARN)
    return chosen


def realized_costs(actions: np.ndarray, y: np.ndarray, amount: np.ndarray, cfg: InterventionConfig) -> dict[str, np.ndarray]:
    """Outcome-based costs on labeled data (expected over intervention success).

    For a scam: residual loss L*(1-e_a) and prevented loss L*e_a; for a legit transfer:
    friction F_a. Operational cost C_a is always paid. Using the expectation over
    intervention success (rather than random draws) keeps evaluation deterministic.
    """
    actions, y, amount = np.asarray(actions), np.asarray(y, dtype=float), np.asarray(amount, dtype=float)
    e, c, f = cfg.vectors()
    loss = amount * cfg.loss_given_scam
    out = {
        "prevented_loss": y * loss * e[actions],
        "residual_loss": y * loss * (1 - e[actions]),
        "operational": c[actions],
        "friction": (1 - y) * f[actions],
    }
    out["total"] = out["residual_loss"] + out["operational"] + out["friction"]
    return out


def tune_fixed_thresholds(p, amount, y, cfg: InterventionConfig, grid: list[float], hold_allowed=True) -> tuple[float, float, float]:
    """Search monotone (warn <= verify <= hold) thresholds minimizing realized total cost.

    Run on the VALIDATION period so the baseline policy gets its best cost-tuned
    thresholds; `inf` means the action is never triggered by score alone.
    """
    cand = sorted(set(float(x) for x in grid)) + [float("inf")]
    best, best_cost = (float("inf"),) * 3, np.inf
    for t in combinations(cand + [float("inf")] * 2, 3):
        if not (t[0] <= t[1] <= t[2]):
            continue
        acts = fixed_threshold_actions(p, amount, t, cfg, hold_allowed)
        cost = realized_costs(acts, y, amount, cfg)["total"].sum()
        if cost < best_cost - 1e-9:
            best, best_cost = t, cost
    return best


@dataclass
class Decision:
    action: str
    expected_costs: dict[str, dict[str, float | bool | str | None]]
    explanation: str


def decide(p: float, amount: float, cfg: InterventionConfig, hold_allowed: bool = True) -> Decision:
    """Single-transaction decision with a transparent per-action cost breakdown."""
    comps = cost_components([p], [amount], cfg)
    elig = eligibility([p], [amount], cfg, hold_allowed)[0]
    idx = int(decide_batch([p], [amount], cfg, hold_allowed)[0])
    g = cfg.guardrails
    reasons = {
        ALLOW: f"blocked by guardrail: risk >= {g.high_risk_floor:.0%} or amount above {g.max_amount_without_warning:,.0f}",
        VERIFY: f"amount below verification minimum ({g.min_amount_verify:,.0f})",
        HOLD: "temporary hold not permitted for this payment" if not hold_allowed else f"amount below hold minimum ({g.min_amount_hold:,.0f})",
    }
    breakdown: dict[str, dict[str, float | bool | str | None]] = {}
    for i, name in enumerate(ACTIONS):
        breakdown[name] = {
            "expected_cost": round(float(comps["total"][0, i]), 2),
            "residual_loss": round(float(comps["residual_loss"][0, i]), 2),
            "operational_cost": round(float(comps["operational"][0, i]), 2),
            "friction_cost": round(float(comps["friction"][0, i]), 2),
            "eligible": bool(elig[i]),
            "ineligible_reason": None if elig[i] else reasons.get(i),
        }
    exposure = p * amount * cfg.loss_given_scam
    eligible_names = [n for n in ACTIONS if breakdown[n]["eligible"] and n != ACTIONS[idx]]
    runner_up = min(eligible_names, key=lambda n: breakdown[n]["expected_cost"]) if eligible_names else None
    text = (
        f"Scam probability {p:.1%} on {amount:,.2f} at risk -> expected exposure {exposure:,.2f}. "
        f"'{ACTIONS[idx]}' has the lowest expected cost ({breakdown[ACTIONS[idx]]['expected_cost']:,.2f})"
    )
    if runner_up:
        text += f"; next best is '{runner_up}' ({breakdown[runner_up]['expected_cost']:,.2f})."
    else:
        text += "."
    return Decision(action=ACTIONS[idx], expected_costs=breakdown, explanation=text)
