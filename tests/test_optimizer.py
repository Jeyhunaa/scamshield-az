"""Cost-aware decision engine: mathematical correctness, guardrails, edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from scamshield.decision.optimizer import (
    ACTIONS,
    ALLOW,
    HOLD,
    VERIFY,
    WARN,
    ActionParams,
    Guardrails,
    InterventionConfig,
    cost_components,
    decide,
    decide_batch,
    fixed_threshold_actions,
    load_intervention_config,
    realized_costs,
    tune_fixed_thresholds,
)


@pytest.fixture
def cfg() -> InterventionConfig:
    return InterventionConfig(
        actions={
            "allow": ActionParams(0.0, 0.0, 0.0),
            "warn": ActionParams(0.2, 0.1, 1.0),
            "verify": ActionParams(0.5, 2.0, 5.0),
            "hold": ActionParams(0.9, 20.0, 50.0),
        },
        guardrails=Guardrails(min_amount_verify=50, min_amount_hold=200, high_risk_floor=0.5, max_amount_without_warning=1e9),
        loss_given_scam=1.0,
    )


def test_expected_cost_matches_formula(cfg):
    p, amount = 0.1, 1000.0
    total = cost_components([p], [amount], cfg)["total"][0]
    for i, a in enumerate(ACTIONS):
        x = cfg.actions[a]
        expected = p * amount * (1 - x.effectiveness) + x.operational_cost + (1 - p) * x.friction_cost
        assert total[i] == pytest.approx(expected)
    # hand-check: verify = 0.1*1000*0.5 + 2 + 0.9*5 = 56.5
    assert total[VERIFY] == pytest.approx(56.5)


def test_loss_given_scam_scales_loss(cfg):
    from dataclasses import replace

    half = replace(cfg, loss_given_scam=0.5)
    assert cost_components([0.2], [1000], half)["residual_loss"][0, ALLOW] == pytest.approx(100.0)


def test_zero_risk_allows_and_certain_large_scam_holds(cfg):
    assert decide_batch([0.0], [5000], cfg)[0] == ALLOW
    assert decide_batch([1.0], [5000], cfg)[0] == HOLD


def test_decision_is_argmin_over_permitted_actions(cfg):
    rng = np.random.default_rng(0)
    p, amt = rng.uniform(0, 1, 500), rng.lognormal(5, 1.5, 500)
    acts = decide_batch(p, amt, cfg)
    total = cost_components(p, amt, cfg)["total"]
    from scamshield.decision.optimizer import eligibility

    elig = eligibility(p, amt, cfg)
    for i in range(len(p)):
        assert elig[i, acts[i]]
        assert total[i, acts[i]] <= total[i][elig[i]].min() + 1e-9


def test_action_escalates_monotonically_with_risk(cfg):
    p = np.linspace(0, 1, 401)
    acts = decide_batch(p, np.full_like(p, 2000.0), cfg)
    assert np.all(np.diff(acts) >= 0)


def test_hold_guardrails(cfg):
    assert decide_batch([0.99], [150.0], cfg)[0] != HOLD          # below hold minimum
    assert decide_batch([0.99], [5000.0], cfg, hold_allowed=False)[0] == VERIFY
    assert decide_batch([0.99], [30.0], cfg)[0] == WARN           # below verify minimum too


def test_high_risk_floor_never_allows(cfg):
    # tiny amount: allow would be cheapest, but the floor forbids it
    assert decide_batch([0.6], [1.0], cfg)[0] == WARN
    assert decide_batch([0.4], [1.0], cfg)[0] == ALLOW


def test_ties_resolve_to_least_intrusive():
    same = ActionParams(0.0, 0.0, 0.0)
    cfg = InterventionConfig(actions={a: same for a in ACTIONS}, guardrails=Guardrails())
    assert decide_batch([0.3], [1000.0], cfg)[0] == ALLOW


@pytest.mark.parametrize("p,amount", [([1.2], [10]), ([-0.1], [10]), ([np.nan], [10]), ([0.5], [-1]), ([0.5, 0.2], [10])])
def test_invalid_inputs_raise(cfg, p, amount):
    with pytest.raises(ValueError):
        decide_batch(p, amount, cfg)


def test_invalid_config_raises():
    good = {"effectiveness": 0.5, "operational_cost": 1, "friction_cost": 1}
    with pytest.raises(ValueError):
        InterventionConfig.from_dict({"actions": {a: good for a in ACTIONS} | {"warn": {**good, "effectiveness": 1.5}}})
    with pytest.raises(ValueError):
        InterventionConfig.from_dict({"actions": {a: good for a in ("allow", "warn", "verify")}})
    with pytest.raises(ValueError):
        InterventionConfig.from_dict({"actions": {a: good for a in ACTIONS} | {"hold": {**good, "friction_cost": -1}}})


def test_yaml_config_loads():
    cfg = load_intervention_config()
    e, c, f = cfg.vectors()
    assert e[ALLOW] == 0 and c[ALLOW] == 0 and f[ALLOW] == 0
    assert np.all(np.diff(e) > 0)  # stronger interventions are more effective


def test_scale_effectiveness_caps(cfg):
    scaled = cfg.scale_effectiveness(2.0)
    assert scaled.actions["hold"].effectiveness == 0.99
    assert scaled.actions["warn"].effectiveness == pytest.approx(0.4)


def test_fixed_thresholds_and_downgrades(cfg):
    th = (0.1, 0.3, 0.6)
    p = np.array([0.05, 0.15, 0.35, 0.65, 0.65, 0.65])
    amt = np.array([1000, 1000, 1000, 1000, 100, 20])
    acts = fixed_threshold_actions(p, amt, th, cfg)
    # last two: hold desired but amount < hold min -> verify; amount < verify min -> warn
    assert list(acts) == [ALLOW, WARN, VERIFY, HOLD, VERIFY, WARN]
    with pytest.raises(ValueError):
        fixed_threshold_actions(p, amt, (0.5, 0.2, 0.9), cfg)


def test_fixed_thresholds_respect_risk_floor(cfg):
    assert fixed_threshold_actions([0.7], [1000], (0.9, 0.95, 0.99), cfg)[0] == WARN


def test_realized_costs_accounting(cfg):
    acts = np.array([ALLOW, HOLD, VERIFY, HOLD])
    y = np.array([1, 1, 0, 0])
    amt = np.array([100.0, 1000.0, 500.0, 500.0])
    r = realized_costs(acts, y, amt, cfg)
    np.testing.assert_allclose(r["prevented_loss"] + r["residual_loss"], y * amt)
    np.testing.assert_allclose(r["prevented_loss"], [0, 900, 0, 0])
    np.testing.assert_allclose(r["friction"], [0, 0, 5, 50])
    np.testing.assert_allclose(r["total"], r["residual_loss"] + r["operational"] + r["friction"])


def test_tuned_thresholds_are_optimal_on_grid(cfg):
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.05, 2000)
    p = np.clip(0.03 + 0.4 * y + rng.normal(0, 0.1, 2000), 0, 1)
    amt = rng.lognormal(6, 1, 2000)
    grid = [0.05, 0.1, 0.2, 0.3, 0.5]
    best = tune_fixed_thresholds(p, amt, y, cfg, grid)
    best_cost = realized_costs(fixed_threshold_actions(p, amt, best, cfg), y, amt, cfg)["total"].sum()
    for th in [(0.05, 0.1, 0.2), (0.1, 0.3, 0.5), (0.2, 0.2, 0.5)]:
        assert best_cost <= realized_costs(fixed_threshold_actions(p, amt, th, cfg), y, amt, cfg)["total"].sum() + 1e-9


def test_decide_explanation_and_breakdown(cfg):
    d = decide(0.3, 3000.0, cfg, hold_allowed=False)
    assert d.action in ACTIONS
    assert not d.expected_costs["hold"]["eligible"]
    assert "not permitted" in d.expected_costs["hold"]["ineligible_reason"]
    costs = {a: v["expected_cost"] for a, v in d.expected_costs.items() if v["eligible"]}
    assert d.action == min(costs, key=costs.get)
    assert d.action in d.explanation
