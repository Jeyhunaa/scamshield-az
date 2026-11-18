"""End-to-end experiment: generate -> features -> train -> calibrate -> decide -> evaluate.

The two end-to-end approaches compared on the same held-out test period:

* **Baseline**   transaction-based fraud score + fixed thresholds (thresholds tuned on the
                 validation period to minimise cost — the baseline's best case).
* **ScamShield** existing score + payment-journey features (CatBoost) + cost-aware optimizer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scamshield.config import DATA_DIR
from scamshield.data.generator import generate_dataset, save_raw
from scamshield.data.preprocessing import build_modeling_table
from scamshield.decision.optimizer import InterventionConfig, decide_batch, fixed_threshold_actions, tune_fixed_thresholds
from scamshield.evaluation.metrics import calibration_table, classification_metrics, paired_bootstrap, policy_summary
from scamshield.models.train import LABEL, VARIANTS, FittedModel, save_models, train_all

log = logging.getLogger("scamshield")


@dataclass
class Experiment:
    models: dict[str, FittedModel]
    scored: pd.DataFrame                    # full table with baseline_logit
    thresholds: tuple[float, float, float]  # tuned fixed thresholds for the baseline policy

    def split(self, name: str) -> pd.DataFrame:
        return self.scored[self.scored["split"] == name]


def run_experiment(
    cfg: dict[str, Any],
    icfg: InterventionConfig,
    grid: list[float],
    signal_strength: float | None = None,
    variants: list[str] | None = None,
    data_dir: Path | None = None,
) -> Experiment:
    seed = int(cfg["seed"])
    data = generate_dataset(cfg, seed, signal_strength=signal_strength)
    table = build_modeling_table(data, cfg)
    if data_dir is not None:
        save_raw(data, data_dir / "raw")
        (data_dir / "processed").mkdir(parents=True, exist_ok=True)
        table.to_parquet(data_dir / "processed" / "modeling_table.parquet", index=False)
    models, scored = train_all(table, cfg, variants)
    val = scored[scored["split"] == "val"]
    thresholds = tune_fixed_thresholds(models["baseline_lr"].predict(val), val["amount"].to_numpy(), val[LABEL].to_numpy(), icfg, grid)
    return Experiment(models, scored, thresholds)


def compare_policies(test: pd.DataFrame, p_base: np.ndarray, p_enh: np.ndarray, thresholds, icfg: InterventionConfig, weights=None) -> dict[str, Any]:
    amount, y = test["amount"].to_numpy(), test[LABEL].to_numpy()
    base_actions = fixed_threshold_actions(p_base, amount, thresholds, icfg)
    enh_actions = decide_batch(p_enh, amount, icfg)
    return {
        "baseline": policy_summary(base_actions, y, amount, icfg, weights),
        "scamshield": policy_summary(enh_actions, y, amount, icfg, weights),
        "_actions": (base_actions, enh_actions),
    }


def _savings(cmp: dict[str, Any]) -> dict[str, float]:
    b, s = cmp["baseline"]["total_cost"], cmp["scamshield"]["total_cost"]
    return {"baseline_total_cost": b, "scamshield_total_cost": s, "cost_reduction": b - s, "cost_reduction_pct": (b - s) / b if b else 0.0}


def _thresholds_json(t) -> dict[str, float | None]:
    return {k: (None if np.isinf(v) else float(v)) for k, v in zip(("warn", "verify", "hold"), t)}


def evaluate(exp: Experiment, cfg: dict[str, Any], icfg: InterventionConfig) -> tuple[dict[str, Any], pd.DataFrame]:
    """Final evaluation on the untouched test period."""
    ecfg = cfg["evaluation"]
    test = exp.split("test")
    y, amount = test[LABEL].to_numpy(), test["amount"].to_numpy()
    preds = {name: m.predict(test) for name, m in exp.models.items()}
    p_base, p_enh = preds["baseline_lr"], preds["scamshield"]

    metrics: dict[str, Any] = {
        "data": {
            split: {"n": int((exp.scored["split"] == split).sum()), "n_scam": int(exp.scored.loc[exp.scored["split"] == split, LABEL].sum())}
            for split in ("train", "val", "test")
        },
        "models": {name: classification_metrics(y, p, ecfg["fpr_levels"]) for name, p in preds.items()},
        "ablation_features": {name: VARIANTS[name][1] for name in preds},
        "calibration": {"baseline_lr": calibration_table(y, p_base), "scamshield": calibration_table(y, p_enh)},
        "baseline_thresholds": _thresholds_json(exp.thresholds),
    }
    cmp = compare_policies(test, p_base, p_enh, exp.thresholds, icfg)
    base_actions, enh_actions = cmp.pop("_actions")
    metrics["end_to_end"] = {**cmp, "savings": _savings(cmp)}

    # Supplementary attribution: which part of the gain comes from the model vs. the optimizer?
    attribution = {
        "baseline_score+fixed_thresholds": cmp["baseline"]["total_cost"],
        "baseline_score+cost_optimizer": policy_summary(decide_batch(p_base, amount, icfg), y, amount, icfg)["total_cost"],
        "scamshield_score+cost_optimizer": cmp["scamshield"]["total_cost"],
    }
    val = exp.split("val")
    enh_thr = tune_fixed_thresholds(exp.models["scamshield"].predict(val), val["amount"].to_numpy(), val[LABEL].to_numpy(), icfg, _grid(cfg))
    attribution["scamshield_score+fixed_thresholds"] = policy_summary(fixed_threshold_actions(p_enh, amount, enh_thr, icfg), y, amount, icfg)["total_cost"]
    metrics["attribution_total_cost"] = attribution

    # Paired bootstrap CIs (stratified by label) on the test period
    reps, seed = int(ecfg["bootstrap_reps"]), int(cfg["seed"])
    from sklearn.metrics import average_precision_score
    from scamshield.decision.optimizer import realized_costs

    cost_b = realized_costs(base_actions, y, amount, icfg)["total"]
    cost_s = realized_costs(enh_actions, y, amount, icfg)["total"]
    metrics["bootstrap_95ci"] = {
        "pr_auc_gain": paired_bootstrap(lambda i: average_precision_score(y[i], p_enh[i]) - average_precision_score(y[i], p_base[i]), len(y), reps, seed, y),
        "total_cost_reduction": paired_bootstrap(lambda i: cost_b[i].sum() - cost_s[i].sum(), len(y), reps, seed, y),
    }

    predictions = test[["txn_id", "timestamp", "customer_id", "amount", LABEL, "scam_type"]].copy()
    predictions["baseline_risk"] = p_base
    predictions["enhanced_risk"] = p_enh
    predictions["baseline_action"] = base_actions
    predictions["scamshield_action"] = enh_actions
    return metrics, predictions


def _grid(cfg) -> list[float]:
    return cfg["_threshold_grid"]


def sensitivity(exp: Experiment, cfg: dict[str, Any], icfg: InterventionConfig) -> dict[str, Any]:
    scfg = cfg["evaluation"]["sensitivity"]
    grid = _grid(cfg)
    out: dict[str, Any] = {}

    # 1) Weaker behavioral signals: regenerate data, retrain both models, re-tune baseline.
    rows = []
    for s in scfg["signal_strengths"]:
        e = exp if s == cfg["data"]["journey_signal_strength"] else run_experiment(cfg, icfg, grid, signal_strength=s, variants=["baseline_lr", "scamshield"])
        t = e.split("test")
        y = t[LABEL].to_numpy()
        pb, pe = e.models["baseline_lr"].predict(t), e.models["scamshield"].predict(t)
        cmp = compare_policies(t, pb, pe, e.thresholds, icfg)
        rows.append(
            {
                "signal_strength": s,
                "baseline_pr_auc": classification_metrics(y, pb, [])["pr_auc"],
                "scamshield_pr_auc": classification_metrics(y, pe, [])["pr_auc"],
                **_savings(cmp),
            }
        )
        log.info("signal %.2f -> %s", s, rows[-1])
    out["journey_signal_strength"] = rows

    # 2) Scam prevalence drift after deployment: reweight test scams (models NOT recalibrated).
    test = exp.split("test")
    y = test[LABEL].to_numpy()
    pb, pe = exp.models["baseline_lr"].predict(test), exp.models["scamshield"].predict(test)
    rows = []
    for m in scfg["prevalence_multipliers"]:
        w = np.where(y == 1, float(m), 1.0)
        cmp = compare_policies(test, pb, pe, exp.thresholds, icfg, weights=w)
        rows.append(
            {
                "prevalence_multiplier": m,
                "effective_prevalence": float((w * y).sum() / w.sum()),
                "baseline_pr_auc": classification_metrics(y, pb, [], w)["pr_auc"],
                "scamshield_pr_auc": classification_metrics(y, pe, [], w)["pr_auc"],
                **_savings(cmp),
            }
        )
    out["scam_prevalence"] = rows

    # 3) Intervention effectiveness assumptions: re-tune baseline thresholds under each assumption.
    val = exp.split("val")
    pb_val = exp.models["baseline_lr"].predict(val)
    rows = []
    for m in scfg["effectiveness_multipliers"]:
        c = icfg.scale_effectiveness(m)
        thr = tune_fixed_thresholds(pb_val, val["amount"].to_numpy(), val[LABEL].to_numpy(), c, grid)
        cmp = compare_policies(test, pb, pe, thr, c)
        rows.append({"effectiveness_multiplier": m, "baseline_thresholds": _thresholds_json(thr), **_savings(cmp)})
    out["intervention_effectiveness"] = rows
    return out


def run_full_pipeline(
    cfg: dict[str, Any],
    icfg_dict: dict[str, Any],
    artifacts_dir: Path,
    with_sensitivity: bool = True,
    data_dir: Path | None = DATA_DIR,
) -> dict[str, Any]:
    from scamshield.evaluation.plots import make_all_figures

    icfg = InterventionConfig.from_dict(icfg_dict)
    cfg = {**cfg, "_threshold_grid": list(icfg_dict["baseline_threshold_grid"])}
    log.info("Generating data, building features, training %d model variants ...", len(VARIANTS))
    exp = run_experiment(cfg, icfg, _grid(cfg), data_dir=data_dir)

    models_dir = artifacts_dir / "models"
    meta = {
        "seed": cfg["seed"],
        "split": cfg["split"],
        "data": {k: v for k, v in cfg["data"].items()},
        "note": "Trained on synthetic data only.",
    }
    save_models(exp.models, models_dir, meta)
    (models_dir / "baseline_policy.json").write_text(json.dumps({"thresholds": _thresholds_json(exp.thresholds), "tuned_on": "validation period, minimising total cost"}, indent=2))

    log.info("Evaluating on held-out test period ...")
    metrics, predictions = evaluate(exp, cfg, icfg)
    if with_sensitivity:
        log.info("Running sensitivity analyses (regenerates data for weaker journey signals) ...")
        metrics["sensitivity"] = sensitivity(exp, cfg, icfg)
    metrics["feature_importance"] = dict(
        zip(exp.models["scamshield"].features, map(float, exp.models["scamshield"].model.get_feature_importance()))
    )
    metrics["assumptions"] = {"intervention": icfg_dict, "disclaimer": "All results are on synthetic data under assumed costs; not evidence of real-world banking performance."}

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(artifacts_dir / "predictions_test.parquet", index=False)
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    make_all_figures(metrics, predictions, icfg, artifacts_dir / "figures")
    return metrics
