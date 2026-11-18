"""Evaluation metrics: discrimination, calibration, equal-FPR comparison, policy outcomes, bootstrap."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from scamshield.decision.optimizer import ACTIONS, HOLD, InterventionConfig, realized_costs


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE with equal-count (quantile) bins — robust for rare-event probabilities."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    order = np.argsort(p)
    ece = 0.0
    for chunk in np.array_split(order, n_bins):
        if len(chunk):
            ece += len(chunk) / len(p) * abs(p[chunk].mean() - y[chunk].mean())
    return float(ece)


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    order = np.argsort(p)
    return [
        {"mean_predicted": float(p[c].mean()), "observed_rate": float(y[c].mean()), "n": int(len(c))}
        for c in np.array_split(order, n_bins)
        if len(c)
    ]


def at_fpr(y: np.ndarray, p: np.ndarray, fpr: float) -> dict[str, float]:
    """Recall/precision at the score threshold that flags `fpr` of legitimate transfers."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    legit = p[y == 0]
    threshold = float(np.quantile(legit, 1 - fpr, method="higher"))
    flagged = p > threshold
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    return {
        "target_fpr": fpr,
        "threshold": threshold,
        "recall": tp / max(1, int((y == 1).sum())),
        "precision": tp / max(1, tp + fp),
        "actual_fpr": fp / max(1, len(legit)),
    }


def classification_metrics(y: np.ndarray, p: np.ndarray, fpr_levels: list[float], weights: np.ndarray | None = None) -> dict[str, Any]:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    out: dict[str, Any] = {
        "pr_auc": float(average_precision_score(y, p, sample_weight=weights)),
        "roc_auc": float(roc_auc_score(y, p, sample_weight=weights)),
        "brier": float(brier_score_loss(y, p, sample_weight=weights)),
    }
    if weights is None:
        out["log_loss"] = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
        out["ece"] = expected_calibration_error(y, p)
        out["mean_predicted"] = float(p.mean())
        out["prevalence"] = float(y.mean())
        out["at_fpr"] = [at_fpr(y, p, f) for f in fpr_levels]
    return out


def policy_summary(actions: np.ndarray, y: np.ndarray, amount: np.ndarray, cfg: InterventionConfig, weights: np.ndarray | None = None) -> dict[str, Any]:
    """Operational and financial outcomes of a decision policy on labeled transactions."""
    actions, y, amount = np.asarray(actions), np.asarray(y), np.asarray(amount, dtype=float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    costs = realized_costs(actions, y, amount, cfg)
    intervened = actions > 0
    scam, legit = y == 1, y == 0
    n_scam, n_legit = w[scam].sum(), w[legit].sum()
    total_scam_value = float((w * y * amount * cfg.loss_given_scam).sum())
    out: dict[str, Any] = {
        "n_transactions": float(w.sum()),
        "action_counts": {a: float(w[actions == i].sum()) for i, a in enumerate(ACTIONS)},
        "intervention_rate": float(w[intervened].sum() / w.sum()),
        "scam_recall_any_intervention": float(w[intervened & scam].sum() / max(n_scam, 1e-12)),
        "legit_intervention_rate": float(w[intervened & legit].sum() / max(n_legit, 1e-12)),  # FPR of the policy
        "holds_total": float(w[actions == HOLD].sum()),
        "unnecessary_holds": float(w[(actions == HOLD) & legit].sum()),
        "scam_value_at_risk": total_scam_value,
        "prevented_loss": float((w * costs["prevented_loss"]).sum()),
        "residual_loss": float((w * costs["residual_loss"]).sum()),
        "operational_cost": float((w * costs["operational"]).sum()),
        "friction_cost": float((w * costs["friction"]).sum()),
        "total_cost": float((w * costs["total"]).sum()),
    }
    out["prevented_loss_share"] = out["prevented_loss"] / total_scam_value if total_scam_value > 0 else 0.0
    return out


def paired_bootstrap(
    stat: Callable[[np.ndarray], float], n: int, reps: int, seed: int, stratify: np.ndarray | None = None
) -> dict[str, float]:
    """Percentile CI of a statistic computed on resampled row indices (paired across models)."""
    rng = np.random.default_rng(seed)
    vals = []
    groups = [np.flatnonzero(stratify == g) for g in np.unique(stratify)] if stratify is not None else [np.arange(n)]
    for _ in range(reps):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        vals.append(stat(idx))
    vals = np.asarray(vals)
    return {"mean": float(vals.mean()), "ci_low": float(np.percentile(vals, 2.5)), "ci_high": float(np.percentile(vals, 97.5))}
