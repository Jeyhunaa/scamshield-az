"""Evaluation figures (matplotlib, PNG). Consistent colors: ScamShield = blue, Baseline = orange."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from scamshield.decision.optimizer import ACTIONS, InterventionConfig  # noqa: E402
from scamshield.features.journey import FEATURE_DESCRIPTIONS  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)

LABELS = {"baseline": "Baseline (fixed thresholds)", "scamshield": "ScamShield AZ (cost-aware)"}


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pr_curves(pred: pd.DataFrame, metrics: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.3))
    y = pred["is_scam"]
    for col, color, name, key in [("enhanced_risk", BLUE, "ScamShield", "scamshield"), ("baseline_risk", ORANGE, "Baseline LR", "baseline_lr")]:
        prec, rec, _ = precision_recall_curve(y, pred[col])
        ax.plot(rec, prec, color=color, label=f"{name}  (PR-AUC {metrics['models'][key]['pr_auc']:.3f})")
    ax.axhline(y.mean(), color=MUTED, lw=1, ls=":", label=f"Prevalence ({y.mean():.1%})")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02), title="Precision–recall on held-out test period")
    ax.legend(loc="upper right")
    _save(fig, out / "pr_curves.png")


def calibration(metrics: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    for key, color, name in [("scamshield", BLUE, "ScamShield"), ("baseline_lr", ORANGE, "Baseline LR")]:
        tab = metrics["calibration"][key]
        x = [r["mean_predicted"] for r in tab]
        yy = [r["observed_rate"] for r in tab]
        ax.plot(x, yy, marker="o", ms=5, color=color, label=f"{name} (ECE {metrics['models'][key]['ece']:.4f})")
    lim = max(max(r["mean_predicted"] for r in metrics["calibration"][k]) for k in ("scamshield", "baseline_lr"))
    lim = max(lim, max(r["observed_rate"] for k in ("scamshield", "baseline_lr") for r in metrics["calibration"][k])) * 1.05
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls=":", label="Perfect calibration")
    ax.set(xlabel="Mean predicted probability (decile bins)", ylabel="Observed scam rate", title="Calibration (test period)")
    ax.legend(loc="upper left")
    _save(fig, out / "calibration.png")


def intervention_mix(predictions: pd.DataFrame, out: Path) -> None:
    """Interventions per action and policy; the label shows how many of them hit actual scams."""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    y = predictions["is_scam"].to_numpy()
    width = 0.38
    for j, (col, color, label) in enumerate([("baseline_action", ORANGE, LABELS["baseline"]), ("scamshield_action", BLUE, LABELS["scamshield"])]):
        acts = predictions[col].to_numpy()
        for i, a in enumerate(ACTIONS[1:], start=1):
            n, n_scam = int((acts == i).sum()), int(((acts == i) & (y == 1)).sum())
            pos = (i - 1) + (j - 0.5) * width
            ax.barh(pos, n, height=width - 0.04, color=color, label=label if i == 1 else None)
            ax.text(n, pos, f"  {n:,}  ({n_scam} scams)", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(3), [a.capitalize() for a in ACTIONS[1:]])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    xmax = max(int((predictions[c] == i).sum()) for c in ("baseline_action", "scamshield_action") for i in (1, 2, 3))
    ax.set_xlim(0, xmax * 1.45)
    ax.set_xlabel("Test transactions receiving the intervention")
    ax.set_title("Interventions issued (and how many hit actual scams)")
    ax.legend(loc="lower right")
    _save(fig, out / "intervention_mix.png")


def cost_breakdown(metrics: dict[str, Any], out: Path) -> None:
    e2e = metrics["end_to_end"]
    comps = [("residual_loss", "Residual scam loss", BLUE), ("operational_cost", "Operational cost", ORANGE), ("friction_cost", "Customer friction", AQUA)]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    keys = ["scamshield", "baseline"]
    for row, key in enumerate(keys):
        left = 0.0
        for field, label, color in comps:
            v = e2e[key][field]
            ax.barh(row, v, left=left, color=color, edgecolor="white", linewidth=2, height=0.55, label=label if row == 0 else None)
            left += v
        ax.text(left, row, f"  {left:,.0f} AZN", va="center", fontsize=9, color=INK)
    ax.set_yticks([0, 1], [LABELS[k] for k in keys])
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(e2e[k]["total_cost"] for k in keys) * 1.2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
    ax.set_xlabel("Total cost on test period (AZN)")
    ax.set_title("Estimated fraud-related and operational costs")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    _save(fig, out / "cost_breakdown.png")


def ablation(metrics: dict[str, Any], out: Path) -> None:
    names = {
        "baseline_lr": "Baseline LR (transaction features)",
        "catboost_txn_only": "CatBoost, transaction features",
        "catboost_journey_only": "CatBoost, journey features only",
        "scamshield": "ScamShield: baseline score + journey",
        "catboost_all_features": "CatBoost, all raw features",
    }
    items = [(names[k], metrics["models"][k]["pr_auc"], k) for k in names if k in metrics["models"]]
    fig, ax = plt.subplots(figsize=(7, 3))
    for i, (label, v, k) in enumerate(items):
        color = BLUE if k == "scamshield" else ORANGE if k == "baseline_lr" else "#a9a8a2"
        ax.barh(i, v, color=color, height=0.6)
        ax.text(v, i, f"  {v:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(items)), [x[0] for x in items])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("PR-AUC (test period)")
    ax.set_title("Ablation: value of payment-journey features")
    _save(fig, out / "ablation_pr_auc.png")


def feature_importance(metrics: dict[str, Any], out: Path) -> None:
    imp = sorted(metrics["feature_importance"].items(), key=lambda kv: kv[1])
    labels = ["Existing fraud score" if k == "baseline_logit" else FEATURE_DESCRIPTIONS.get(k, k) for k, _ in imp]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh(range(len(imp)), [v for _, v in imp], color=BLUE, height=0.6)
    ax.set_yticks(range(len(imp)), labels)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("CatBoost importance (PredictionValuesChange)")
    ax.set_title("ScamShield model — feature importance")
    _save(fig, out / "feature_importance.png")


def sensitivity(metrics: dict[str, Any], out: Path) -> None:
    sens = metrics.get("sensitivity")
    if not sens:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    panels = [
        ("journey_signal_strength", "signal_strength", "Journey signal strength (1 = default)", "Weaker behavioral signals"),
        ("scam_prevalence", "prevalence_multiplier", "Scam prevalence multiplier", "Prevalence drift (no recalibration)"),
        ("intervention_effectiveness", "effectiveness_multiplier", "Effectiveness multiplier", "Intervention effectiveness"),
    ]
    for ax, (key, xcol, xlabel, title) in zip(axes, panels):
        rows = sorted(sens[key], key=lambda r: r[xcol])
        x = [r[xcol] for r in rows]
        y = [100 * r["cost_reduction_pct"] for r in rows]
        ax.plot(x, y, color=BLUE, marker="o", ms=6)
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.0f}%", (xi, yi), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5, color=INK)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set(xlabel=xlabel, title=title)
        ax.set_ylim(min(-5, min(y) - 8), max(y) + 12)
    axes[0].set_ylabel("Total-cost reduction vs. baseline")
    _save(fig, out / "sensitivity.png")


def make_all_figures(metrics: dict[str, Any], predictions: pd.DataFrame, icfg: InterventionConfig, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pr_curves(predictions, metrics, out_dir)
    calibration(metrics, out_dir)
    intervention_mix(predictions, out_dir)
    cost_breakdown(metrics, out_dir)
    ablation(metrics, out_dir)
    feature_importance(metrics, out_dir)
    sensitivity(metrics, out_dir)
