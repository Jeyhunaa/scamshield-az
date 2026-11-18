"""Interactive dashboard charts (Altair — bundled with Streamlit, no extra dependency).

All charts read from the saved evaluation artifacts (metrics.json, predictions_test.parquet)
or from the live decision engine; nothing is recomputed differently from the pipeline.
ScamShield = blue, baseline = neutral gray throughout.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from scamshield.decision.optimizer import ACTIONS, InterventionConfig, decide_batch
from scamshield.features.journey import FEATURE_DESCRIPTIONS

from ui_kit import ACTION_COLORS, BASELINE, INK, INK_3, LINE_SOFT, MAP_FILLS, SCAMSHIELD

FONT = "Inter, ui-sans-serif, system-ui, sans-serif"
POLICY_SCALE = alt.Scale(domain=["ScamShield", "Baseline"], range=[SCAMSHIELD, BASELINE])


def _style(chart: alt.TopLevelMixin, height: int = 280) -> alt.TopLevelMixin:
    return (
        chart.properties(height=height)
        .configure(font=FONT, background="transparent")
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=INK_3, titleColor=INK_3, labelFontSize=11.5, titleFontSize=11.5, titleFontWeight=500,
            gridColor=LINE_SOFT, domainColor="#d0d5dd", tickColor="#d0d5dd",
        )
        .configure_legend(orient="top", title=None, labelFontSize=12, labelColor=INK, symbolStrokeWidth=3)
    )


def pr_curves(predictions: pd.DataFrame, metrics: dict) -> alt.TopLevelMixin:
    frames = []
    y = predictions["is_scam"].to_numpy()
    for col, name, key in [("enhanced_risk", "ScamShield", "scamshield"), ("baseline_risk", "Baseline", "baseline_lr")]:
        prec, rec, _ = precision_recall_curve(y, predictions[col])
        df = pd.DataFrame({"Recall": rec, "Precision": prec}).iloc[::-1].drop_duplicates("Recall", keep="last")
        df["Model"] = name
        df["PR-AUC"] = round(metrics["models"][key]["pr_auc"], 3)
        frames.append(df)
    data = pd.concat(frames)
    lines = alt.Chart(data).mark_line(strokeWidth=2, interpolate="step-after").encode(
        x=alt.X("Recall:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        y=alt.Y("Precision:Q", title="Precision", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("Model:N", scale=POLICY_SCALE),
        tooltip=["Model", alt.Tooltip("Recall:Q", format=".1%"), alt.Tooltip("Precision:Q", format=".1%"), "PR-AUC"],
    )
    prevalence = alt.Chart(pd.DataFrame({"p": [y.mean()]})).mark_rule(strokeDash=[3, 3], color=INK_3).encode(y=alt.Y("p:Q", title="Precision"))
    return _style(lines + prevalence)


def calibration(metrics: dict) -> alt.TopLevelMixin:
    rows = []
    for key, name in [("scamshield", "ScamShield"), ("baseline_lr", "Baseline")]:
        for r in metrics["calibration"][key]:
            rows.append({"Model": name, "Predicted": r["mean_predicted"], "Observed": r["observed_rate"], "n": r["n"]})
    data = pd.DataFrame(rows)
    lim = float(max(data["Predicted"].max(), data["Observed"].max()) * 1.05)
    diag = alt.Chart(pd.DataFrame({"x": [0, lim], "y": [0, lim]})).mark_line(strokeDash=[3, 3], color=INK_3, strokeWidth=1).encode(x="x:Q", y="y:Q")
    pts = alt.Chart(data).mark_line(point=alt.OverlayMarkDef(size=45, filled=True), strokeWidth=2).encode(
        x=alt.X("Predicted:Q", title="Mean predicted probability (decile bins)", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, lim])),
        y=alt.Y("Observed:Q", title="Observed scam rate", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, lim])),
        color=alt.Color("Model:N", scale=POLICY_SCALE),
        tooltip=["Model", alt.Tooltip("Predicted:Q", format=".2%"), alt.Tooltip("Observed:Q", format=".2%"), "n"],
    )
    return _style(diag + pts)


def interventions(predictions: pd.DataFrame) -> alt.TopLevelMixin:
    rows = []
    y = predictions["is_scam"].to_numpy()
    for col, name in [("scamshield_action", "ScamShield"), ("baseline_action", "Baseline")]:
        acts = predictions[col].to_numpy()
        for i, a in enumerate(ACTIONS[1:], start=1):
            n, hit = int((acts == i).sum()), int(((acts == i) & (y == 1)).sum())
            rows.append({"Action": a.capitalize(), "Policy": name, "Interventions": n, "On actual scams": hit, "On legitimate": n - hit})
    data = pd.DataFrame(rows)
    return _style(
        alt.Chart(data).mark_bar(cornerRadiusEnd=3, height=14).encode(
            y=alt.Y("Action:N", sort=["Warn", "Verify", "Hold"], title=None),
            yOffset=alt.YOffset("Policy:N", sort=["ScamShield", "Baseline"]),
            x=alt.X("Interventions:Q", title="Test transactions"),
            color=alt.Color("Policy:N", scale=POLICY_SCALE),
            tooltip=["Policy", "Action", "Interventions", "On actual scams", "On legitimate"],
        ),
        height=230,
    )


def cost_breakdown(e2e: dict) -> alt.TopLevelMixin:
    parts = [("residual_loss", "Residual scam loss", "#14365d"), ("operational_cost", "Operational", "#6b8fc4"), ("friction_cost", "Customer friction", "#c3d3ea")]
    rows = [
        {"Policy": name, "Component": label, "AZN": e2e[key][field], "order": i}
        for key, name in [("scamshield", "ScamShield"), ("baseline", "Baseline")]
        for i, (field, label, _) in enumerate(parts)
    ]
    data = pd.DataFrame(rows)
    bars = alt.Chart(data).mark_bar(height=26).encode(
        y=alt.Y("Policy:N", sort=["ScamShield", "Baseline"], title=None),
        x=alt.X("sum(AZN):Q", title="Total cost on test period (AZN)", axis=alt.Axis(format=",.0f")),
        color=alt.Color("Component:N", scale=alt.Scale(domain=[p[1] for p in parts], range=[p[2] for p in parts])),
        order=alt.Order("order:Q"),
        tooltip=["Policy", "Component", alt.Tooltip("AZN:Q", format=",.0f")],
    )
    totals = pd.DataFrame([{"Policy": n, "Total": e2e[k]["total_cost"]} for k, n in [("scamshield", "ScamShield"), ("baseline", "Baseline")]])
    labels = alt.Chart(totals).mark_text(align="left", dx=6, fontWeight=600, color=INK).encode(
        y=alt.Y("Policy:N", sort=["ScamShield", "Baseline"]), x="Total:Q", text=alt.Text("Total:Q", format=",.0f")
    )
    return _style(bars + labels, height=150)


ABLATION_NAMES = {
    "baseline_lr": "Baseline LR (transaction features)",
    "catboost_txn_only": "CatBoost · transaction features",
    "catboost_journey_only": "CatBoost · journey features only",
    "scamshield": "ScamShield · baseline score + journey",
    "catboost_all_features": "CatBoost · all raw features",
}


def ablation(metrics: dict) -> alt.TopLevelMixin:
    data = pd.DataFrame(
        [{"Model": ABLATION_NAMES[k], "PR-AUC": metrics["models"][k]["pr_auc"], "role": "ScamShield" if k == "scamshield" else "Baseline" if k == "baseline_lr" else "Other"} for k in ABLATION_NAMES]
    )
    color = alt.Color("role:N", scale=alt.Scale(domain=["ScamShield", "Baseline", "Other"], range=[SCAMSHIELD, BASELINE, "#d0d5dd"]), legend=None)
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=3, height=18).encode(
        y=alt.Y("Model:N", sort=list(ABLATION_NAMES.values()), title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X("PR-AUC:Q", scale=alt.Scale(domain=[0, 1])),
        color=color,
        tooltip=["Model", alt.Tooltip("PR-AUC:Q", format=".3f")],
    )
    text = bars.mark_text(align="left", dx=5, color=INK, fontSize=11.5).encode(text=alt.Text("PR-AUC:Q", format=".3f"), color=alt.value(INK))
    return _style(bars + text, height=210)


def importance(metrics: dict) -> alt.TopLevelMixin:
    data = pd.DataFrame(
        [{"Feature": "Existing fraud score" if k == "baseline_logit" else FEATURE_DESCRIPTIONS.get(k, k), "Importance": v} for k, v in metrics["feature_importance"].items()]
    ).sort_values("Importance", ascending=False)
    return _style(
        alt.Chart(data).mark_bar(cornerRadiusEnd=3, height=12, color=SCAMSHIELD).encode(
            y=alt.Y("Feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("Importance:Q", title="CatBoost importance (PredictionValuesChange)"),
            tooltip=["Feature", alt.Tooltip("Importance:Q", format=".1f")],
        ),
        height=300,
    )


def sensitivity(rows: list[dict], x: str, x_title: str) -> alt.TopLevelMixin:
    data = pd.DataFrame(rows)[[x, "cost_reduction_pct"]].sort_values(x)
    base = alt.Chart(data).encode(
        x=alt.X(f"{x}:Q", title=x_title),
        y=alt.Y("cost_reduction_pct:Q", title="Cost reduction vs baseline", axis=alt.Axis(format="%"), scale=alt.Scale(domainMin=0)),
        tooltip=[alt.Tooltip(f"{x}:Q", title=x_title), alt.Tooltip("cost_reduction_pct:Q", title="Cost reduction", format=".1%")],
    )
    line = base.mark_line(color=SCAMSHIELD, strokeWidth=2, point=alt.OverlayMarkDef(size=50, filled=True, color=SCAMSHIELD))
    labels = base.mark_text(dy=-11, fontSize=11, color=INK).encode(text=alt.Text("cost_reduction_pct:Q", format=".0%"))
    return _style(line + labels, height=200)


_P_TICKS = [-4, -3, -2, -1, 0]
_A_TICKS = [1, 2, 3, 4]


def decision_map(cfg: InterventionConfig, current: tuple[float, float] | None) -> alt.TopLevelMixin:
    """Action that minimises expected cost over (probability, amount), in log space."""
    nx, ny = 110, 90
    lp = np.linspace(-4, 0, nx + 1)
    la = np.linspace(1, 4.5, ny + 1)
    cx, cy = (lp[:-1] + lp[1:]) / 2, (la[:-1] + la[1:]) / 2
    P, A = np.meshgrid(cx, cy)
    Z = decide_batch(10 ** P.ravel(), 10 ** A.ravel(), cfg)
    X0, Y0 = np.meshgrid(lp[:-1], la[:-1])
    X1, Y1 = np.meshgrid(lp[1:], la[1:])
    data = pd.DataFrame({"x": X0.ravel(), "x2": X1.ravel(), "y": Y0.ravel(), "y2": Y1.ravel(), "Action": [ACTIONS[i].capitalize() for i in Z]})
    dom = [a.capitalize() for a in ACTIONS]
    p_axis = alt.Axis(values=_P_TICKS, labelExpr="datum.value == 0 ? '100%' : format(pow(10, datum.value), '.2~%')", title="Scam probability (log scale)")
    a_axis = alt.Axis(values=_A_TICKS, labelExpr="format(pow(10, datum.value), ',')", title="Amount at risk, AZN (log scale)")
    rect = alt.Chart(data).mark_rect(strokeWidth=0.6).encode(
        x=alt.X("x:Q", axis=p_axis, scale=alt.Scale(domain=[-4, 0], nice=False)),
        x2="x2", y=alt.Y("y:Q", axis=a_axis, scale=alt.Scale(domain=[1, 4.5], nice=False)), y2="y2",
        color=alt.Color("Action:N", scale=alt.Scale(domain=dom, range=[MAP_FILLS[a] for a in ACTIONS]), legend=None),
        stroke=alt.Stroke("Action:N", scale=alt.Scale(domain=dom, range=[MAP_FILLS[a] for a in ACTIONS]), legend=None),  # hides cell seams
        tooltip=["Action"],
    )
    centroids = (
        pd.DataFrame({"x": P.ravel(), "y": A.ravel(), "Action": [ACTIONS[i] for i in Z]})
        .groupby("Action", as_index=False)[["x", "y"]].mean()
    )
    centroids["label"] = centroids["Action"].str.upper()
    centroids["color"] = centroids["Action"].map(lambda a: ACTION_COLORS[a][0])
    text = alt.Chart(centroids).mark_text(fontWeight=700, fontSize=12).encode(x="x:Q", y="y:Q", text="label:N", color=alt.Color("color:N", scale=None))
    layers = rect + text
    if current is not None:
        p, amt = current
        pt = pd.DataFrame({"x": [np.log10(max(p, 1e-4))], "y": [np.log10(min(max(amt, 10), 10**4.5))], "label": ["Current transfer"]})
        layers = layers + alt.Chart(pt).mark_point(shape="circle", size=110, filled=True, color=INK, stroke="white", strokeWidth=2).encode(x="x:Q", y="y:Q", tooltip=["label"])
        layers = layers + alt.Chart(pt).mark_text(align="right" if p > 0.1 else "left", dx=-10 if p > 0.1 else 10, dy=16, fontSize=11.5, fontWeight=600, color=INK).encode(x="x:Q", y="y:Q", text="label:N")
    return _style(layers, height=360)
