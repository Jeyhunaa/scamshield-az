"""ScamShield AZ — analyst dashboard (Streamlit).

Run:  streamlit run app/dashboard.py
Every score and decision is computed live by the trained models and the decision engine
(in-process, or via the FastAPI backend if selected in the sidebar). Evaluation views read
the saved pipeline artifacts; nothing is recomputed differently from the pipeline.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st

from scamshield import __version__
from scamshield.config import ARTIFACTS_DIR
from scamshield.decision.optimizer import ACTIONS, ActionParams, Guardrails, InterventionConfig, decide_batch, fixed_threshold_actions
from scamshield.evaluation.metrics import policy_summary
from scamshield.features.journey import EVENT_TYPES, FEATURE_DESCRIPTIONS, JOURNEY_FEATURES
from scamshield.service import AssessmentService, journey_features_from_input, load_scenarios

# Sibling modules in app/ (Streamlit puts the script's folder on sys.path).
import charts  # noqa: E402
import ui_kit as ui  # noqa: E402
from customer_view import render_customer_view  # noqa: E402

st.set_page_config(page_title="ScamShield AZ · Analyst dashboard", page_icon="🛡️", layout="wide")
ui.inject_css()


# ---------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading trained models ...")
def get_service() -> AssessmentService:
    return AssessmentService.load(ARTIFACTS_DIR)


@st.cache_data
def get_metrics() -> dict:
    return json.loads((ARTIFACTS_DIR / "metrics.json").read_text())


@st.cache_data
def get_predictions() -> pd.DataFrame:
    return pd.read_parquet(ARTIFACTS_DIR / "predictions_test.parquet")


try:
    service = get_service()
    metrics = get_metrics()
    predictions = get_predictions()
except FileNotFoundError as exc:
    st.error(f"Artifacts missing — run `python scripts/run_pipeline.py` first.\n\n{exc}")
    st.stop()

scenarios = load_scenarios()


# ---------------------------------------------------------------- helpers
def run_assessment(tx: dict, journey: dict, hold_allowed: bool, cfg: InterventionConfig | None, api_url: str | None) -> dict:
    if api_url:
        import httpx

        payload = {"transaction": tx, "journey_features": _features_payload(journey), "hold_allowed": hold_allowed}
        resp = httpx.post(f"{api_url.rstrip('/')}/assess", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    return service.assess(tx, journey, hold_allowed=hold_allowed, intervention=cfg)


def _features_payload(journey: dict) -> dict:
    out = {}
    for k, v in journey.items():
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out[k] = None
        elif k in ("payee_pasted", "call_active"):
            out[k] = bool(v)
        elif k.startswith("n_"):
            out[k] = int(v)
        else:
            out[k] = float(v)
    return out


def current_config() -> InterventionConfig:
    """Intervention config from the Decision-policy widgets (falls back to YAML values).

    Read from session state at the top of the script so every tab uses the same, current values.
    """
    base, ss = service.intervention, st.session_state
    actions = {"allow": base.actions["allow"]}
    for a in ("warn", "verify", "hold"):
        p0 = base.actions[a]
        actions[a] = ActionParams(
            float(ss.get(f"e_{a}", p0.effectiveness)), float(ss.get(f"c_{a}", p0.operational_cost)), float(ss.get(f"f_{a}", p0.friction_cost))
        )
    g0 = base.guardrails
    guard = Guardrails(
        min_amount_verify=float(ss.get("g_verify", g0.min_amount_verify)),
        min_amount_hold=float(ss.get("g_hold", g0.min_amount_hold)),
        high_risk_floor=float(ss.get("g_floor", g0.high_risk_floor)),
        max_amount_without_warning=g0.max_amount_without_warning,
    )
    return replace(base, actions=actions, guardrails=guard)


def delta(value: float, fmt: str, lower_is_better: bool = False) -> tuple[str, str]:
    """Formatted change and its tone (good/bad/neutral)."""
    if abs(value) < 1e-12:
        return fmt.format(value), "neutral"
    good = (value < 0) if lower_is_better else (value > 0)
    return fmt.format(value), "good" if good else "bad"


def money(x: float) -> str:
    return f"{x:,.0f}"


def chart(c) -> None:
    st.altair_chart(c, width="stretch", theme=None)


# ---------------------------------------------------------------- sidebar
keys = list(scenarios)
with st.sidebar:
    st.html('<div class="ss-eyebrow">Scenarios</div>')
    choice = st.radio(
        "Scenario",
        keys,
        format_func=lambda k: scenarios[k]["title"],
        captions=[f"{'Legitimate' if scenarios[k]['kind'] == 'legitimate' else 'Scam'} · {scenarios[k]['intended_action'].capitalize()}" for k in keys],
        label_visibility="collapsed",
    )
    st.html(f'<div class="ss-note">{scenarios[choice]["story"]}</div>')
    st.divider()
    st.html('<div class="ss-eyebrow">Inference</div>')
    backend = st.radio("Inference backend", ["In-process (same code as API)", "FastAPI service"], index=0, label_visibility="collapsed")
    api_url = st.text_input("API URL", "http://127.0.0.1:8000") if backend.startswith("FastAPI") else None
    st.divider()
    st.html('<div class="ss-eyebrow">Customer banking app</div>')
    st.html(
        '<div class="ss-note">The standalone customer app is served by the API at '
        '<a href="http://127.0.0.1:8000/app/" target="_blank">127.0.0.1:8000/app</a>.</div>'
    )
    st.divider()
    st.html(
        '<div class="ss-note">Synthetic data only. Costs and intervention effectiveness are assumptions '
        "(configs/intervention.yaml). Results are not evidence of real-world banking performance.</div>"
    )

d = metrics["data"]
ui.header(
    [
        (f"Models loaded · v{__version__}", "#32d583"),
        ("Inference: FastAPI" if api_url else "Inference: in-process", None),
        ("Simulated data", "#fdb022"),
    ]
)

tab_assess, tab_models, tab_costs, tab_customer = st.tabs(["Assessment", "Model performance", "Decision policy", "Customer view"])

# ================================================================= ASSESSMENT
with tab_assess:
    sc = scenarios[choice]
    left, right = st.columns([5, 8], gap="large")

    with left:
        with st.container(border=True):
            st.html('<div class="ss-eyebrow">Transaction</div>')
            t = sc["transaction"]
            c1, c2 = st.columns(2)
            amount = c1.number_input("Amount (AZN)", min_value=1.0, value=float(t["amount"]), step=50.0, key=f"amt_{choice}")
            hour = c2.slider("Hour of day", 0, 23, int(t["hour"]), key=f"hour_{choice}")
            c1, c2 = st.columns(2)
            is_new = c1.toggle("New beneficiary", value=bool(t["is_new_beneficiary"]), key=f"new_{choice}")
            hold_allowed = c2.toggle("Hold permitted", value=True, key=f"hold_{choice}")
            with st.expander("Customer history (before this transfer)"):
                h1, h2 = st.columns(2)
                prior_cnt = h1.number_input("Previous transfers", 0, 10_000, int(t["prior_txn_count"]), key=f"pc_{choice}")
                avg = h2.number_input("Average amount", 0.0, 1e7, float(t["prior_avg_amount"] or 0), key=f"avg_{choice}")
                mx = h1.number_input("Largest amount", 0.0, 1e7, float(t["prior_max_amount"] or 0), key=f"mx_{choice}")
                hrs = h2.number_input("Hours since last transfer", 0.0, 1e5, float(t["hours_since_last_txn"] or 0), key=f"hrs_{choice}")
                l24 = h1.number_input("Transfers last 24h", 0, 100, int(t["txn_last_24h"]), key=f"l24_{choice}")
                l7 = h2.number_input("Transfers last 7d", 0, 500, int(t["txn_last_7d"]), key=f"l7_{choice}")
                a24 = h1.number_input("Amount sent last 24h", 0.0, 1e7, float(t["amount_last_24h"]), key=f"a24_{choice}")
                nben = h2.number_input("Distinct past beneficiaries", 0, 10_000, int(t["n_distinct_beneficiaries"]), key=f"nb_{choice}")
                tenure = h1.number_input("Tenure (days)", 0, 30_000, int(t["tenure_days"]), key=f"ten_{choice}")
            tx = {
                "amount": amount, "hour": hour, "is_new_beneficiary": is_new, "prior_txn_count": prior_cnt,
                "prior_avg_amount": avg if prior_cnt > 0 else None, "prior_max_amount": mx if prior_cnt > 0 else None,
                "txn_last_24h": l24, "txn_last_7d": max(l7, l24), "amount_last_24h": a24,
                "hours_since_last_txn": hrs if prior_cnt > 0 else None, "n_distinct_beneficiaries": nben, "tenure_days": tenure,
            }

        with st.container(border=True):
            st.html('<div class="ss-eyebrow">Payment journey</div>')
            mode = st.radio("Journey input", ["Recorded event log", "Behavioral summary"], horizontal=True, key=f"mode_{choice}", label_visibility="collapsed")
            scenario_features = journey_features_from_input(events=sc["journey_events"])
            if mode == "Recorded event log":
                ev_df = st.data_editor(
                    pd.DataFrame(sc["journey_events"]),
                    column_config={
                        "event_type": st.column_config.SelectboxColumn("Event", options=list(EVENT_TYPES), required=True),
                        "t_offset_s": st.column_config.NumberColumn("Seconds since login", min_value=0.0, format="%.1f", required=True),
                    },
                    num_rows="dynamic",
                    width="stretch",
                    height=300,
                    key=f"events_{choice}",
                )
                try:
                    journey = journey_features_from_input(events=ev_df.dropna().to_dict("records"))
                except ValueError as exc:
                    st.error(f"Invalid event log: {exc}")
                    st.stop()
            else:
                f = scenario_features
                s1, s2 = st.columns(2)
                journey = dict(f)
                journey["session_duration_s"] = s1.slider("Session duration (s)", 0.0, 900.0, float(f["session_duration_s"]), key=f"sd_{choice}")
                journey["confirm_time_s"] = s2.slider("Confirmation time (s)", 0.0, 180.0, float(np.nan_to_num(f["confirm_time_s"])), key=f"ct_{choice}")
                journey["n_app_switches"] = s1.slider("App switches", 0, 10, int(f["n_app_switches"]), key=f"sw_{choice}")
                journey["background_time_s"] = s2.slider("Time outside app (s)", 0.0, 600.0, float(f["background_time_s"]), key=f"bg_{choice}")
                journey["n_amount_edits"] = s1.slider("Amount edits", 0, 10, int(f["n_amount_edits"]), key=f"ae_{choice}")
                journey["n_payee_edits"] = s2.slider("Beneficiary edits", 0, 10, int(f["n_payee_edits"]), key=f"pe_{choice}")
                journey["n_back_navigations"] = s1.slider("Back-navigations", 0, 10, int(f["n_back_navigations"]), key=f"bn_{choice}")
                journey["n_payment_screen_visits"] = s2.slider("Payment screen visits", 1, 10, int(max(1, f["n_payment_screen_visits"])), key=f"pv_{choice}")
                journey["time_to_payment_s"] = s1.slider("Login → payments (s)", 0.0, 300.0, float(np.nan_to_num(f["time_to_payment_s"])), key=f"tp_{choice}")
                journey["n_browse_events"] = s2.slider("Other screens browsed", 0, 15, int(f["n_browse_events"]), key=f"br_{choice}")
                journey["call_active"] = float(s1.toggle("Phone call active", bool(f["call_active"]), key=f"ca_{choice}"))
                journey["payee_pasted"] = float(s2.toggle("Beneficiary pasted", bool(f["payee_pasted"]), key=f"pp_{choice}"))
                journey["background_time_s"] = min(journey["background_time_s"], journey["session_duration_s"])
                journey["background_share"] = journey["background_time_s"] / journey["session_duration_s"] if journey["session_duration_s"] > 0 else 0.0

    with right:
        try:
            res = run_assessment(tx, journey, hold_allowed, current_config() if api_url is None else None, api_url)
        except Exception as exc:  # surfaced to the presenter rather than crashing the page
            st.error(f"Assessment failed: {exc}")
            st.stop()
        st.session_state["last_assessment"] = {"tx": tx, "journey": journey, "hold_allowed": hold_allowed}

        ui.decision_summary(res)

        ui.section("Expected cost by action", "EC(a) = p·L·(1−eₐ) + Cₐ + (1−p)·Fₐ, with p = ScamShield probability and L = amount × loss-given-scam (AZN)")
        costs = res["expected_costs"]
        top = max(c["expected_cost"] for c in costs.values()) or 1.0
        rows, muted = [], []
        for i, a in enumerate(ACTIONS):
            c = costs[a]
            if not c["eligible"]:
                muted.append(i)
            bar = f'<span class="ss-bar" style="width:{max(2, 70 * c["expected_cost"] / top):.0f}px;background:{"#1f5aa6" if a == res["recommended_action"] else "#d0d5dd"}"></span>'
            rows.append(
                [
                    ui.action_tag(a),
                    f'<span class="ss-cost">{bar}{c["expected_cost"]:,.2f}</span>',
                    f'{c["residual_loss"]:,.2f}',
                    f'{c["operational_cost"]:,.2f}',
                    f'{c["friction_cost"]:,.2f}',
                    "Recommended" if a == res["recommended_action"] else ("Permitted" if c["eligible"] else f'<span title="{c["ineligible_reason"]}">Not permitted</span><span class="sub">{c["ineligible_reason"]}</span>'),
                ]
            )
        ui.table(
            ["Action", "Expected cost", "Residual scam loss", "Operational", "Customer friction", "Status"],
            rows,
            numeric=(1, 2, 3, 4),
            highlight=(ACTIONS.index(res["recommended_action"]),),
            muted=muted,
        )

        f1, f2 = st.columns(2, gap="medium")
        with f1:
            ui.section("Why ScamShield scored it this way", "SHAP contributions, log-odds · red raises risk, green lowers it")
            ui.factor_list(res["risk_factors"])
        with f2:
            ui.section("What drives the existing score", "Logistic-regression contributions (coef × standardized value)")
            ui.factor_list(res["baseline_drivers"])

        with st.expander("Extracted journey features"):
            ui.table(
                ["Feature", "Value"],
                [[FEATURE_DESCRIPTIONS[k], "—" if res["journey_features"].get(k) is None else f'{res["journey_features"][k]:,.2f}'.rstrip("0").rstrip(".")] for k in JOURNEY_FEATURES],
                numeric=(1,),
            )

# ================================================================= MODEL PERFORMANCE
with tab_models:
    m, e2e = metrics["models"], metrics["end_to_end"]
    b, s = e2e["baseline"], e2e["scamshield"]
    st.html(
        f'<div class="ss-note" style="margin:8px 0 12px">Held-out test period (Oct–Dec 2025): {d["test"]["n"]:,} transfers, {d["test"]["n_scam"]} scams. '
        f'Models trained on {d["train"]["n"]:,} transfers; calibrated and baseline thresholds tuned on {d["val"]["n"]:,} validation transfers. '
        "Both approaches are evaluated on the same transactions with the same cost assumptions.</div>"
    )
    r1 = m["scamshield"]["at_fpr"][0]["recall"] - m["baseline_lr"]["at_fpr"][0]["recall"]
    ui.kpi_strip(
        [
            ("PR-AUC", f"{m['scamshield']['pr_auc']:.3f}", *delta(m["scamshield"]["pr_auc"] - m["baseline_lr"]["pr_auc"], "{:+.3f} vs baseline")),
            ("Recall at 1% FPR", f"{m['scamshield']['at_fpr'][0]['recall']:.1%}", *delta(r1 * 100, "{:+.1f} pts vs baseline")),
            ("Total cost (AZN)", money(s["total_cost"]), *delta(-e2e["savings"]["cost_reduction_pct"] * 100, "{:+.1f}% vs baseline", lower_is_better=True)),
            ("Unnecessary holds", f"{s['unnecessary_holds']:.0f}", *delta(s["unnecessary_holds"] - b["unnecessary_holds"], "{:+.0f} vs baseline", lower_is_better=True)),
            ("Scam loss prevented", f"{s['prevented_loss_share']:.1%}", *delta((s["prevented_loss_share"] - b["prevented_loss_share"]) * 100, "{:+.1f} pts vs baseline")),
        ]
    )
    ci = metrics["bootstrap_95ci"]
    st.html(
        '<div class="ss-note" style="margin-top:6px">Paired bootstrap 95% CI (stratified by label): PR-AUC gain '
        f"[{ci['pr_auc_gain']['ci_low']:+.3f}, {ci['pr_auc_gain']['ci_high']:+.3f}], cost reduction "
        f"[{ci['total_cost_reduction']['ci_low']:,.0f}, {ci['total_cost_reduction']['ci_high']:,.0f}] AZN.</div>"
    )

    ui.section("End-to-end comparison", "Baseline: transaction score + fixed thresholds · ScamShield: journey-aware model + cost-aware optimizer")

    def cmp_row(label: str, bv: float, sv: float, fmt: str, lower_is_better: bool, dfmt: str) -> list[str]:
        txt, tone = delta(sv - bv, dfmt, lower_is_better)
        return [label, fmt.format(bv), fmt.format(sv), f'<span class="{tone}">{txt}</span>']

    ui.table(
        ["Metric", "Baseline", "ScamShield", "Change"],
        [
            cmp_row("Scam recall (any intervention)", b["scam_recall_any_intervention"], s["scam_recall_any_intervention"], "{:.1%}", False, "{:+.1%}"),
            cmp_row("Legitimate transfers interrupted (FPR)", b["legit_intervention_rate"], s["legit_intervention_rate"], "{:.2%}", True, "{:+.2%}"),
            ["Warn / Verify / Hold", " / ".join(f"{b['action_counts'][a]:.0f}" for a in ACTIONS[1:]), " / ".join(f"{s['action_counts'][a]:.0f}" for a in ACTIONS[1:]), ""],
            cmp_row("Unnecessary holds", b["unnecessary_holds"], s["unnecessary_holds"], "{:,.0f}", True, "{:+,.0f}"),
            cmp_row("Prevented scam loss (AZN)", b["prevented_loss"], s["prevented_loss"], "{:,.0f}", False, "{:+,.0f}"),
            cmp_row("Residual scam loss (AZN)", b["residual_loss"], s["residual_loss"], "{:,.0f}", True, "{:+,.0f}"),
            cmp_row("Operational + friction cost (AZN)", b["operational_cost"] + b["friction_cost"], s["operational_cost"] + s["friction_cost"], "{:,.0f}", True, "{:+,.0f}"),
            cmp_row("Total cost (AZN)", b["total_cost"], s["total_cost"], "{:,.0f}", True, "{:+,.0f}"),
        ],
        numeric=(1, 2, 3),
        highlight=(7,),
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        ui.section("Precision–recall", "Dotted line: scam prevalence in the test period")
        chart(charts.pr_curves(predictions, metrics))
    with c2:
        ui.section("Calibration", f"ECE · ScamShield {m['scamshield']['ece']:.4f} · Baseline {m['baseline_lr']['ece']:.4f}")
        chart(charts.calibration(metrics))

    c1, c2 = st.columns(2, gap="large")
    with c1:
        ui.section("Interventions issued", "Hover for how many hit actual scams")
        chart(charts.interventions(predictions))
    with c2:
        ui.section("Estimated costs", "Residual scam loss + operational cost + customer friction")
        chart(charts.cost_breakdown(e2e))
        att = metrics["attribution_total_cost"]
        ui.section("Where the saving comes from", "Total cost (AZN) for each score × policy combination")
        ui.table(
            ["Score", "Fixed thresholds", "Cost-aware optimizer"],
            [
                ["Baseline score", money(att["baseline_score+fixed_thresholds"]), money(att["baseline_score+cost_optimizer"])],
                ["ScamShield score", money(att["scamshield_score+fixed_thresholds"]), f'<b>{money(att["scamshield_score+cost_optimizer"])}</b>'],
            ],
            numeric=(1, 2),
        )

    ui.section("Classification metrics", "Test period · equal-FPR comparisons read the same false-positive budget for every model")
    rows = []
    for key, label in charts.ABLATION_NAMES.items():
        mm = m[key]
        rows.append(
            [label, f"{mm['pr_auc']:.3f}", f"{mm['roc_auc']:.3f}", f"{mm['brier']:.4f}", f"{mm['ece']:.4f}"]
            + [f"{r['recall']:.1%}" for r in mm["at_fpr"]]
            + [f"{r['precision']:.1%}" for r in mm["at_fpr"]]
        )
    fprs = [f"{int(r['target_fpr'] * 100)}%" for r in m["scamshield"]["at_fpr"]]
    ui.table(
        ["Model", "PR-AUC", "ROC-AUC", "Brier", "ECE", *[f"Recall @{f}" for f in fprs], *[f"Precision @{f}" for f in fprs]],
        rows,
        numeric=tuple(range(1, 5 + 2 * len(fprs))),
        highlight=(list(charts.ABLATION_NAMES).index("scamshield"),),
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        ui.section("Ablation: value of payment-journey features", "PR-AUC on the test period")
        chart(charts.ablation(metrics))
    with c2:
        ui.section("ScamShield feature importance")
        chart(charts.importance(metrics))

    if "sensitivity" in metrics:
        sens = metrics["sensitivity"]
        ui.section("Sensitivity analysis", "Total-cost reduction of ScamShield vs. baseline under different assumptions")
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.html('<div class="ss-eyebrow">Weaker behavioral signals</div>')
            chart(charts.sensitivity(sens["journey_signal_strength"], "signal_strength", "Journey signal strength (1 = default)"))
        with c2:
            st.html('<div class="ss-eyebrow">Prevalence drift (no recalibration)</div>')
            chart(charts.sensitivity(sens["scam_prevalence"], "prevalence_multiplier", "Scam prevalence multiplier"))
        with c3:
            st.html('<div class="ss-eyebrow">Intervention effectiveness</div>')
            chart(charts.sensitivity(sens["intervention_effectiveness"], "effectiveness_multiplier", "Effectiveness multiplier"))
        st.html(
            '<div class="ss-note">Weaker signals: data regenerated and both models retrained. Prevalence: test scams re-weighted, '
            "models not recalibrated. Effectiveness: baseline thresholds re-tuned on validation for each assumption.</div>"
        )

# ================================================================= DECISION POLICY
with tab_costs:
    st.html(
        '<div class="ss-note" style="margin:8px 0 12px">Change the cost assumptions and see how ScamShield\'s decisions and costs respond on the '
        "held-out test transactions. Predictions are fixed; only the decision layer is re-run. The baseline keeps its fixed thresholds, "
        "which is precisely its limitation. Changes also apply to the Assessment tab (in-process backend).</div>"
    )
    base_cfg = service.intervention
    with st.container(border=True):
        cols = st.columns(3, gap="large")
        for col, a in zip(cols, ["warn", "verify", "hold"]):
            p0 = base_cfg.actions[a]
            with col:
                st.html(ui.action_tag(a))
                st.slider("Effectiveness", 0.0, 0.99, float(p0.effectiveness), 0.01, key=f"e_{a}", help=f"Probability that '{a}' stops a scam")
                st.number_input("Operational cost (AZN)", 0.0, 500.0, float(p0.operational_cost), 0.5, key=f"c_{a}")
                st.number_input("Customer friction cost (AZN)", 0.0, 1000.0, float(p0.friction_cost), 0.5, key=f"f_{a}")
        st.divider()
        st.html('<div class="ss-eyebrow">Policy guardrails</div>')
        g0 = base_cfg.guardrails
        g1, g2, g3, g4 = st.columns([3, 3, 3, 2], gap="large", vertical_alignment="bottom")
        g1.number_input("Minimum amount for verify (AZN)", 0.0, 1e5, float(g0.min_amount_verify), 10.0, key="g_verify")
        g2.number_input("Minimum amount for hold (AZN)", 0.0, 1e5, float(g0.min_amount_hold), 10.0, key="g_hold")
        g3.slider("Never allow above risk", 0.05, 1.0, float(g0.high_risk_floor), 0.05, key="g_floor")
        if g4.button("Reset assumptions", width="stretch", help="Restore the values in configs/intervention.yaml"):
            for key in list(st.session_state):
                if key.startswith(("e_", "c_", "f_", "g_")):
                    del st.session_state[key]
            st.rerun()
    cfg = current_config()

    y = predictions["is_scam"].to_numpy()
    amt = predictions["amount"].to_numpy()
    base_act = fixed_threshold_actions(predictions["baseline_risk"].to_numpy(), amt, service.baseline_thresholds, cfg)
    ss_act = decide_batch(predictions["enhanced_risk"].to_numpy(), amt, cfg)
    pb, ps = policy_summary(base_act, y, amt, cfg), policy_summary(ss_act, y, amt, cfg)

    st.html('<div style="height:6px"></div>')
    ui.kpi_strip(
        [
            ("ScamShield total cost (AZN)", money(ps["total_cost"]), *delta((ps["total_cost"] - pb["total_cost"]) / pb["total_cost"] * 100, "{:+.1f}% vs baseline", True)),
            ("Scam loss prevented (AZN)", money(ps["prevented_loss"]), *delta(ps["prevented_loss"] - pb["prevented_loss"], "{:+,.0f} vs baseline")),
            ("Legitimate transfers interrupted", f"{ps['legit_intervention_rate']:.2%}", *delta((ps["legit_intervention_rate"] - pb["legit_intervention_rate"]) * 100, "{:+.2f} pts vs baseline", True)),
            ("Unnecessary holds", f"{ps['unnecessary_holds']:.0f}", *delta(ps["unnecessary_holds"] - pb["unnecessary_holds"], "{:+.0f} vs baseline", True)),
        ]
    )

    c1, c2 = st.columns([7, 5], gap="large")
    with c1:
        ui.section("Decision map", "Action with the lowest expected cost for each risk × amount · the dot is the transfer on the Assessment tab")
        last = st.session_state.get("last_assessment")
        chart(charts.decision_map(cfg, (res["enhanced_risk"], last["tx"]["amount"]) if last else None))
        st.html('<div class="ss-note">Boundaries move as you change effectiveness and costs. Guardrails (minimum amounts, risk floor) carve out regions.</div>')
    with c2:
        ui.section("Intervention frequency", "Test transactions")
        ui.table(
            ["Action", "Baseline", "ScamShield"],
            [[ui.action_tag(a), f"{pb['action_counts'][a]:,.0f}", f"{ps['action_counts'][a]:,.0f}"] for a in ACTIONS],
            numeric=(1, 2),
        )
        ui.section("Cost components (AZN)")
        ui.table(
            ["Component", "Baseline", "ScamShield"],
            [
                ["Residual scam loss", money(pb["residual_loss"]), money(ps["residual_loss"])],
                ["Operational", money(pb["operational_cost"]), money(ps["operational_cost"])],
                ["Customer friction", money(pb["friction_cost"]), money(ps["friction_cost"])],
                ["Total", money(pb["total_cost"]), money(ps["total_cost"])],
            ],
            numeric=(1, 2),
            highlight=(3,),
        )

# ================================================================= CUSTOMER VIEW
with tab_customer:
    st.html(
        '<div class="ss-note" style="margin:8px 0 12px">A compact version of the customer experience for this dashboard. '
        'The full standalone banking app runs at <a href="http://127.0.0.1:8000/app/" target="_blank">127.0.0.1:8000/app</a>.</div>'
    )
    render_customer_view(
        choice,
        scenarios,
        lambda tx, journey: run_assessment(tx, journey, True, current_config() if api_url is None else None, api_url),
    )
