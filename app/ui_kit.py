"""Dashboard visual system: CSS and small HTML components (tables, KPI strips, tags).

Kept separate from dashboard.py so the page code reads as layout + logic only.
Palette matches the customer app (web/src/styles.css).
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape

import streamlit as st

INK, INK_2, INK_3, LINE, LINE_SOFT = "#0f1b2d", "#33415a", "#64708a", "#e3e7ee", "#eef1f5"
BRAND, BRAND_INK, SAND, LINK = "#10305a", "#0b2340", "#c9a867", "#1f5aa6"
SCAMSHIELD, BASELINE = "#1f5aa6", "#a3abb8"
RISK_UP, RISK_DOWN = "#b42318", "#067647"
MAP_FILLS = {"allow": "#d3f3df", "warn": "#fdecc0", "verify": "#fcd9bd", "hold": "#fbd0cc"}  # decision-map regions
ACTION_COLORS = {  # (text, background)
    "allow": ("#067647", "#ecfdf3"),
    "warn": ("#b54708", "#fffaeb"),
    "verify": ("#9a3412", "#fff4ed"),
    "hold": ("#b42318", "#fef3f2"),
}

CSS = f"""
<style>
.block-container {{ max-width: 1440px; padding-top: 4.2rem; padding-bottom: 3rem; }}
.stAppDeployButton, [data-testid="stDecoration"] {{ display: none; }}
h1, h2, h3, h4, h5 {{ letter-spacing: -0.01em; }}
.num, .ss-table td.n {{ font-variant-numeric: tabular-nums; }}

/* header: dark identity band */
.ss-header {{ position:relative; overflow:hidden; display:flex; align-items:center; justify-content:space-between; gap:16px; padding: 16px 20px; margin-bottom: 6px;
  background:{BRAND_INK}; border-radius: 12px; color:#fff; animation: ssFade .5s ease-out both; }}
.ss-header::after {{ content:""; position:absolute; right:-40px; top:-60px; width:340px; height:200px; pointer-events:none; opacity:.14;
  background: repeating-radial-gradient(ellipse at 100% 0%, transparent 0 14px, {SAND} 14px 15.5px); }}
.ss-brand {{ display:flex; align-items:center; gap:14px; flex: 0 1 auto; min-width: 0; position:relative; z-index:1; }}
.ss-mark {{ width:34px; height:38px; background:{SAND}; color:{BRAND_INK}; font-weight:800; font-size:17px; display:flex; align-items:center; justify-content:center; padding-bottom:3px;
  clip-path: polygon(50% 0, 100% 14%, 100% 52%, 50% 100%, 0 52%, 0 14%); animation: ssPop .6s cubic-bezier(.34,1.45,.64,1) both; }}
.ss-brand .name {{ font-size: 18px; font-weight: 680; color:#fff; line-height:1.15; letter-spacing:-0.015em; }}
.ss-brand .desc {{ font-size: 13px; color: rgba(255,255,255,.66); }}
.ss-chips {{ display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; flex: 0 0 auto; position:relative; z-index:1; }}
.ss-chip {{ display:inline-flex; align-items:center; gap:7px; font-size:12px; font-weight:560; color:rgba(255,255,255,.88); border:1px solid rgba(255,255,255,.16); border-radius:999px; padding:4px 11px; background:rgba(255,255,255,.06); white-space:nowrap; }}
.ss-chip .led {{ width:7px; height:7px; border-radius:50%; position:relative; }}
.ss-chip .led.live::after {{ content:""; position:absolute; inset:-3px; border-radius:50%; background:inherit; opacity:.35; animation: ssBreathe 2.4s ease-in-out infinite; }}

/* tabs */
[data-testid="stTabs"] [role="tablist"] {{ gap: 4px; border-bottom: 1px solid {LINE}; }}
[data-testid="stTab"] {{ padding: 10px 16px !important; margin-right: 4px; }}
[data-testid="stTab"] p {{ font-size: 14px !important; font-weight: 600 !important; }}
[data-testid="stTab"][aria-selected="false"] p {{ color: {INK_3}; }}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background-color: {BRAND}; height: 2.5px; }}
[data-testid="stTab"] {{ transition: color .15s ease; }}
[data-testid="stTabs"] [role="tabpanel"] {{ animation: ssFade .38s ease-out both; }}

/* section headings */
.ss-section {{ margin: 18px 0 8px; }}
.ss-section .t {{ font-size: 15px; font-weight: 650; color:{INK}; }}
.ss-section .s {{ font-size: 13px; color:{INK_3}; margin-top: 2px; }}
.ss-eyebrow {{ font-size: 11px; font-weight: 650; letter-spacing: .06em; text-transform: uppercase; color:{INK_3}; margin-bottom: 6px; }}
.ss-note {{ font-size: 12.5px; color:{INK_3}; line-height: 1.5; }}

/* KPI strip: one panel, divided cells (not a grid of cards) */
.ss-kpis {{ display:grid; border:1px solid {LINE}; border-radius:10px; background:#fff; overflow:hidden; }}
.ss-kpi {{ padding: 14px 16px; border-left:1px solid {LINE_SOFT}; min-width:0; }}
.ss-kpi:first-child {{ border-left: 0; }}
.ss-kpi .l {{ font-size:12px; color:{INK_3}; font-weight:550; }}
.ss-kpi .v {{ font-size:24px; font-weight:650; color:{INK}; font-variant-numeric: tabular-nums; letter-spacing:-0.01em; margin-top:2px; }}
.ss-kpi .d {{ font-size:12px; font-weight:600; margin-top:2px; font-variant-numeric: tabular-nums; }}
.ss-kpi .d.good {{ color:{RISK_DOWN}; }} .ss-kpi .d.bad {{ color:{RISK_UP}; }} .ss-kpi .d.neutral {{ color:{INK_3}; }}

/* decision summary */
.ss-decision {{ display:grid; grid-template-columns: 1fr 1fr 1.1fr; border:1px solid {LINE}; border-radius:10px; overflow:hidden; background:#fff; }}
.ss-decision > div {{ padding: 14px 16px; border-left: 1px solid {LINE_SOFT}; }}
.ss-decision > div:first-child {{ border-left: 0; }}
.ss-decision .l {{ font-size:12px; color:{INK_3}; font-weight:550; }}
.ss-decision .v {{ font-size:28px; font-weight:650; letter-spacing:-0.015em; font-variant-numeric: tabular-nums; margin: 2px 0 6px; }}
.ss-decision .rec {{ background:#fafbfc; }}
.ss-decision .rec .big {{ margin: 8px 0 6px; }}
.ss-tag {{ display:inline-block; font-size:11.5px; font-weight:700; letter-spacing:.05em; padding:3px 9px; border-radius:6px; }}
.ss-tag.lg {{ font-size:15px; padding:6px 14px; }}
.ss-explain {{ font-size:13.5px; color:{INK_2}; margin: 10px 2px 0; line-height:1.5; }}

/* tables */
.ss-table {{ width:100%; border-collapse:separate; border-spacing:0; border:1px solid {LINE}; border-radius:10px; overflow:hidden; background:#fff; font-size:13.5px; }}
.ss-table th {{ text-align:left; font-size:11.5px; font-weight:600; letter-spacing:.03em; color:{INK_3}; background:#f8f9fb; padding:9px 12px; border-bottom:1px solid {LINE}; line-height:1.3; }}
.ss-table td {{ padding:9px 12px; border-bottom:1px solid {LINE_SOFT}; color:{INK}; vertical-align:middle; }}
.ss-table tr:last-child td {{ border-bottom:0; }}
.ss-table th.n, .ss-table td.n {{ text-align:right; }}
.ss-table tr.hl td {{ background:#f3f6fb; font-weight:600; }}
.ss-table tr.hl td:first-child {{ box-shadow: inset 3px 0 0 {BRAND}; }}
.ss-table tr.muted td {{ color:{INK_3}; }}
.ss-table td .sub {{ display:block; font-size:12px; color:{INK_3}; font-weight:400; }}
.ss-bar {{ height:6px; border-radius:3px; background:{SCAMSHIELD}; display:inline-block; vertical-align:middle; flex: 0 0 auto; }}
.ss-cost {{ display:inline-flex; align-items:center; justify-content:flex-end; gap:10px; white-space:nowrap; }}
.good {{ color:{RISK_DOWN}; }} .bad {{ color:{RISK_UP}; }}

/* factor contributions */
.ss-factor {{ display:grid; grid-template-columns: minmax(0,1fr) 56px; grid-template-areas: 'n n' 't c'; column-gap:10px; row-gap:4px; align-items:center; padding:8px 0; border-bottom:1px solid {LINE_SOFT}; font-size:13px; }}
.ss-factor:last-child {{ border-bottom:0; }}
.ss-factor .n {{ grid-area:n; color:{INK}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.ss-factor .n b {{ font-weight:600; font-variant-numeric: tabular-nums; }}
.ss-factor .track {{ grid-area:t; position:relative; height:8px; background:{LINE_SOFT}; border-radius:4px; }}
.ss-factor .fill {{ position:absolute; top:0; height:8px; border-radius:4px; }}
.ss-factor .c {{ grid-area:c; text-align:right; font-variant-numeric: tabular-nums; font-weight:600; font-size:12.5px; }}

/* motion */
@keyframes ssFade {{ from {{ opacity:0; transform: translateY(6px); }} to {{ opacity:1; transform:none; }} }}
@keyframes ssPop {{ 0% {{ opacity:0; transform: scale(.7); }} 100% {{ opacity:1; transform: scale(1); }} }}
@keyframes ssGrowX {{ from {{ transform: scaleX(0); }} to {{ transform: scaleX(1); }} }}
@keyframes ssBreathe {{ 0%,100% {{ transform: scale(.8); opacity:.15; }} 50% {{ transform: scale(1.5); opacity:.4; }} }}
.ss-kpi {{ animation: ssFade .45s ease-out both; transition: background .15s ease; }}
.ss-kpi:hover {{ background:#fafbfd; }}
.ss-kpi:nth-child(2) {{ animation-delay:.05s; }} .ss-kpi:nth-child(3) {{ animation-delay:.1s; }} .ss-kpi:nth-child(4) {{ animation-delay:.15s; }} .ss-kpi:nth-child(5) {{ animation-delay:.2s; }}
.ss-decision > div {{ animation: ssFade .45s ease-out both; }}
.ss-decision > div:nth-child(2) {{ animation-delay:.07s; }} .ss-decision > div:nth-child(3) {{ animation-delay:.14s; }}
.ss-tag.lg {{ animation: ssPop .5s .2s cubic-bezier(.34,1.45,.64,1) both; }}
.ss-table tbody tr {{ animation: ssFade .35s ease-out both; transition: background .12s ease; }}
.ss-table tbody tr:hover td {{ background:#f9fafc; }}
.ss-table tbody tr.hl:hover td {{ background:#eef3fa; }}
.ss-table tbody tr:nth-child(2) {{ animation-delay:.04s; }} .ss-table tbody tr:nth-child(3) {{ animation-delay:.08s; }} .ss-table tbody tr:nth-child(4) {{ animation-delay:.12s; }}
.ss-table tbody tr:nth-child(5) {{ animation-delay:.16s; }} .ss-table tbody tr:nth-child(n+6) {{ animation-delay:.2s; }}
.ss-bar {{ transform-origin:left; animation: ssGrowX .7s cubic-bezier(.22,.8,.24,1) both; }}
.ss-factor .fill {{ animation: ssGrowX .7s cubic-bezier(.22,.8,.24,1) both; }}
.ss-factor .fill.pos {{ transform-origin:left; }} .ss-factor .fill.neg {{ transform-origin:right; }}
.ss-factor {{ animation: ssFade .4s ease-out both; }}
.ss-factor:nth-child(2) {{ animation-delay:.05s; }} .ss-factor:nth-child(3) {{ animation-delay:.1s; }} .ss-factor:nth-child(4) {{ animation-delay:.15s; }} .ss-factor:nth-child(5) {{ animation-delay:.2s; }}
.ss-meter {{ margin: 2px 0 8px; }}
.ss-meter .track {{ position:relative; height:6px; border-radius:3px; background:{LINE_SOFT}; overflow:hidden; }}
.ss-meter .fill {{ position:absolute; left:0; top:0; bottom:0; border-radius:3px; transform-origin:left; animation: ssGrowX .9s .1s cubic-bezier(.22,.8,.24,1) both; }}
.ss-meter .ticks {{ display:flex; justify-content:space-between; font-size:10px; color:{INK_3}; margin-top:3px; font-variant-numeric: tabular-nums; }}
[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"] {{ animation: ssFade .5s ease-out both; }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; transition: none !important; }} }}

/* sidebar */
section[data-testid="stSidebar"] .ss-eyebrow {{ margin-top: 4px; }}
section[data-testid="stSidebar"] [data-testid="stRadio"] label p {{ font-size: 13.5px; }}
section[data-testid="stSidebar"] [role="radiogroup"] > div {{ padding: 6px 10px; margin: 1px 0; border-radius: 8px; transition: background .15s ease, box-shadow .2s ease; }}
section[data-testid="stSidebar"] [role="radiogroup"] > div:hover {{ background: #eef1f5; }}
section[data-testid="stSidebar"] [role="radiogroup"] > div[data-selected="true"] {{ background: #e6edf7; box-shadow: inset 3px 0 0 {BRAND}; }}
div[data-testid="stExpander"] details summary p {{ font-weight: 600; }}
</style>
"""


def inject_css() -> None:
    st.html(CSS)


def header(chips: Sequence[tuple[str, str | None]]) -> None:
    """Product header. chips: (text, led colour or None)."""
    def chip(text: str, led: str | None) -> str:
        dot = f'<span class="led live" style="background:{led}"></span>' if led else ""
        return f'<span class="ss-chip">{dot}{escape(text)}</span>'

    chip_html = "".join(chip(t, c) for t, c in chips)
    st.html(
        '<div class="ss-header"><div class="ss-brand"><div class="ss-mark">S</div>'
        '<div><div class="name">ScamShield AZ</div><div class="desc">APP-scam risk scoring and cost-aware intervention · decision support</div></div></div>'
        f'<div class="ss-chips">{chip_html}</div></div>'
    )


def section(title: str, subtitle: str | None = None) -> None:
    sub = f'<div class="s">{subtitle}</div>' if subtitle else ""
    st.html(f'<div class="ss-section"><div class="t">{title}</div>{sub}</div>')


def action_tag(action: str, large: bool = False) -> str:
    fg, bg = ACTION_COLORS[action]
    return f'<span class="ss-tag{" lg" if large else ""}" style="color:{fg};background:{bg}">{action.upper()}</span>'


def fmt_pct(p: float) -> str:
    return f"{p:.2%}" if p < 0.1 else f"{p:.1%}"


def kpi_strip(items: Sequence[tuple[str, str, str | None, str]]) -> None:
    """items: (label, value, delta text or None, tone: good|bad|neutral)."""
    cells = "".join(
        f'<div class="ss-kpi"><div class="l">{escape(l)}</div><div class="v">{v}</div>'
        + (f'<div class="d {tone}">{d}</div>' if d else "")
        + "</div>"
        for l, v, d, tone in items
    )
    st.html(f'<div class="ss-kpis" style="grid-template-columns: repeat({len(items)}, minmax(0, 1fr))">{cells}</div>')


def table(
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
    numeric: Sequence[int] = (),
    highlight: Sequence[int] = (),
    muted: Sequence[int] = (),
) -> None:
    """Static HTML table. Cell strings may contain trusted HTML (built in this app)."""
    head = "".join(f'<th class="{"n" if i in numeric else ""}">{c}</th>' for i, c in enumerate(columns))
    body = ""
    for r, row in enumerate(rows):
        cls = "hl" if r in highlight else "muted" if r in muted else ""
        body += f'<tr class="{cls}">' + "".join(f'<td class="{"n" if i in numeric else ""}">{v}</td>' for i, v in enumerate(row)) + "</tr>"
    st.html(f'<table class="ss-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>')


def _log_pos(p: float) -> float:
    """Position of a probability on a log scale from 0.01% to 100% (0..1)."""
    import math

    return min(1.0, max(0.02, (math.log10(max(p, 1e-4)) + 4) / 4))


def _meter(p: float, color: str) -> str:
    return (
        f'<div class="ss-meter"><div class="track"><div class="fill" style="width:{_log_pos(p) * 100:.1f}%;background:{color}"></div></div>'
        '<div class="ticks"><span>0.01%</span><span>0.1%</span><span>1%</span><span>10%</span><span>100%</span></div></div>'
    )


def decision_summary(res: dict) -> None:
    st.html(
        '<div class="ss-decision">'
        f'<div><div class="l">Existing system · transaction score</div><div class="v" style="color:{INK_3}">{fmt_pct(res["baseline_risk"])}</div>'
        f'{_meter(res["baseline_risk"], "#a3abb8")}'
        f'<div class="l">Fixed-threshold policy&nbsp; {action_tag(res["baseline_policy_action"])}</div></div>'
        f'<div><div class="l">ScamShield · journey-aware probability</div><div class="v">{fmt_pct(res["enhanced_risk"])}</div>'
        f'{_meter(res["enhanced_risk"], SCAMSHIELD)}'
        '<div class="l">Calibrated on the validation period</div></div>'
        f'<div class="rec"><div class="l">Recommended action (lowest expected cost)</div><div class="big">{action_tag(res["recommended_action"], large=True)}</div>'
        '<div class="l">Cost-aware optimizer with policy guardrails</div></div>'
        "</div>"
        f'<div class="ss-explain">{escape(res["decision_explanation"])}</div>'
    )


def factor_list(factors: Sequence[dict]) -> None:
    """Signed contributions on a shared scale: red raises risk, green lowers it."""
    if not factors:
        return
    scale = max(abs(f["contribution"]) for f in factors) or 1.0
    rows = []
    for f in factors:
        c = f["contribution"]
        w = 50 * abs(c) / scale
        left = 50 if c > 0 else 50 - w
        color = RISK_UP if c > 0 else RISK_DOWN
        val = f["value"]
        val_s = "—" if val is None else (f"{val:.0%}" if f["feature"] == "baseline_logit" else f"{val:,.2f}".rstrip("0").rstrip("."))
        rows.append(
            f'<div class="ss-factor"><div class="n" title="{escape(f["description"])}">{escape(f["description"])}: <b>{val_s}</b></div>'
            f'<div class="track"><div style="position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:#d0d5dd"></div>'
            f'<div class="fill {"pos" if c > 0 else "neg"}" style="left:{left:.1f}%;width:{w:.1f}%;background:{color}"></div></div>'
            f'<div class="c" style="color:{color}">{"+" if c > 0 else "−"}{abs(c):.2f}</div></div>'
        )
    st.html("".join(rows))
