"""Customer View: a simulated mobile-banking payment flow showing how ScamShield's
ALLOW / WARN / VERIFY / HOLD decisions appear to a customer.

The decision is computed live (same assessment function as the rest of the dashboard)
when the customer confirms the payment. Nothing about the outcome is scripted; the
customer never sees scores, model features or costs.
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable
from datetime import datetime
from html import escape
from typing import Any

import streamlit as st

from scamshield.service import journey_features_from_input

BANK = "Karvan"
SENDER_ACCOUNT = "Main account •• 4471"
SENDER_BALANCE = 12_480.35
PHONE_MASK = "+994 •• ••• •• 47"
ACTION_LABEL = {"allow": "ALLOW", "warn": "WARN", "verify": "VERIFY", "hold": "HOLD"}
ACTION_COLOR = {"allow": ("#1f7a4d", "#e6f4ec"), "warn": ("#8a5a00", "#fdf3dc"), "verify": ("#a5410f", "#fde9df"), "hold": ("#b3261e", "#fbe4e2")}

FORM_FIELDS = ("recipient", "iban", "amount", "purpose")

PURPOSE_OPTIONS = [
    "Paying for goods or services",
    "Family or friends",
    "Rent, deposit or bills",
    "Investment or savings",
    "Moving my money to a safe account",
    "Something else",
]

# ------------------------------------------------------------------ icons (inline SVG, stroke style)
_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{p}</svg>'
_PATHS = {
    "shield_check": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/>',
    "shield_alert": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="M12 8v4"/><path d="M12 16h.01"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "x": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
    "lock": '<rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
    "user": '<circle cx="12" cy="8" r="4"/><path d="M4 21v-1a6 6 0 0 1 6-6h4a6 6 0 0 1 6 6v1"/>',
    "phone": '<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.8 19.8 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6A19.8 19.8 0 0 1 2.12 4.18 2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.13.96.36 1.9.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.91.34 1.85.57 2.81.7A2 2 0 0 1 22 16.92z"/>',
    "message": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "cart": '<circle cx="9" cy="21" r="1"/><circle cx="20" cy="21" r="1"/><path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6"/>',
    "user_plus": '<circle cx="9" cy="8" r="4"/><path d="M2 21v-1a6 6 0 0 1 6-6h2a6 6 0 0 1 6 6v1"/><path d="M19 8v6"/><path d="M22 11h-6"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "send": '<path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/>',
    "signal": '<path d="M2 20h.01"/><path d="M7 20v-4"/><path d="M12 20v-8"/><path d="M17 20V8"/>',
    "wifi": '<path d="M5 13a10 10 0 0 1 14 0"/><path d="M8.5 16.5a5 5 0 0 1 7 0"/><path d="M12 20h.01"/>',
    "battery": '<rect x="2" y="7" width="16" height="10" rx="2"/><path d="M22 11v2"/><rect x="4" y="9" width="11" height="6" rx="1" fill="#0f1115" stroke="none"/>',
}


def icon(name: str, size: int = 20, color: str = "currentColor") -> str:
    return _SVG.format(s=size, c=color, p=_PATHS[name])


# ------------------------------------------------------------------ styling (scoped to .st-key-cv_* / .cv-*)
CSS = """
<style>
.st-key-cv_phone {
  max-width: 392px; width: 100%; margin: 0 auto; background: #f4f6fb; border: 11px solid #0f1115;
  border-radius: 48px; padding: 6px 18px 18px 18px !important; min-height: 790px;
  box-shadow: 0 30px 60px -20px rgba(13,42,79,.45), 0 0 0 1px #2a2d33, inset 0 0 0 1px #d9dde6;
  gap: .55rem !important; position: relative;
}
.cv-notch {width: 112px; height: 26px; background:#0f1115; border-radius: 0 0 16px 16px; margin: -6px auto 0 auto;}
.cv-status {display:flex; justify-content:space-between; align-items:center; font-size:.78rem; font-weight:600; color:#0f1115; padding: 2px 6px 0 6px;}
.cv-status .ic {display:flex; gap:5px; align-items:center;}
.cv-appbar {display:flex; align-items:center; justify-content:space-between; margin: 4px 0 2px 0;}
.cv-brand {display:flex; align-items:center; gap:8px; font-weight:700; color:#0d2a4f; font-size:.95rem;}
.cv-logo {width:28px; height:28px; border-radius:9px; background:#10305a; display:flex; align-items:center; justify-content:center;}
.cv-avatar {width:30px; height:30px; border-radius:50%; background:#dfe6f3; color:#10305a; font-weight:700; font-size:.75rem; display:flex; align-items:center; justify-content:center;}
.cv-title {font-size:1.32rem; font-weight:750; color:#0b0b0b; margin: 4px 0 0 0; letter-spacing:-.01em;}
.cv-sub {font-size:.86rem; color:#5b6170; margin: 2px 0 2px 0; line-height:1.45;}
.cv-steps {display:flex; gap:5px; margin: 2px 0 4px 0;}
.cv-steps span {flex:1; height:4px; border-radius:4px; background:#dfe3ec;}
.cv-steps span.on {background:#10305a;}
.cv-card {background:#fff; border-radius:18px; padding:14px 16px; box-shadow:0 1px 2px rgba(16,24,40,.06), 0 1px 3px rgba(16,24,40,.05); border:1px solid #edf0f5;}
.cv-from {display:flex; justify-content:space-between; align-items:center;}
.cv-from .lbl {font-size:.74rem; color:#6b7280; text-transform:uppercase; letter-spacing:.05em;}
.cv-from .acc {font-weight:650; color:#111827; font-size:.92rem;}
.cv-from .bal {font-weight:700; color:#0d2a4f; font-size:.95rem; text-align:right;}
.cv-amount {font-size:2.05rem; font-weight:780; color:#0b0b0b; letter-spacing:-.02em; text-align:center; margin: 4px 0 0 0;}
.cv-amount small {font-size:1rem; font-weight:600; color:#6b7280;}
.cv-row {display:flex; justify-content:space-between; gap:12px; padding:9px 0; border-bottom:1px solid #f0f2f6; font-size:.85rem;}
.cv-row:last-child {border-bottom:none;}
.cv-row .k {color:#6b7280; white-space:nowrap;}
.cv-row .v {color:#111827; font-weight:600; text-align:right; word-break:break-word;}
.cv-recipient {display:flex; align-items:center; gap:10px; margin-bottom:6px;}
.cv-recipient .av {width:40px; height:40px; border-radius:50%; background:#e8eefb; color:#10305a; font-weight:700; display:flex; align-items:center; justify-content:center;}
.cv-recipient .nm {font-weight:700; color:#111827;}
.cv-recipient .bk {font-size:.78rem; color:#6b7280;}
.cv-chip {display:inline-flex; align-items:center; gap:5px; font-size:.72rem; font-weight:650; padding:3px 9px; border-radius:999px;}
.cv-hero {display:flex; flex-direction:column; align-items:center; text-align:center; gap:6px; margin-top:6px;}
.cv-hero .ring {width:74px; height:74px; border-radius:50%; display:flex; align-items:center; justify-content:center;}
.cv-tip {display:flex; gap:11px; align-items:flex-start; background:#fff; border-radius:14px; padding:11px 13px; border:1px solid #edf0f5; font-size:.84rem; color:#1f2937; line-height:1.45; margin-bottom:7px;}
.cv-tip .ti {flex:0 0 30px; height:30px; border-radius:9px; display:flex; align-items:center; justify-content:center;}
.cv-timeline {position:relative; margin: 4px 0 0 4px;}
.cv-tl {display:flex; gap:12px; align-items:flex-start; position:relative; padding-bottom:14px; font-size:.84rem;}
.cv-tl:last-child {padding-bottom:0;}
.cv-tl .dot {flex:0 0 22px; height:22px; border-radius:50%; display:flex; align-items:center; justify-content:center; z-index:1;}
.cv-tl:not(:last-child)::before {content:""; position:absolute; left:10px; top:22px; bottom:0; width:2px; background:#e3e7ef;}
.cv-tl .t {font-weight:650; color:#111827;} .cv-tl .d {color:#6b7280; font-size:.78rem;}
.cv-sms {background:rgba(255,255,255,.96); border-radius:16px; padding:10px 12px; box-shadow:0 8px 24px -6px rgba(15,17,21,.25); border:1px solid #e6e9f0; font-size:.8rem; color:#1f2937; animation: cvDrop .5s ease-out;}
.cv-sms .hd {display:flex; justify-content:space-between; font-size:.7rem; color:#6b7280; margin-bottom:3px; font-weight:600;}
.cv-foot {text-align:center; font-size:.68rem; color:#9aa1ae; margin-top:6px;}
.cv-err {background:#fde8e7; color:#b3261e; border-radius:12px; padding:8px 12px; font-size:.82rem; font-weight:600;}
.cv-sim-banner {display:flex; gap:10px; align-items:center; background:#fff; color:#344054; border:1px solid #e4e7ec; border-radius:10px; padding:.6rem .9rem; font-size:.85rem; margin-bottom:.9rem;}
.cv-guide {font-size:.84rem;}
.cv-guide td {padding:4px 6px; border-bottom:1px solid #eee; vertical-align:middle;}
[class*="st-key-cv_screen_"] {animation: cvSlide .38s cubic-bezier(.2,.7,.2,1); gap:.55rem !important;}
@keyframes cvSlide {from {opacity:0; transform: translateX(18px);} to {opacity:1; transform:none;}}
@keyframes cvDrop {from {opacity:0; transform: translateY(-14px);} to {opacity:1; transform:none;}}
/* native widgets inside the phone */
.st-key-cv_phone [data-testid="stWidgetLabel"] p {font-size:.78rem !important; color:#5b6170 !important; font-weight:600;}
.st-key-cv_phone [data-testid="stTextInput"] input, .st-key-cv_phone [data-testid="stNumberInput"] input {
  border-radius: 12px !important; font-size:.95rem; padding: .62rem .8rem; background:#fff;}
.st-key-cv_phone [data-testid="stTextInput"] > div > div, .st-key-cv_phone [data-testid="stNumberInput"] > div > div {border-radius:12px; border-color:#dfe3ec; background:#fff;}
.st-key-cv_phone button[data-testid^="stBaseButton"] {border-radius: 14px; min-height: 48px; font-weight: 650;}
.st-key-cv_phone button[data-testid="stBaseButton-primary"] {background:#10305a; border-color:#10305a; box-shadow: 0 6px 16px -6px rgba(29,79,145,.6);}
.st-key-cv_phone button[data-testid="stBaseButton-primary"]:hover {background:#173f72; border-color:#173f72;}
.st-key-cv_phone button[data-testid="stBaseButton-secondary"] {background:#fff; border-color:#dfe3ec; color:#1f2937;}
.st-key-cv_phone [data-testid="stRadio"] label p, .st-key-cv_phone [data-testid="stCheckbox"] label p {font-size:.85rem;}
.st-key-cv_phone [data-testid="stRadio"] [role="radiogroup"] {gap:.15rem;}
.st-key-cv_phone [data-testid="stMarkdownContainer"] {margin-bottom: 0 !important;}  /* Streamlit uses -1rem for HTML blocks */
.st-key-cv_phone button[data-testid="stBaseButton-primary"]:disabled {background:#c7d2e3; border-color:#c7d2e3; color:#ffffff; box-shadow:none; cursor:not-allowed;}
.st-key-cv_phone button[data-testid="stBaseButton-secondary"]:disabled {color:#9aa1ae; background:#f7f8fb;}
</style>
"""


# ------------------------------------------------------------------ helpers
def _ss() -> Any:
    return st.session_state


def _reset(scenario_key: str, sc: dict[str, Any]) -> None:
    for k in [k for k in st.session_state if k.startswith("cv_")]:
        del st.session_state[k]
    cv = sc.get("customer_view", {})
    _ss()["cv_scenario"] = scenario_key
    _ss()["cv_step"] = "details"
    _ss()["cv_recipient"] = cv.get("recipient", "")
    _ss()["cv_iban"] = cv.get("iban", "")
    _ss()["cv_purpose"] = cv.get("purpose", "")
    _ss()["cv_amount"] = float(sc["transaction"]["amount"])


def _go(step: str) -> None:
    _ss()["cv_step"] = step


def _money(x: float) -> str:
    return f"{x:,.2f}"


def _initials(name: str) -> str:
    parts = [p for p in re.split(r"\s+", name.strip()) if p and p[0].isalpha()]
    return ("".join(p[0] for p in parts[:2]) or "?").upper()


def _iban_pretty(iban: str) -> str:
    s = re.sub(r"\s+", "", iban).upper()
    return " ".join(s[i : i + 4] for i in range(0, len(s), 4))


def _html(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def _hero(icon_name: str, fg: str, bg: str, title: str, sub: str) -> None:
    _html(
        f'<div class="cv-hero"><div class="ring" style="background:{bg}">{icon(icon_name, 34, fg)}</div>'
        f'<div class="cv-title" style="text-align:center">{title}</div><div class="cv-sub">{sub}</div></div>'
    )


def _tip(icon_name: str, fg: str, bg: str, text: str) -> str:
    return f'<div class="cv-tip"><div class="ti" style="background:{bg}">{icon(icon_name, 17, fg)}</div><div>{text}</div></div>'


def _progress(n: int) -> None:
    _html('<div class="cv-steps">' + "".join(f'<span class="{"on" if i < n else ""}"></span>' for i in range(3)) + "</div>")


def _summary_card(show_amount: bool = True) -> None:
    ss = _ss()
    sc_cv = ss.get("cv_meta", {})
    amount = ss["cv_amount"]
    rows = [
        ("IBAN", _iban_pretty(ss["cv_iban"])),
        ("Purpose", escape(ss["cv_purpose"] or "—")),
        ("From", SENDER_ACCOUNT),
        ("Fee", "0.00 AZN"),
    ]
    amt = f'<div class="cv-amount">{_money(amount)} <small>AZN</small></div>' if show_amount else ""
    _html(
        f'<div class="cv-card">{amt}<div class="cv-recipient" style="margin-top:10px"><div class="av">{_initials(ss["cv_recipient"])}</div>'
        f'<div><div class="nm">{escape(ss["cv_recipient"])}</div><div class="bk">{escape(sc_cv.get("recipient_bank", "Recipient bank"))}</div></div></div>'
        + "".join(f'<div class="cv-row"><span class="k">{k}</span><span class="v">{v}</span></div>' for k, v in rows)
        + "</div>"
    )


def _reference() -> str:
    if "cv_ref" not in _ss():
        _ss()["cv_ref"] = f"DB{datetime.now():%y%m%d}-{random.randint(100000, 999999)}"
    return _ss()["cv_ref"]


# ------------------------------------------------------------------ screens
def _screen_details(sc: dict[str, Any]) -> None:
    ss = _ss()
    _progress(1)
    _html('<div class="cv-title">New transfer</div><div class="cv-sub">Send money to any bank account in Azerbaijan.</div>')
    _html(
        f'<div class="cv-card cv-from"><div><div class="lbl">From</div><div class="acc">{SENDER_ACCOUNT}</div></div>'
        f'<div><div class="lbl" style="text-align:right">Available</div><div class="bal">{_money(SENDER_BALANCE)} AZN</div></div></div>'
    )
    new = bool(sc["transaction"]["is_new_beneficiary"])
    chip = (
        f'<span class="cv-chip" style="background:#eef2fb;color:#10305a">{icon("user_plus", 13, "#10305a")} New recipient</span>'
        if new
        else f'<span class="cv-chip" style="background:#e6f4ec;color:#1f7a4d">{icon("check", 13, "#1f7a4d")} Saved recipient</span>'
    )
    _html(f'<div style="display:flex;justify-content:space-between;align-items:center;margin-top:4px"><span class="cv-sub" style="font-weight:650;color:#111827">To</span>{chip}</div>')
    # Streamlit discards widget state once a widget stops rendering, so the form uses
    # cv_in_* widget keys and copies values into persistent cv_* keys on Continue.
    for f in FORM_FIELDS:
        ss.setdefault(f"cv_in_{f}", ss[f"cv_{f}"])
    st.text_input("Recipient name", key="cv_in_recipient")
    st.text_input("IBAN", key="cv_in_iban", help="All IBANs here are fictional.")
    st.number_input("Amount (AZN)", min_value=1.0, max_value=SENDER_BALANCE, step=10.0, format="%.2f", key="cv_in_amount")
    st.text_input("Payment purpose", key="cv_in_purpose", max_chars=60)

    err = None
    iban = re.sub(r"\s+", "", ss["cv_in_iban"]).upper()
    if st.button("Continue", type="primary", width="stretch", key="cv_btn_continue"):
        if not ss["cv_in_recipient"].strip():
            err = "Please enter the recipient's name."
        elif not re.fullmatch(r"AZ\d{2}[A-Z0-9]{24}", iban):
            err = "Please enter a valid Azerbaijani IBAN (AZ + 26 characters)."
        else:
            for f in FORM_FIELDS:
                ss[f"cv_{f}"] = ss[f"cv_in_{f}"]
            ss["cv_iban"] = iban
            _go("confirm")
            st.rerun()
    if err:
        _html(f'<div class="cv-err">{err}</div>')


def _screen_confirm(assess: Callable[[float], dict[str, Any]]) -> None:
    ss = _ss()
    _progress(2)
    _html('<div class="cv-title">Review transfer</div><div class="cv-sub">Check the details carefully before you send.</div>')
    _summary_card()
    _html(f'<div class="cv-sub" style="display:flex;gap:6px;align-items:center;margin-top:4px">{icon("lock", 15, "#10305a")} Confirm with your 4-digit PIN</div>')
    pin = st.text_input("PIN", type="password", max_chars=4, key="cv_pin", placeholder="••••", label_visibility="collapsed")
    _html('<div class="cv-foot" style="margin-top:-4px;text-align:left">Any 4 digits · simulated sign-in</div>')
    err = None
    if st.button(f"Send {_money(ss['cv_amount'])} AZN", type="primary", width="stretch", key="cv_btn_send"):
        if not re.fullmatch(r"\d{4}", pin or ""):
            err = "Enter your 4-digit PIN to confirm."
        else:
            with st.spinner("Securing your payment…"):
                started = time.time()
                result = assess(float(ss["cv_amount"]))
                time.sleep(max(0.0, 0.9 - (time.time() - started)))  # a calm, consistent pause
            ss["cv_result"] = result
            action = result["recommended_action"]
            ss["cv_action"] = action
            if action == "allow":
                ss["cv_outcome"] = "sent"
                _go("done")
            else:
                _go(action)
            st.rerun()
    if err:
        _html(f'<div class="cv-err">{err}</div>')
    if st.button("Back", width="stretch", key="cv_btn_back"):
        _go("details")
        st.rerun()


def _warning_tips(sc: dict[str, Any], journey: dict[str, float]) -> list[str]:
    """Plain-language prompts chosen from what is observable about this payment."""
    ss = _ss()
    tips: list[str] = []
    purpose = (ss.get("cv_purpose") or "").lower()
    if journey.get("call_active"):
        tips.append(_tip("phone", "#8a5a00", "#fdf3dc", "<b>Is someone on the phone with you?</b> We will never call and ask you to move money, even to a “safe account”."))
    if any(w in purpose for w in ("order", "purchase", "buy", "marketplace", "phone", "iphone", "ticket")):
        tips.append(_tip("cart", "#8a5a00", "#fdf3dc", "<b>Buying something online?</b> Sellers who ask for a bank transfer instead of the platform’s secure checkout are a common scam. You may not get your money back."))
    if journey.get("payee_pasted") or journey.get("n_app_switches", 0) > 0:
        tips.append(_tip("message", "#8a5a00", "#fdf3dc", "<b>Did these details come in a chat or message?</b> Scammers send convincing account details. Check them another way before paying."))
    if sc["transaction"]["is_new_beneficiary"]:
        tips.append(_tip("user_plus", "#8a5a00", "#fdf3dc", f"<b>You haven’t paid {escape(ss['cv_recipient'])} before.</b> Make sure you know who they are."))
    tips.append(_tip("info", "#8a5a00", "#fdf3dc", "<b>If in doubt, stop.</b> Once sent, a transfer can be very hard to recover."))
    return tips[:3]


def _screen_warn(sc: dict[str, Any], journey: dict[str, float]) -> None:
    _hero("alert", "#8a5a00", "#fdf3dc", "Before you send this", "This payment has some signs we often see in scams. Please take a moment to check.")
    _html(f'<div class="cv-card" style="padding:10px 14px"><div class="cv-row"><span class="k">To</span><span class="v">{escape(_ss()["cv_recipient"])}</span></div>'
          f'<div class="cv-row"><span class="k">Amount</span><span class="v">{_money(_ss()["cv_amount"])} AZN</span></div></div>')
    _html("".join(_warning_tips(sc, journey)))
    ok = st.checkbox("I’ve checked and I trust this recipient", key="cv_warn_ack")
    if st.button("Cancel payment", type="primary", width="stretch", key="cv_btn_warn_cancel"):
        _ss()["cv_outcome"] = "cancelled"
        _go("done")
        st.rerun()
    if st.button("Send anyway", width="stretch", disabled=not ok, key="cv_btn_warn_send"):
        _ss()["cv_outcome"] = "sent_after_warning"
        _go("done")
        st.rerun()


def _screen_verify() -> None:
    ss = _ss()
    sub = ss.get("cv_verify_stage", "questions")
    if sub == "questions":
        _hero("shield_check", "#10305a", "#e8eefb", "Quick security check", "To keep your money safe, please answer three short questions. It takes less than a minute.")
        st.radio("What is this payment for?", PURPOSE_OPTIONS, index=None, key="cv_q_purpose")
        st.radio("Has anyone contacted you by phone, SMS or messenger and asked you to make this payment?", ["No", "Yes"], index=None, horizontal=True, key="cv_q_contacted")
        st.radio("Has anyone told you to ignore bank warnings, or to keep this payment secret?", ["No", "Yes"], index=None, horizontal=True, key="cv_q_secret")
        answered = all(ss.get(k) is not None for k in ("cv_q_purpose", "cv_q_contacted", "cv_q_secret"))
        if st.button("Continue", type="primary", width="stretch", disabled=not answered, key="cv_btn_q_next"):
            ss["cv_contacted"] = ss["cv_q_contacted"] == "Yes"  # persisted for the code step
            if ss["cv_q_secret"] == "Yes" or ss["cv_q_purpose"] == "Moving my money to a safe account":
                ss["cv_outcome"] = "stopped"
                _go("done")
            else:
                ss["cv_verify_stage"] = "code"
                ss["cv_otp"] = f"{random.randint(0, 999999):06d}"
            st.rerun()
        if st.button("Cancel payment", width="stretch", key="cv_btn_q_cancel"):
            ss["cv_outcome"] = "cancelled"
            _go("done")
            st.rerun()
        return

    # one-time code step (with a simulated SMS notification)
    _html(
        f'<div class="cv-sms"><div class="hd"><span>{icon("message", 12, "#6b7280")}&nbsp; MESSAGES · {BANK.upper()}</span><span>now</span></div>'
        f'<b>{ss["cv_otp"]}</b> is your code to confirm a transfer of {_money(ss["cv_amount"])} AZN to {escape(ss["cv_recipient"])}. '
        "Never share this code with anyone, including bank staff.</div>"
    )
    _hero("lock", "#10305a", "#e8eefb", "Enter your code", f"We sent a 6-digit code to {PHONE_MASK}.")
    contacted = ss.get("cv_contacted", False)
    confirmed = True
    if contacted:
        _html(_tip("phone", "#a5410f", "#fde9df", "<b>You said someone asked you to make this payment.</b> Before continuing, call them on a number you already have — not one from the message — and confirm it’s really them."))
        confirmed = st.checkbox("I’ve called them on a number I already know and confirmed the request", key="cv_called_back")
    code = st.text_input("Security code", max_chars=6, key="cv_code", placeholder="6-digit code", label_visibility="collapsed")
    err = None
    if st.button("Verify and send", type="primary", width="stretch", disabled=not confirmed, key="cv_btn_verify"):
        if (code or "").strip() == ss["cv_otp"]:
            ss["cv_outcome"] = "sent_after_verification"
            _go("done")
            st.rerun()
        err = "That code doesn’t match. Please check the message and try again."
    if err:
        _html(f'<div class="cv-err">{err}</div>')
    if st.button("Cancel payment", width="stretch", key="cv_btn_code_cancel"):
        ss["cv_outcome"] = "cancelled"
        _go("done")
        st.rerun()


def _screen_hold() -> None:
    ss = _ss()
    now = datetime.now()
    _hero("shield_check", "#10305a", "#e8eefb", "Your payment is on hold", "For your security, we’ve paused this payment for a quick check. <b>Your money hasn’t left your account.</b>")
    _html(
        f'<div class="cv-card" style="padding:10px 14px"><div class="cv-row"><span class="k">To</span><span class="v">{escape(ss["cv_recipient"])}</span></div>'
        f'<div class="cv-row"><span class="k">Amount</span><span class="v">{_money(ss["cv_amount"])} AZN</span></div>'
        f'<div class="cv-row"><span class="k">Reference</span><span class="v">{_reference()}</span></div></div>'
    )
    _html(
        '<div class="cv-card"><div class="cv-sub" style="font-weight:700;color:#111827;margin-bottom:8px">What happens next</div><div class="cv-timeline">'
        f'<div class="cv-tl"><div class="dot" style="background:#e6f4ec">{icon("check", 13, "#1f7a4d")}</div><div><div class="t">Payment requested</div><div class="d">Today, {now:%H:%M}</div></div></div>'
        f'<div class="cv-tl"><div class="dot" style="background:#e8eefb">{icon("clock", 13, "#10305a")}</div><div><div class="t">Security review</div><div class="d">Our team will call you within 30 minutes</div></div></div>'
        f'<div class="cv-tl"><div class="dot" style="background:#f0f2f6">{icon("send", 12, "#9aa1ae")}</div><div><div class="t" style="color:#6b7280">Released or cancelled</div><div class="d">We’ll notify you as soon as it’s done</div></div></div>'
        "</div></div>"
    )
    _html(_tip("phone", "#10305a", "#e8eefb", "We’ll only call from the number on the back of your card. <b>We will never ask for your PIN, passwords or codes.</b>"))
    if st.button("OK, I’ll wait for your call", type="primary", width="stretch", key="cv_btn_hold_ok"):
        ss["cv_outcome"] = "on_hold"
        _go("done")
        st.rerun()
    if st.button("Cancel this payment", width="stretch", key="cv_btn_hold_cancel"):
        ss["cv_outcome"] = "hold_cancelled"
        _go("done")
        st.rerun()


def _screen_done() -> None:
    ss = _ss()
    outcome = ss.get("cv_outcome", "sent")
    amount, who = _money(ss["cv_amount"]), escape(ss["cv_recipient"])
    now = datetime.now()
    if outcome.startswith("sent"):
        extra = {"sent_after_warning": " after you confirmed the warning", "sent_after_verification": " after a successful security check"}.get(outcome, "")
        _hero("check", "#1f7a4d", "#e6f4ec", "Money sent", f"{amount} AZN is on its way to {who}{extra}.")
        _html(
            f'<div class="cv-card"><div class="cv-row"><span class="k">Status</span><span class="v" style="color:#1f7a4d">Completed</span></div>'
            f'<div class="cv-row"><span class="k">Date</span><span class="v">{now:%d %b %Y, %H:%M}</span></div>'
            f'<div class="cv-row"><span class="k">IBAN</span><span class="v">{_iban_pretty(ss["cv_iban"])}</span></div>'
            f'<div class="cv-row"><span class="k">Reference</span><span class="v">{_reference()}</span></div></div>'
        )
    elif outcome == "on_hold":
        _hero("clock", "#10305a", "#e8eefb", "Payment under review", f"Your transfer of {amount} AZN to {who} is safely on hold. We’ll call you shortly.")
        _html(
            f'<div class="cv-card"><div class="cv-row"><span class="k">Status</span><span class="v"><span class="cv-chip" style="background:#e8eefb;color:#10305a">On hold · in review</span></span></div>'
            f'<div class="cv-row"><span class="k">Reference</span><span class="v">{_reference()}</span></div>'
            f'<div class="cv-row"><span class="k">Balance</span><span class="v">{_money(SENDER_BALANCE)} AZN (unchanged)</span></div></div>'
        )
        _html(_tip("info", "#10305a", "#e8eefb", "If someone is pressuring you to make this payment, hang up. Genuine organisations will understand the wait."))
    elif outcome == "stopped":
        _hero("shield_alert", "#b3261e", "#fbe4e2", "We’ve stopped this payment", "Your answers suggest someone may be trying to trick you. <b>No money has left your account.</b>")
        _html(
            _tip("phone", "#b3261e", "#fbe4e2", "<b>Banks never ask you to move money to a “safe account”</b> or to keep a payment secret.")
            + _tip("x", "#b3261e", "#fbe4e2", "If you’re on a call about this payment, hang up now.")
            + _tip("info", "#b3261e", "#fbe4e2", "Call us on the number on the back of your card — we’re here 24/7.")
        )
    else:  # cancelled / hold_cancelled
        _hero("x", "#4b5563", "#eef0f4", "Payment cancelled", f"Nothing was sent to {who}. <b>No money has left your account.</b>")
        _html(_tip("info", "#10305a", "#e8eefb", "Good call — checking first is always the right move. You can make the payment later if you’re sure."))
    if st.button("Make another payment", type="primary", width="stretch", key="cv_btn_restart"):
        ss["cv_restart"] = True
        st.rerun()


# ------------------------------------------------------------------ entry point
def render_customer_view(
    scenario_key: str,
    scenarios: dict[str, dict[str, Any]],
    assess_fn: Callable[[dict[str, Any], dict[str, float]], dict[str, Any]],
) -> None:
    """Render the Customer View tab. `assess_fn(tx, journey)` is the dashboard's live assessment."""
    sc = scenarios[scenario_key]
    if _ss().get("cv_scenario") != scenario_key or _ss().pop("cv_restart", False):
        _reset(scenario_key, sc)
    _ss()["cv_meta"] = sc.get("customer_view", {})
    journey = journey_features_from_input(events=sc["journey_events"])

    def assess(amount: float) -> dict[str, Any]:
        return assess_fn({**sc["transaction"], "amount": amount}, journey)

    _html(CSS)
    _html(
        f'<div class="cv-sim-banner">{icon("info", 18, "#344054")}<div><b>Simulated banking experience.</b> '
        "Fictional bank, people and accounts — no real payment is made. ScamShield's decision is computed live by the trained models when the customer confirms.</div></div>"
    )
    phone_col, notes_col = st.columns([6, 5], gap="large")

    with phone_col:
        with st.container(key="cv_phone"):
            _html(
                '<div class="cv-notch"></div>'
                f'<div class="cv-status"><span>{datetime.now():%H:%M}</span><span class="ic">{icon("signal", 14, "#0f1115")}{icon("wifi", 14, "#0f1115")}{icon("battery", 18, "#0f1115")}</span></div>'
                f'<div class="cv-appbar"><div class="cv-brand"><div class="cv-logo">{icon("shield_check", 16, "#ffffff")}</div>{BANK}</div><div class="cv-avatar">AH</div></div>'
            )
            step = _ss()["cv_step"]
            with st.container(key=f"cv_screen_{step}"):
                if step == "details":
                    _screen_details(sc)
                elif step == "confirm":
                    _screen_confirm(assess)
                elif step == "warn":
                    _screen_warn(sc, journey)
                elif step == "verify":
                    _screen_verify()
                elif step == "hold":
                    _screen_hold()
                else:
                    _screen_done()
            _html(f'<div class="cv-foot">{icon("lock", 10, "#9aa1ae")} Protected by ScamShield · simulated app</div>')

    with notes_col:
        st.markdown("##### Presenter notes")
        st.markdown(f"**{sc['title']}**")
        st.caption(sc["story"])
        action = _ss().get("cv_action")
        if action:
            fg, bg = ACTION_COLOR[action]
            _html(
                f'<div style="margin:.4rem 0 .6rem 0">ScamShield decision for this payment: '
                f'<span class="badge" style="color:{fg};background:{bg};border:1px solid {fg}33;padding:.2rem .65rem;border-radius:999px;font-weight:700">{ACTION_LABEL[action]}</span></div>'
            )
            st.caption("Scores, explanations and the cost breakdown are deliberately hidden from the customer — see the Transaction assessment tab.")
        else:
            st.caption("The decision is computed live when the customer confirms the payment.")
        st.caption(
            "In-session behaviour (calls, app switching, hesitation) comes from this scenario's recorded journey; "
            "editing the amount on the phone changes the live risk assessment."
        )
        if st.button("↺ Restart", key="cv_btn_presenter_restart"):
            _ss()["cv_restart"] = True
            st.rerun()

        st.markdown("##### Scenario guide")
        rows = "".join(
            f'<tr><td><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{"#067647" if s["kind"] == "legitimate" else "#b42318"}"></span></td><td>{escape(s["title"])}</td>'
            f'<td><span style="color:{ACTION_COLOR[s.get("intended_action", "allow")][0]};font-weight:700;font-size:.75rem">'
            f'{ACTION_LABEL.get(s.get("intended_action", ""), "—")}</span></td></tr>'
            for s in scenarios.values()
        )
        _html(f'<table class="cv-guide">{rows}</table>')
        st.caption("Outcome each scenario is designed to illustrate (verified by automated tests). Pick a scenario in the sidebar.")
