"""Smoke test: the Streamlit dashboard renders every tab without errors for every scenario."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / "artifacts" / "models" / "scamshield_models.joblib").exists(), reason="run scripts/run_pipeline.py first"
)


def test_dashboard_renders_all_scenarios():
    from streamlit.testing.v1 import AppTest

    from scamshield.service import load_scenarios

    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    for key in load_scenarios():
        at.sidebar.radio[0].set_value(key).run()
        assert not at.exception, f"{key}: {at.exception}"


def test_customer_view_flow_produces_live_decision():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=180).run()
    at.sidebar.radio[0].set_value("marketplace_seller").run()
    at.button(key="cv_btn_continue").click().run()
    at.text_input(key="cv_pin").input("1234").run()
    at.button(key="cv_btn_send").click().run()
    assert not at.exception, at.exception
    assert at.session_state["cv_action"] == "warn"
