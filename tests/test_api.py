"""FastAPI integration tests against models trained in a temporary directory."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scamshield.service import load_scenarios


@pytest.fixture(scope="module")
def client(small_artifacts, monkeypatch_module):
    monkeypatch_module.setenv("SCAMSHIELD_ARTIFACTS_DIR", str(small_artifacts))
    from app.api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture
def scenario():
    sc = load_scenarios()["safe_account_scam"]
    return {"transaction": sc["transaction"], "journey_events": sc["journey_events"]}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["models_loaded"] is True
    assert set(body["baseline_thresholds"]) == {"warn", "verify", "hold"}


def test_assess_with_events(client, scenario):
    r = client.post("/assess", json=scenario)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["baseline_risk"] <= 1 and 0 <= body["enhanced_risk"] <= 1
    assert body["recommended_action"] in {"allow", "warn", "verify", "hold"}
    assert set(body["expected_costs"]) == {"allow", "warn", "verify", "hold"}
    assert body["risk_factors"] and body["disclaimer"]


def test_assess_with_features_matches_events(client, scenario):
    from scamshield.features.journey import extract_journey_features

    feats = extract_journey_features(scenario["journey_events"])
    feats = {k: (bool(v) if k in ("payee_pasted", "call_active") else int(v) if k.startswith("n_") else v) for k, v in feats.items()}
    r1 = client.post("/assess", json=scenario).json()
    r2 = client.post("/assess", json={"transaction": scenario["transaction"], "journey_features": feats}).json()
    assert r1["enhanced_risk"] == pytest.approx(r2["enhanced_risk"])
    assert r1["recommended_action"] == r2["recommended_action"]


def test_hold_not_allowed_respected(client, scenario):
    body = client.post("/assess", json={**scenario, "hold_allowed": False}).json()
    assert body["recommended_action"] != "hold"
    assert body["expected_costs"]["hold"]["eligible"] is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.pop("journey_events"),                                         # no journey
        lambda p: p.update(journey_features={"session_duration_s": 10, "n_payment_screen_visits": 1}),  # both
        lambda p: p["transaction"].update(amount=-5),                              # invalid amount
        lambda p: p["transaction"].update(hour=25),                                # invalid hour
        lambda p: p["transaction"].update(txn_last_24h=5, txn_last_7d=1),          # inconsistent history
        lambda p: p["transaction"].update(prior_avg_amount=None),                  # history missing
        lambda p: p["journey_events"].append({"event_type": "teleport", "t_offset_s": 3}),  # unknown event
        lambda p: p["transaction"].update(unexpected_field=1),                     # extra field
    ],
)
def test_invalid_requests_rejected(client, scenario, mutate):
    mutate(scenario)
    assert client.post("/assess", json=scenario).status_code == 422


def test_degraded_when_models_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("SCAMSHIELD_ARTIFACTS_DIR", str(tmp_path))
    from app.api import main

    with TestClient(main.app) as c:
        h = c.get("/health").json()
        assert h["status"] == "degraded" and h["models_loaded"] is False
        sc = load_scenarios()["routine_rent"]
        assert c.post("/assess", json={"transaction": sc["transaction"], "journey_events": sc["journey_events"]}).status_code == 503
    monkeypatch.undo()
    main.load_service()  # restore the trained models for any later test


def test_scenarios_endpoint_has_inputs_only(client):
    r = client.get("/scenarios")
    assert r.status_code == 200
    body = r.json()
    assert {s["intended_action"] for s in body} == {"allow", "warn", "verify", "hold"}
    first = body[0]
    assert {"transaction", "journey_events", "customer_view"} <= set(first)
    assert "enhanced_risk" not in first and "recommended_action" not in first


def test_scenario_payload_round_trips_through_assess(client):
    sc = client.get("/scenarios").json()[0]
    r = client.post("/assess", json={"transaction": sc["transaction"], "journey_events": sc["journey_events"]})
    assert r.status_code == 200
