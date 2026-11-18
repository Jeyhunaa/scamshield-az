"""Model training, calibration, stacking leakage controls, artifacts and inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scamshield.data.generator import generate_dataset
from scamshield.data.preprocessing import TRANSACTION_FEATURES, build_modeling_table
from scamshield.features.journey import JOURNEY_FEATURES
from scamshield.models.inference import ScamShieldPredictor
from scamshield.models.train import BASELINE_SCORE, PlattCalibrator, add_baseline_score, fit_logreg, logit, oof_baseline_logit
from scamshield.service import AssessmentService, journey_features_from_input, load_scenarios


@pytest.fixture(scope="module")
def table(small_cfg) -> pd.DataFrame:
    return build_modeling_table(generate_dataset(small_cfg, seed=small_cfg["seed"]), small_cfg)


def test_platt_calibrator_recovers_probabilities():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.001, 0.3, 20000)
    y = rng.binomial(1, true_p)
    distorted = 1 / (1 + np.exp(-(2.0 * logit(true_p) + 1.5)))  # over-confident raw scores
    cal = PlattCalibrator().fit(distorted, y)
    assert cal.a == pytest.approx(0.5, abs=0.08) and cal.b == pytest.approx(-0.75, abs=0.3)
    fixed = cal.transform(distorted)
    assert np.all(np.diff(fixed[np.argsort(distorted)]) >= -1e-12)  # monotone
    assert abs(fixed.mean() - y.mean()) < 0.01


def test_platt_requires_both_classes():
    with pytest.raises(ValueError):
        PlattCalibrator().fit(np.array([0.1, 0.2]), np.array([0, 0]))


def test_modeling_table_has_no_label_derived_features(table):
    for col in [*TRANSACTION_FEATURES, *JOURNEY_FEATURES]:
        assert col in table.columns
    forbidden = {"is_scam", "scam_type", "episode_id"}
    assert not forbidden & set(TRANSACTION_FEATURES + JOURNEY_FEATURES)
    # chronological: every train timestamp precedes every test timestamp
    assert table.loc[table.split == "train", "timestamp"].max() < table.loc[table.split == "val", "timestamp"].min()
    assert table.loc[table.split == "val", "timestamp"].max() < table.loc[table.split == "test", "timestamp"].min()


def test_oof_scores_differ_from_in_sample(table, small_cfg):
    train = table[table.split == "train"]
    model = fit_logreg(train[TRANSACTION_FEATURES], train["is_scam"], small_cfg)
    oof = oof_baseline_logit(train[TRANSACTION_FEATURES], train["is_scam"], small_cfg, seed=0)
    scored = add_baseline_score(table, model, oof)
    np.testing.assert_allclose(scored.loc[scored.split == "train", BASELINE_SCORE], oof)
    in_sample = logit(model.predict_proba(train[TRANSACTION_FEATURES])[:, 1])
    assert not np.allclose(oof, in_sample)
    # non-train rows use the full baseline model
    test = scored[scored.split == "test"]
    np.testing.assert_allclose(test[BASELINE_SCORE], logit(model.predict_proba(test[TRANSACTION_FEATURES])[:, 1]))


def test_artifacts_written(small_artifacts):
    for name in ["models/scamshield_models.joblib", "models/enhanced_catboost.cbm", "models/metadata.json", "models/baseline_policy.json", "metrics.json", "predictions_test.parquet"]:
        assert (small_artifacts / name).exists(), name
    metrics = json.loads((small_artifacts / "metrics.json").read_text())
    for key in ("baseline_lr", "scamshield"):
        m = metrics["models"][key]
        assert 0 <= m["pr_auc"] <= 1 and 0 <= m["roc_auc"] <= 1 and 0 <= m["brier"] <= 1
    assert {"baseline", "scamshield", "savings"} <= set(metrics["end_to_end"])


def test_loaded_predictor_reproduces_saved_predictions(small_artifacts, table):
    predictor = ScamShieldPredictor.load(small_artifacts / "models")
    saved = pd.read_parquet(small_artifacts / "predictions_test.parquet").set_index("txn_id")
    test = table[table.split == "test"].set_index("txn_id").loc[saved.index]
    scores = predictor.score(test.reset_index(), test[JOURNEY_FEATURES].reset_index(drop=True))
    np.testing.assert_allclose(scores["baseline_risk"], saved["baseline_risk"], rtol=1e-6)
    np.testing.assert_allclose(scores["enhanced_risk"], saved["enhanced_risk"], rtol=1e-6)
    assert scores.to_numpy().min() >= 0 and scores.to_numpy().max() <= 1


def test_predictor_missing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError):
        ScamShieldPredictor.load(tmp_path)


def test_explanations_sorted_and_complete(small_artifacts):
    svc = AssessmentService.load(small_artifacts)
    sc = load_scenarios()["safe_account_scam"]
    res = svc.assess(sc["transaction"], journey_features_from_input(events=sc["journey_events"]))
    contribs = [abs(f["contribution"]) for f in res["risk_factors"]]
    assert contribs == sorted(contribs, reverse=True)
    assert res["recommended_action"] in {"allow", "warn", "verify", "hold"}
    assert res["recommended_action"] == min(
        (a for a, c in res["expected_costs"].items() if c["eligible"]), key=lambda a: res["expected_costs"][a]["expected_cost"]
    )


def test_all_demo_scenarios_score(small_artifacts):
    svc = AssessmentService.load(small_artifacts)
    for sc in load_scenarios().values():
        res = svc.assess(sc["transaction"], journey_features_from_input(events=sc["journey_events"]))
        assert 0 <= res["baseline_risk"] <= 1 and 0 <= res["enhanced_risk"] <= 1


def test_journey_input_requires_exactly_one_source():
    with pytest.raises(ValueError):
        journey_features_from_input()
    with pytest.raises(ValueError):
        journey_features_from_input(events=[{"event_type": "login", "t_offset_s": 0}], features={})
    with pytest.raises(ValueError):
        journey_features_from_input(features={"session_duration_s": 1.0})


def test_demo_scenarios_cover_all_actions():
    scenarios = load_scenarios()
    assert {s["intended_action"] for s in scenarios.values()} == {"allow", "warn", "verify", "hold"}
    assert {s["kind"] for s in scenarios.values()} == {"legitimate", "scam"}
    for s in scenarios.values():
        assert {"recipient", "iban", "purpose"} <= set(s["customer_view"])


_PROJECT_MODELS = Path(__file__).resolve().parents[1] / "artifacts"


@pytest.mark.skipif(not (_PROJECT_MODELS / "models" / "scamshield_models.joblib").exists(), reason="run scripts/run_pipeline.py first")
def test_demo_scenarios_produce_intended_action_with_trained_models():
    """The live models + decision engine (not the YAML) must yield each scenario's intended action."""
    svc = AssessmentService.load(_PROJECT_MODELS)
    for key, sc in load_scenarios().items():
        res = svc.assess(sc["transaction"], journey_features_from_input(events=sc["journey_events"]))
        assert res["recommended_action"] == sc["intended_action"], f"{key}: got {res['recommended_action']}"
