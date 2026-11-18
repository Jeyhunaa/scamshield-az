"""Assessment service: model scoring + cost-aware decision for one transaction.

Shared by the FastAPI backend and the Streamlit dashboard so both always produce
identical results from the same trained artifacts and decision logic.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scamshield import __version__
from scamshield.config import ARTIFACTS_DIR
from scamshield.decision.optimizer import ACTIONS, InterventionConfig, decide, fixed_threshold_actions, load_intervention_config
from scamshield.features.journey import JOURNEY_FEATURES, extract_journey_features
from scamshield.models.inference import ScamShieldPredictor

DISCLAIMER = (
    "Models trained on synthetic data; costs and intervention effectiveness are configurable "
    "assumptions. Not validated on real banking data."
)


def journey_features_from_input(
    events: Sequence[Mapping[str, Any]] | None = None, features: Mapping[str, Any] | None = None
) -> dict[str, float]:
    """Accept either raw journey events (preferred) or pre-computed journey features."""
    if (events is None) == (features is None):
        raise ValueError("provide exactly one of journey events or journey features")
    if events is not None:
        if len(events) == 0:
            raise ValueError("journey events must not be empty")
        return extract_journey_features(events)
    missing = [f for f in JOURNEY_FEATURES if f not in features]
    if missing:
        raise ValueError(f"missing journey features: {missing}")
    return {f: (float("nan") if features[f] is None else float(features[f])) for f in JOURNEY_FEATURES}


@dataclass
class AssessmentService:
    predictor: ScamShieldPredictor
    intervention: InterventionConfig
    baseline_thresholds: tuple[float, float, float]

    @classmethod
    def load(cls, artifacts_dir: str | Path | None = None, intervention_path: str | Path | None = None) -> AssessmentService:
        models_dir = Path(artifacts_dir or ARTIFACTS_DIR) / "models"
        predictor = ScamShieldPredictor.load(models_dir)
        policy_path = models_dir / "baseline_policy.json"
        if not policy_path.exists():
            raise FileNotFoundError(f"Baseline policy not found at {policy_path}. Run the pipeline first.")
        th = json.loads(policy_path.read_text())["thresholds"]
        thresholds = tuple(float("inf") if v is None else float(v) for v in (th["warn"], th["verify"], th["hold"]))
        return cls(predictor, load_intervention_config(intervention_path), thresholds)  # type: ignore[arg-type]

    def assess(
        self,
        context: Mapping[str, Any],
        journey: Mapping[str, float],
        hold_allowed: bool = True,
        intervention: InterventionConfig | None = None,
    ) -> dict[str, Any]:
        cfg = intervention or self.intervention
        ctx_df = pd.DataFrame([dict(context)])
        jf_df = pd.DataFrame([dict(journey)])
        scores = self.predictor.score(ctx_df, jf_df).iloc[0]
        p_base, p_enh = float(scores["baseline_risk"]), float(scores["enhanced_risk"])
        amount = float(context["amount"])

        decision = decide(p_enh, amount, cfg, hold_allowed)
        base_idx = int(fixed_threshold_actions([p_base], [amount], self.baseline_thresholds, cfg, hold_allowed)[0])
        explanation = self.predictor.explain(ctx_df, jf_df)
        return {
            "baseline_risk": round(p_base, 6),
            "enhanced_risk": round(p_enh, 6),
            "recommended_action": decision.action,
            "baseline_policy_action": ACTIONS[base_idx],
            "decision_explanation": decision.explanation,
            "expected_costs": decision.expected_costs,
            "risk_factors": explanation["risk_factors"],
            "baseline_drivers": explanation["baseline_drivers"],
            "journey_features": {k: (None if np.isnan(v) else round(float(v), 3)) for k, v in journey.items()},
            "model_version": f"scamshield-{__version__}",
            "disclaimer": DISCLAIMER,
        }


def load_scenarios(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Demo scenarios (inputs only) with journey events expanded to dicts."""
    from scamshield.config import CONFIG_DIR, load_yaml

    raw = load_yaml(path or CONFIG_DIR / "scenarios.yaml")
    for sc in raw.values():
        sc["journey_events"] = [{"event_type": e, "t_offset_s": float(t)} for e, t in sc["journey_events"]]
    return raw
