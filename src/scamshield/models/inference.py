"""Load trained artifacts and score transactions, with per-prediction explanations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import Pool

from scamshield.data.preprocessing import derive_transaction_features
from scamshield.features.journey import FEATURE_DESCRIPTIONS, JOURNEY_FEATURES
from scamshield.models.train import BASELINE_SCORE, FittedModel, logit

TRANSACTION_DESCRIPTIONS: dict[str, str] = {
    "log_amount": "Transfer amount (AZN)",
    "log_amount_to_avg": "Amount vs. customer's average transfer (×)",
    "amount_to_max": "Amount vs. customer's largest previous transfer (×)",
    "is_new_beneficiary": "New beneficiary",
    "has_history": "Customer has transfer history",
    "log_prior_txn_count": "Number of previous transfers",
    "txn_last_24h": "Transfers in the previous 24h",
    "txn_last_7d": "Transfers in the previous 7 days",
    "log_amount_last_24h": "Amount sent in the previous 24h (AZN)",
    "log_hours_since_last": "Hours since previous transfer",
    "is_night": "Night-time transfer (23:00–05:59)",
    "log_n_distinct_beneficiaries": "Number of distinct past beneficiaries",
    "log_tenure_days": "Customer tenure (days)",
}


class ScamShieldPredictor:
    """Wraps the baseline (existing-system) model and the enhanced ScamShield model."""

    def __init__(self, baseline: FittedModel, enhanced: FittedModel, metadata: dict[str, Any] | None = None):
        self.baseline = baseline
        self.enhanced = enhanced
        self.metadata = metadata or {}

    @classmethod
    def load(cls, models_dir: str | Path) -> ScamShieldPredictor:
        models_dir = Path(models_dir)
        path = models_dir / "scamshield_models.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Model bundle not found at {path}. Run `python scripts/run_pipeline.py` first.")
        bundle = joblib.load(path)
        meta_path = models_dir / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return cls(bundle["baseline"], bundle["enhanced"], meta)

    def _inputs(self, context: pd.DataFrame, journey: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        tx = derive_transaction_features(context.reset_index(drop=True))
        jf = journey.reset_index(drop=True)
        missing = [c for c in JOURNEY_FEATURES if c not in jf.columns]
        if missing:
            raise ValueError(f"missing journey features: {missing}")
        enh = jf[JOURNEY_FEATURES].astype(float).copy()
        enh[BASELINE_SCORE] = logit(self.baseline.predict_raw(tx))
        return tx, enh

    def score(self, context: pd.DataFrame, journey: pd.DataFrame) -> pd.DataFrame:
        """Calibrated baseline and enhanced scam probabilities for each row."""
        if len(context) != len(journey):
            raise ValueError("context and journey must have the same number of rows")
        tx, enh = self._inputs(context, journey)
        return pd.DataFrame(
            {
                "baseline_risk": self.baseline.calibrator.transform(self.baseline.predict_raw(tx)),
                "enhanced_risk": self.enhanced.predict(enh),
            }
        )

    def explain(self, context: pd.DataFrame, journey: pd.DataFrame, top_k: int = 5) -> dict[str, list[dict[str, Any]]]:
        """Explain the first row.

        * `risk_factors`: CatBoost SHAP contributions (log-odds) of the enhanced model;
          the existing fraud score appears as one factor.
        * `baseline_drivers`: linear contributions coef * standardized value of the
          baseline logistic regression (what drives the existing score).
        """
        tx, enh = self._inputs(context.iloc[:1], journey.iloc[:1])
        X = enh[self.enhanced.features]
        shap = self.enhanced.model.get_feature_importance(Pool(X), type="ShapValues")[0, :-1]
        factors = []
        for name, contrib in zip(self.enhanced.features, shap):
            factors.append(
                {
                    "feature": name,
                    "description": "Existing transaction fraud score (uncalibrated)" if name == BASELINE_SCORE else FEATURE_DESCRIPTIONS.get(name, name),
                    "value": float(1 / (1 + np.exp(-X[name].iloc[0]))) if name == BASELINE_SCORE else _clean(X[name].iloc[0]),
                    "contribution": float(contrib),
                    "direction": "increases risk" if contrib > 0 else "decreases risk",
                    "source": "transaction" if name == BASELINE_SCORE else "journey",
                }
            )
        factors.sort(key=lambda f: abs(f["contribution"]), reverse=True)

        pipe = self.baseline.model
        scaler, lr = pipe.steps[0][1], pipe.steps[-1][1]
        z = (tx[self.baseline.features].to_numpy(dtype=float)[0] - scaler.mean_) / scaler.scale_
        contribs = lr.coef_[0] * z
        drivers = [
            {
                "feature": f,
                "description": TRANSACTION_DESCRIPTIONS.get(f, f),
                "value": _readable(f, tx[f].iloc[0]),
                "contribution": float(c),
                "direction": "increases risk" if c > 0 else "decreases risk",
                "source": "transaction",
            }
            for f, c in zip(self.baseline.features, contribs)
        ]
        drivers.sort(key=lambda d: abs(d["contribution"]), reverse=True)
        return {"risk_factors": factors[:top_k], "baseline_drivers": drivers[:3]}


def _clean(v: Any) -> float | None:
    v = float(v)
    return None if np.isnan(v) else round(v, 4)


def _readable(feature: str, v: float) -> float | None:
    """Undo log transforms so explanations show values a fraud analyst recognises."""
    v = float(v)
    if feature == "log_amount_to_avg":
        v = float(np.exp(v))
    elif feature.startswith("log_"):
        v = float(np.expm1(v))
    return _clean(v)
