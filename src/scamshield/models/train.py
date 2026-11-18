"""Model training: baseline logistic regression, enhanced CatBoost, Platt calibration.

Leakage controls
----------------
* The enhanced model consumes the baseline score as a feature (stacking). On training
  rows it receives OUT-OF-FOLD baseline scores, so it never sees a score produced by a
  model that was fitted on that same row. Validation/test rows receive scores from the
  baseline fitted on the full training period.
* Calibrators are fitted on the validation period only. The test period is untouched
  until final evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from scamshield.data.preprocessing import TRANSACTION_FEATURES
from scamshield.features.journey import JOURNEY_FEATURES

BASELINE_SCORE = "baseline_logit"
ENHANCED_FEATURES: list[str] = [BASELINE_SCORE, *JOURNEY_FEATURES]
LABEL = "is_scam"
EPS = 1e-6

# Ablation variants: name -> (model family, feature list)
VARIANTS: dict[str, tuple[str, list[str]]] = {
    "baseline_lr": ("logreg", TRANSACTION_FEATURES),
    "catboost_txn_only": ("catboost", TRANSACTION_FEATURES),
    "catboost_journey_only": ("catboost", JOURNEY_FEATURES),
    "scamshield": ("catboost", ENHANCED_FEATURES),
    "catboost_all_features": ("catboost", [*TRANSACTION_FEATURES, *JOURNEY_FEATURES]),
}


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


@dataclass
class PlattCalibrator:
    """Platt scaling on the logit of the raw score: p_cal = sigmoid(a * logit(p_raw) + b).

    Chosen over isotonic regression because the validation set contains ~100 positives,
    where isotonic step functions overfit; two parameters are stable and monotone.
    """

    a: float = 1.0
    b: float = 0.0

    def fit(self, raw_p: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        y = np.asarray(y)
        if len(np.unique(y)) < 2:
            raise ValueError("calibration data must contain both classes")
        lr = LogisticRegression(C=1e4, max_iter=1000)
        lr.fit(logit(raw_p).reshape(-1, 1), y)
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def transform(self, raw_p: np.ndarray) -> np.ndarray:
        z = self.a * logit(raw_p) + self.b
        return 1.0 / (1.0 + np.exp(-z))


@dataclass
class FittedModel:
    name: str
    model: Any                      # sklearn Pipeline or CatBoostClassifier
    features: list[str]
    calibrator: PlattCalibrator = field(default_factory=PlattCalibrator)

    def predict_raw(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(frame[self.features])[:, 1]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Calibrated scam probability."""
        return self.calibrator.transform(self.predict_raw(frame))


def fit_logreg(X: pd.DataFrame, y: pd.Series, cfg: dict[str, Any]) -> Pipeline:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(cfg["baseline"]["C"]), max_iter=int(cfg["baseline"]["max_iter"])),
    )
    return model.fit(X, y)


def fit_catboost(X: pd.DataFrame, y: pd.Series, cfg: dict[str, Any], seed: int) -> CatBoostClassifier:
    p = cfg["enhanced"]
    model = CatBoostClassifier(
        iterations=int(p["iterations"]),
        learning_rate=float(p["learning_rate"]),
        depth=int(p["depth"]),
        l2_leaf_reg=float(p["l2_leaf_reg"]),
        random_strength=float(p["random_strength"]),
        loss_function="Logloss",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    return model.fit(X, y)


def oof_baseline_logit(X: pd.DataFrame, y: pd.Series, cfg: dict[str, Any], seed: int) -> np.ndarray:
    """Out-of-fold baseline scores for the training rows (stacking without leakage)."""
    oof = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=int(cfg["baseline"]["oof_folds"]), shuffle=True, random_state=seed)
    for fit_idx, pred_idx in skf.split(X, y):
        m = fit_logreg(X.iloc[fit_idx], y.iloc[fit_idx], cfg)
        oof[pred_idx] = m.predict_proba(X.iloc[pred_idx])[:, 1]
    return logit(oof)


def add_baseline_score(table: pd.DataFrame, baseline: Pipeline, oof_train_logit: np.ndarray | None = None) -> pd.DataFrame:
    """Attach `baseline_logit`: OOF for train rows (if given), full baseline model elsewhere."""
    out = table.copy()
    out[BASELINE_SCORE] = logit(baseline.predict_proba(out[TRANSACTION_FEATURES])[:, 1])
    if oof_train_logit is not None:
        is_train = (out["split"] == "train").to_numpy()
        if is_train.sum() != len(oof_train_logit):
            raise ValueError("OOF score length does not match number of training rows")
        out.loc[is_train, BASELINE_SCORE] = oof_train_logit
    return out


def train_all(table: pd.DataFrame, cfg: dict[str, Any], variants: list[str] | None = None) -> tuple[dict[str, FittedModel], pd.DataFrame]:
    """Train + calibrate the requested variants. Returns models and the table with baseline scores."""
    seed = int(cfg["seed"])
    variants = variants or list(VARIANTS)
    train = table[table["split"] == "train"]
    val = table[table["split"] == "val"]
    if train[LABEL].nunique() < 2 or val[LABEL].nunique() < 2:
        raise ValueError("train and validation periods must contain both classes")

    baseline = fit_logreg(train[TRANSACTION_FEATURES], train[LABEL], cfg)
    oof = oof_baseline_logit(train[TRANSACTION_FEATURES], train[LABEL], cfg, seed)
    scored = add_baseline_score(table, baseline, oof)
    train, val = scored[scored["split"] == "train"], scored[scored["split"] == "val"]

    models: dict[str, FittedModel] = {}
    for name in variants:
        family, feats = VARIANTS[name]
        if name == "baseline_lr":
            est = baseline
        elif family == "catboost":
            est = fit_catboost(train[feats], train[LABEL], cfg, seed)
        else:
            est = fit_logreg(train[feats], train[LABEL], cfg)
        fm = FittedModel(name=name, model=est, features=list(feats))
        fm.calibrator = PlattCalibrator().fit(fm.predict_raw(val), val[LABEL].to_numpy())
        models[name] = fm
    return models, scored


def save_models(models: dict[str, FittedModel], out_dir: str | Path, metadata: dict[str, Any]) -> Path:
    """Persist the two serving models (baseline + ScamShield) and human-readable metadata."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle = {"baseline": models["baseline_lr"], "enhanced": models["scamshield"], "transaction_features": TRANSACTION_FEATURES}
    path = out / "scamshield_models.joblib"
    joblib.dump(bundle, path)
    models["scamshield"].model.save_model(str(out / "enhanced_catboost.cbm"))
    meta = {
        **metadata,
        "baseline_features": models["baseline_lr"].features,
        "enhanced_features": models["scamshield"].features,
        "calibration": {k: {"a": models[k].calibrator.a, "b": models[k].calibrator.b} for k in ("baseline_lr", "scamshield")},
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    return path
