"""Shared fixtures: a small, fast end-to-end training run into a temporary artifacts dir.

Tests never depend on the full pipeline having been run, and never write into the
project's data/ or artifacts/ folders.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scamshield.config import load_intervention_yaml, load_model_config


@pytest.fixture(scope="session")
def small_cfg() -> dict:
    cfg = copy.deepcopy(load_model_config())
    cfg["data"]["n_customers"] = 700
    cfg["data"]["scam_episode_rate"] = 0.04  # higher prevalence so small splits contain positives
    cfg["enhanced"]["iterations"] = 80
    cfg["baseline"]["oof_folds"] = 3
    cfg["evaluation"]["bootstrap_reps"] = 20
    return cfg


@pytest.fixture(scope="session")
def intervention_dict() -> dict:
    return load_intervention_yaml()


@pytest.fixture(scope="session")
def small_artifacts(tmp_path_factory, small_cfg, intervention_dict) -> Path:
    from scamshield.pipeline import run_full_pipeline

    out = tmp_path_factory.mktemp("artifacts")
    run_full_pipeline(small_cfg, intervention_dict, out, with_sensitivity=False, data_dir=None)
    return out
