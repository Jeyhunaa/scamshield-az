"""Pydantic request/response schemas for the ScamShield AZ API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scamshield.features.journey import EVENT_TYPES

Action = Literal["allow", "warn", "verify", "hold"]


class TransactionContext(BaseModel):
    """Transfer details plus customer context supplied by the bank's core systems.

    All history fields must describe ONLY transfers made before this one.
    """

    model_config = ConfigDict(extra="forbid")

    amount: float = Field(..., gt=0, le=10_000_000, description="Transfer amount (AZN)")
    hour: int = Field(..., ge=0, le=23, description="Local hour of the transfer")
    is_new_beneficiary: bool = Field(..., description="Customer has never paid this beneficiary before")
    prior_txn_count: int = Field(..., ge=0)
    prior_avg_amount: float | None = Field(None, ge=0, description="Mean previous transfer amount; null if no history")
    prior_max_amount: float | None = Field(None, ge=0, description="Largest previous transfer; null if no history")
    txn_last_24h: int = Field(0, ge=0)
    txn_last_7d: int = Field(0, ge=0)
    amount_last_24h: float = Field(0.0, ge=0)
    hours_since_last_txn: float | None = Field(None, ge=0, description="Null if no previous transfer")
    n_distinct_beneficiaries: int = Field(0, ge=0)
    tenure_days: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _history_consistency(self) -> TransactionContext:
        if self.prior_txn_count > 0 and (self.prior_avg_amount is None or self.prior_max_amount is None):
            raise ValueError("prior_avg_amount and prior_max_amount are required when prior_txn_count > 0")
        if self.txn_last_24h > self.txn_last_7d:
            raise ValueError("txn_last_24h cannot exceed txn_last_7d")
        if self.txn_last_7d > self.prior_txn_count:
            raise ValueError("txn_last_7d cannot exceed prior_txn_count")
        return self


class JourneyEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: str = Field(..., description=f"One of: {', '.join(EVENT_TYPES)}")
    t_offset_s: float = Field(..., ge=0, le=86_400, description="Seconds since session start")

    @model_validator(mode="after")
    def _known_type(self) -> JourneyEvent:
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event_type '{self.event_type}'")
        return self


class JourneyFeatures(BaseModel):
    """Pre-computed journey summary (alternative to raw events)."""

    model_config = ConfigDict(extra="forbid")

    session_duration_s: float = Field(..., ge=0)
    time_to_payment_s: float | None = Field(None, ge=0)
    n_payment_screen_visits: int = Field(..., ge=0)
    n_amount_edits: int = Field(0, ge=0)
    n_payee_edits: int = Field(0, ge=0)
    n_back_navigations: int = Field(0, ge=0)
    n_app_switches: int = Field(0, ge=0)
    background_time_s: float = Field(0.0, ge=0)
    background_share: float = Field(0.0, ge=0, le=1)
    confirm_time_s: float | None = Field(None, ge=0)
    n_browse_events: int = Field(0, ge=0)
    payee_pasted: bool = False
    call_active: bool = False


class AssessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction: TransactionContext
    journey_events: list[JourneyEvent] | None = Field(None, min_length=2, max_length=2000)
    journey_features: JourneyFeatures | None = None
    hold_allowed: bool = Field(True, description="Whether a temporary hold is legally/operationally permitted")

    @model_validator(mode="after")
    def _exactly_one_journey(self) -> AssessRequest:
        if (self.journey_events is None) == (self.journey_features is None):
            raise ValueError("provide exactly one of 'journey_events' or 'journey_features'")
        return self


class ActionCost(BaseModel):
    expected_cost: float
    residual_loss: float
    operational_cost: float
    friction_cost: float
    eligible: bool
    ineligible_reason: str | None


class RiskFactor(BaseModel):
    feature: str
    description: str
    value: float | None
    contribution: float = Field(..., description="Contribution to log-odds of scam")
    direction: str
    source: Literal["transaction", "journey"]


class AssessResponse(BaseModel):
    baseline_risk: float = Field(..., description="Calibrated scam probability from the existing transaction model")
    enhanced_risk: float = Field(..., description="Calibrated scam probability from ScamShield")
    recommended_action: Action
    baseline_policy_action: Action = Field(..., description="Action the fixed-threshold baseline policy would take")
    decision_explanation: str
    expected_costs: dict[Action, ActionCost]
    risk_factors: list[RiskFactor]
    baseline_drivers: list[RiskFactor]
    journey_features: dict[str, float | None]
    model_version: str
    disclaimer: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    models_loaded: bool
    model_version: str | None
    trained_on: str | None
    baseline_thresholds: dict[str, float | None] | None
    detail: str | None = None


class Scenario(BaseModel):
    """Demo scenario inputs (no outcomes): consumed by the customer banking app."""

    id: str
    title: str
    kind: Literal["legitimate", "scam"]
    intended_action: Action
    story: str
    customer_view: dict[str, str]
    transaction: TransactionContext
    journey_events: list[JourneyEvent]
