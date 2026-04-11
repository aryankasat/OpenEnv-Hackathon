"""
OpenEnv-FragileChain: Typed Pydantic Models

Defines the Action, Observation, State, and Reward interfaces
for the pharmaceutical cold-chain logistics environment.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    REBALANCE = "rebalance"
    REROUTE_HUB = "reroute_hub"
    SCOUT = "scout"
    DO_NOTHING = "do_nothing"


class ShippingMode(str, Enum):
    STANDARD = "standard"
    EXPRESS = "express"
    BIO_HAZARD = "bio_hazard"


class TrialPhase(str, Enum):
    PHASE_I = "Phase I"
    PHASE_II = "Phase II"
    PHASE_III = "Phase III"


class AlertType(str, Enum):
    NONE = "none"
    WEATHER_EVENT = "weather_event"
    STRIKE = "strike"
    HUB_CLOSURE = "hub_closure"
    FRIDGE_MALFUNCTION = "fridge_malfunction"
    HURRICANE = "hurricane"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class SiteStatus(BaseModel):
    """Per-site snapshot delivered to the agent each step."""

    site_id: str = Field(..., description="Unique identifier for the clinical trial site")
    vials_in_stock: int = Field(..., ge=0, description="Current vials inventory")
    patient_load: int = Field(..., ge=0, description="Number of active patients at this site")
    trial_phase: TrialPhase = Field(..., description="Trial phase determines priority weight")
    current_temp_c: float = Field(..., description="Current storage temperature in °C")
    avg_thermal_debt: float = Field(
        ..., ge=0.0, le=1.0,
        description="Cumulative thermal stress (0=pristine, 1=fully degraded)"
    )
    days_until_stockout: Optional[int] = Field(
        None, description="Estimated days until vials run out at current demand"
    )
    is_hub: bool = Field(False, description="True if this site also serves as a distribution hub")
    is_isolated: bool = Field(False, description="True if site is cut off from normal routes")
    alert: AlertType = Field(AlertType.NONE, description="Active alert at this site")

    model_config = ConfigDict(use_enum_values=True, populate_by_name=True)


class GlobalAlert(BaseModel):
    """A global event affecting the supply chain."""

    alert_type: AlertType
    affected_sites: List[str] = Field(default_factory=list)
    severity: float = Field(0.0, ge=0.0, le=1.0)
    description: str = ""


# ---------------------------------------------------------------------------
# Core OpenEnv models
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    Agent observation returned after each step() and reset().

    Contains the full view of the supply network state.
    """

    # Per-site data
    sites: List[SiteStatus] = Field(..., description="Status of all clinical trial sites")

    # Global state
    current_day: int = Field(0, description="Current simulation day")
    remaining_budget: float = Field(0.0, description="Remaining logistics budget in USD")
    global_alerts: List[GlobalAlert] = Field(
        default_factory=list,
        description="Active global disruption events"
    )

    # Unstructured protocol metadata (simulates real document snippets)
    protocol_metadata: str = Field(
        "",
        description="Excerpt from trial protocol document with priority instructions"
    )

    # Episode signals
    done: bool = Field(False, description="True if episode has ended")
    reward: float = Field(0.0, description="Reward signal from last action")
    info: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic information")


class Action(BaseModel):
    """
    Agent action sent to the environment via step().

    The primary action types are:
    - rebalance: Move vials between two sites
    - reroute_hub: Change which hub services a set of sites
    - scout: Check real temperature/status of a site (costs budget)
    - do_nothing: Pass, allow one day to elapse
    """

    action_type: ActionType = Field(..., description="Primary action to execute")

    # For rebalance / reroute_hub
    source_id: Optional[str] = Field(None, description="Source site ID")
    target_id: Optional[str] = Field(None, description="Target site ID")
    amount: Optional[int] = Field(None, ge=0, description="Number of vials to move")
    mode: ShippingMode = Field(
        ShippingMode.STANDARD,
        description="Shipping mode (affects cost and speed)"
    )

    # For reroute_hub
    affected_sites: Optional[List[str]] = Field(
        None,
        description="List of site IDs to reroute to a new hub"
    )

    # LLM planning field
    internal_thought: str = Field(
        "",
        description="Agent's internal reasoning (logged but does not affect sim)"
    )

    @model_validator(mode='after')
    def validate_action_fields(self) -> 'Action':
        VALID_SITES = {"HUB_CENTRAL", "HUB_COAST", "SITE_ALPHA", "SITE_BETA", "SITE_GAMMA", "SITE_DELTA", "SITE_EPSILON"}
        
        # 1. Validate common site IDs if provided
        for sid in [self.source_id, self.target_id]:
            if sid is not None and sid not in VALID_SITES:
                raise ValueError(f"Invalid site_id '{sid}'. Must be one of {VALID_SITES}")
                
        if self.affected_sites is not None:
            for sid in self.affected_sites:
                if sid not in VALID_SITES:
                    raise ValueError(f"Invalid site_id '{sid}' in affected_sites. Must be one of {VALID_SITES}")

        # 2. Action-specific requirement checks
        atype = self.action_type
        if atype == ActionType.REBALANCE:
            if not self.source_id or not self.target_id or self.amount is None:
                raise ValueError("REBALANCE action requires 'source_id', 'target_id', and 'amount'.")
            if self.amount <= 0:
                raise ValueError("REBALANCE 'amount' must be strictly positive (greater than 0).")
            if self.source_id == self.target_id:
                raise ValueError("REBALANCE 'source_id' and 'target_id' must be different.")
        elif atype == ActionType.REROUTE_HUB:
            if not self.target_id or not self.affected_sites:
                raise ValueError("REROUTE_HUB action requires 'target_id' (new hub) and 'affected_sites'.")
        elif atype == ActionType.SCOUT:
            if not self.source_id:
                raise ValueError("SCOUT action requires a 'source_id'.")
                
        return self

    model_config = ConfigDict(use_enum_values=True, populate_by_name=True)


class State(BaseModel):
    """
    Persistent episode metadata accessible via state().

    Tracks the meta-level state of the running episode.
    """

    episode_id: str = Field(..., description="Unique identifier for this episode")
    step_count: int = Field(0, description="Number of steps taken in this episode")
    task_id: str = Field("task1", description="Active task identifier")
    current_day: int = Field(0, description="Current simulation day")
    max_days: int = Field(30, description="Episode horizon in days")
    total_reward: float = Field(0.0, description="Cumulative episode reward")
    task_score: float = Field(0.0001, description="Current grader score [0.0-1.0]")


class Reward(BaseModel):
    """
    Structured reward breakdown for a single step.

    R_t = 0.4 * ΔPSL - 0.2 * (ΔFiscalCost / MaxBudget) - 0.2 * ShippingCost
          + 0.2 * ThermalDebtPenalty
    """

    total: float = Field(0.0, description="Total scalar reward")
    patient_service_level_delta: float = Field(
        0.0, description="Change in patient service level (PSL)"
    )
    fiscal_cost_penalty: float = Field(0.0, description="Budget consumption penalty")
    shipping_cost_penalty: float = Field(0.0, description="Direct shipping cost penalty")
    thermal_debt_penalty: float = Field(0.0, description="Penalty for rising thermal debt")
    stockout_penalty: float = Field(0.0, description="Penalty per site that ran out of vials")


class TaskResult(BaseModel):
    """Final grader result for a completed episode."""

    task_id: str
    score: float = Field(0.0001, gt=0.0, lt=1.0, description="Final task score, strictly in (0, 1)")
    scientific_integrity: float = Field(0.0001, description="SI metric from proposal formula")
    doses_delivered: int = Field(0, description="Total effective doses delivered")
    total_demand: int = Field(0, description="Total demand across all sites")
    mean_thermal_debt: float = Field(0.0, description="Mean thermal debt at episode end")
    budget_remaining: float = Field(0.0, description="Budget not consumed")
    breakdown: Dict[str, float] = Field(default_factory=dict, description="Score components")
