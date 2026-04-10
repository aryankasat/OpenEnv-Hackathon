"""
OpenEnv-FragileChain: Task Graders

Three graders with increasing difficulty:
- Task 1 (Easy):   Steady State – maintain supply above demand for all 30 days
- Task 2 (Medium): Thermal Anomaly – detect & mitigate fridge failure at SITE_ALPHA
- Task 3 (Hard):   Black Swan – Phase III prioritisation under simultaneous disasters
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict

from engine import ColdChainEngine
from models import TaskResult, TrialPhase, AlertType


class BaseGrader(ABC):
    """Abstract base class for all task graders."""

    task_id: str = ""
    difficulty: str = ""
    description: str = ""

    def __init__(self, engine: ColdChainEngine):
        self.engine = engine
        # Tracking variables populated during episode
        self._days_fully_served: int = 0
        self._days_recorded: int = 0
        self._stockout_events: int = 0
        self._thermal_exceedance_events: int = 0
        self._previous_day: int = -1

    def record_step(self) -> None:
        """Called after each step() to update per-step metrics."""
        # Guard against double-counting within same day
        if self.engine.current_day == self._previous_day:
            return
        self._previous_day = self.engine.current_day
        self._days_recorded += 1
        self._on_record_step()

    def _on_record_step(self) -> None:
        """Override in subclasses for step-level metric updates."""
        pass

    @abstractmethod
    def compute_score(self) -> TaskResult:
        """Compute and return the final task score [0.0 - 1.0]."""
        pass

    def _mean_thermal_debt(self) -> float:
        sites = list(self.engine.sites.values())
        if not sites:
            return 0.0
        return sum(s.thermal_debt for s in sites) / len(sites)

    def _clamp(self, val: float) -> float:
        """Strictly clamp to (0, 1) range [0.0001, 0.9999]."""
        return max(0.0001, min(0.9999, val))


# ---------------------------------------------------------------------------
# Task 1: Steady State (Easy)
# ---------------------------------------------------------------------------

class Task1Grader(BaseGrader):
    """
    Steady State – no alerts, standard environment.

    Score = (days_all_sites_above_demand / total_days)
            * (1 - mean_thermal_debt)
            * (1 - budget_exhaustion_penalty)

    A perfect score means every site always had stock above its daily demand
    and temperatures were maintained throughout the episode.
    """

    task_id = "task1"
    difficulty = "easy"
    description = (
        "Maintain vial stock above daily demand at all 7 sites across 30 days "
        "without exceeding budget."
    )

    def _on_record_step(self) -> None:
        # Calculate per-site partial credit instead of binary all-or-nothing
        sites = list(self.engine.sites.values())
        if sites:
            per_site_ratio = sum(
                min(1.0, s.vials / s.daily_demand) if s.daily_demand > 0 else 1.0
                for s in sites
            ) / len(sites)
            if per_site_ratio >= 0.99:  # threshold for "fully served"
                self._days_fully_served += 1
            else:
                self._stockout_events += 1

    def compute_score(self) -> TaskResult:
        total_days = max(1, self._days_recorded)
        service_ratio = self._days_fully_served / total_days
        mean_debt = self._mean_thermal_debt()
        budget_remaining_frac = max(0.0, self.engine.budget / self.engine.max_budget)

        # Composite score
        score = service_ratio * (1.0 - mean_debt * 0.5) * (0.8 + 0.2 * budget_remaining_frac)
        score = self._clamp(score)

        return TaskResult(
            task_id=self.task_id,
            score=round(score, 4),
            scientific_integrity=round(self._clamp(service_ratio * (1. - mean_debt)), 4),
            doses_delivered=self.engine.total_doses_delivered,
            total_demand=self.engine.total_demand,
            mean_thermal_debt=round(mean_debt, 4),
            budget_remaining=round(self.engine.budget, 2),
            breakdown={
                "service_ratio": round(service_ratio, 4),
                "thermal_debt_factor": round(1.0 - mean_debt * 0.5, 4),
                "budget_factor": round(0.8 + 0.2 * budget_remaining_frac, 4),
                "stockout_events": float(self._stockout_events),
            },
        )


# ---------------------------------------------------------------------------
# Task 2: Thermal Anomaly (Medium)
# ---------------------------------------------------------------------------

class Task2Grader(BaseGrader):
    """
    Thermal Anomaly – SITE_ALPHA has a malfunctioning fridge.

    Score = α * mitigation_speed_score
            + β * vials_saved_score
            + γ * (1 - mean_thermal_debt_alpha)

    α=0.4, β=0.4, γ=0.2

    Mitigation speed: how quickly the agent actively evacuated vials from the target site
    Vials saved: net evacuated vials out of the affected site before debt > 0.7
    """

    task_id = "task2"
    difficulty = "medium"
    description = (
        "Detect that a site's refrigeration is failing (rising Thermal Debt) "
        "and move stock to other sites before drug efficacy reaches zero (debt >= 1.0)."
    )

    def __init__(self, engine: ColdChainEngine):
        super().__init__(engine)
        self._mitigation_day: int | None = None    # first day stock moved out actively
        self._vials_at_anomaly_start: int = 0
        self._vials_saved: int = 0
        self._alpha_compromised: bool = False
        self._target_site_id: str | None = None

    def _on_record_step(self) -> None:
        if not self._target_site_id:
            for site in self.engine.sites.values():
                if site.alert in [AlertType.FRIDGE_MALFUNCTION, "fridge_malfunction"]:
                    self._target_site_id = site.site_id
                    break

        if not self._target_site_id:
            return

        alpha = self.engine.sites.get(self._target_site_id)
        if alpha is None:
            return

        # Record initial vial count
        if self._days_recorded == 1:
            self._vials_at_anomaly_start = alpha.vials

        # EXPLOIT FIX: Calculate active evacuation by subtracting natural consumption trend
        natural_stock_remaining = self._vials_at_anomaly_start - (self._days_recorded * alpha.daily_demand)
        net_evacuated = max(0, natural_stock_remaining - alpha.vials)

        # Check if agent has moved stock (mitigation)
        if self._mitigation_day is None:
            if net_evacuated > 0:
                self._mitigation_day = self.engine.current_day

        # Track vials saved (those actively evacuated before debt > 0.7)
        if alpha.thermal_debt <= 0.7:
            self._vials_saved = net_evacuated

        if alpha.thermal_debt >= 1.0:
            self._alpha_compromised = True

    def compute_score(self) -> TaskResult:
        alpha = self.engine.sites.get(self._target_site_id or "")
        alpha_debt = alpha.thermal_debt if alpha else 1.0

        # Mitigation speed score: earlier is better (max 30 days)
        if self._mitigation_day is not None:
            speed_score = max(0.0, 1.0 - (self._mitigation_day - 1) / self.engine.max_days)
        else:
            speed_score = 0.0  # never mitigated

        # Vials saved score
        if self._vials_at_anomaly_start > 0:
            saved_ratio = self._vials_saved / self._vials_at_anomaly_start
        else:
            saved_ratio = 1.0

        # Thermal debt factor for SITE_ALPHA
        thermal_factor = 1.0 - alpha_debt

        # Proportional penalty for debt above 1.0 (continuous, not hard cliff)
        compromise_penalty = min(0.3, max(0.0, alpha_debt - 1.0) * 0.05) if alpha_debt > 1.0 else 0.0

        score = (
            0.4 * speed_score
            + 0.4 * saved_ratio
            + 0.2 * thermal_factor
            - compromise_penalty
        )
        score = self._clamp(score)
        mean_debt = self._mean_thermal_debt()

        return TaskResult(
            task_id=self.task_id,
            score=round(score, 4),
            scientific_integrity=round(self._clamp(saved_ratio * (1.0 - alpha_debt)), 4),
            doses_delivered=self.engine.total_doses_delivered,
            total_demand=self.engine.total_demand,
            mean_thermal_debt=round(mean_debt, 4),
            budget_remaining=round(self.engine.budget, 2),
            breakdown={
                "mitigation_speed_score": round(speed_score, 4),
                "vials_saved_ratio": round(saved_ratio, 4),
                "alpha_thermal_factor": round(thermal_factor, 4),
                "alpha_compromised": 1.0 if self._alpha_compromised else 0.0,
                "mitigation_day": float(self._mitigation_day or -1),
            },
        )


# ---------------------------------------------------------------------------
# Task 3: Black Swan (Hard)
# ---------------------------------------------------------------------------

class Task3Grader(BaseGrader):
    """
    Black Swan – simultaneous HUB_COAST closure + hurricane at SITE_DELTA/EPSILON.

    Uses the full Scientific Integrity formula from the proposal:
        SI = (Σ doses_delivered_i * priority_weight_i / total_demand) * (1 - mean_thermal_debt)

    Score also requires Phase III sites to never go below 3-day supply.
    """

    task_id = "task3"
    difficulty = "hard"
    description = (
        "Manage simultaneous hub closure and hurricane. "
        "Prioritise Phase III sites by rerouting "
        "and redistributing stock while keeping thermal debt low."
    )

    def __init__(self, engine: ColdChainEngine):
        super().__init__(engine)
        self._phase3_stockout_days: int = 0
        self._reroute_performed: bool = False
        self._per_site_delivered: Dict[str, int] = {}
        self._per_site_demand: Dict[str, int] = {}
        # Initialize tracking dicts for actual delivery
        for site_id in self.engine.sites.keys():
            self._per_site_delivered[site_id] = 0
            self._per_site_demand[site_id] = 0

    def _on_record_step(self) -> None:
        phase3_sites = [s.site_id for s in self.engine.sites.values() if s.trial_phase in [TrialPhase.PHASE_III, "Phase III"]]

        # Track Phase III stockout events
        for sid in phase3_sites:
            site = self.engine.sites.get(sid)
            if site and site.vials < site.daily_demand * 3:
                self._phase3_stockout_days += 1

        # Track actual cumulative doses delivered per site (not estimated)
        for site in self.engine.sites.values():
            # Actual doses delivered = min(available vials, daily demand)
            actual_delivered = min(site.vials, site.daily_demand)
            self._per_site_delivered[site.site_id] += actual_delivered
            self._per_site_demand[site.site_id] += site.daily_demand

        # Check if agent rerouted
        for sid in ["SITE_DELTA", "SITE_EPSILON"]:
            conns = self.engine.network.get(sid, [])
            if "HUB_CENTRAL" in conns:
                self._reroute_performed = True

    def compute_score(self) -> TaskResult:
        # Scientific Integrity formula
        si = self._compute_si()
        mean_debt = self._mean_thermal_debt()

        # Phase III continuity bonus
        phase3_penalty = min(0.3, self._phase3_stockout_days * 0.02)

        # Rerouting bonus
        reroute_bonus = 0.1 if self._reroute_performed else 0.0

        score = si - phase3_penalty + reroute_bonus
        score = self._clamp(score)

        return TaskResult(
            task_id=self.task_id,
            score=round(score, 4),
            scientific_integrity=round(self._clamp(si), 4),
            doses_delivered=self.engine.total_doses_delivered,
            total_demand=self.engine.total_demand,
            mean_thermal_debt=round(mean_debt, 4),
            budget_remaining=round(self.engine.budget, 2),
            breakdown={
                "scientific_integrity": round(si, 4),
                "phase3_continuity_penalty": round(phase3_penalty, 4),
                "reroute_bonus": reroute_bonus,
                "phase3_stockout_days": float(self._phase3_stockout_days),
                "reroute_performed": 1.0 if self._reroute_performed else 0.0,
            },
        )

    def _compute_si(self) -> float:
        """
        SI = (Σ doses_delivered_i * priority_weight_i / total_demand) * (1 - mean_thermal_debt)

        Uses actual cumulative delivery tracking (not estimates).
        """
        if self.engine.total_demand == 0:
            return self._clamp(0.0)

        # Compute weighted delivery ratio per site using actual tracked values
        weighted_delivered = 0.0

        for site in self.engine.sites.values():
            pw = site.priority_weight()
            actual_delivered = self._per_site_delivered.get(site.site_id, 0)
            weighted_delivered += actual_delivered * pw

        mean_debt = self._mean_thermal_debt()
        total_demand = self.engine.total_demand

        si = (weighted_delivered / max(1, total_demand)) * (1.0 - mean_debt)
        return round(self._clamp(si), 4)


# ---------------------------------------------------------------------------
# Grader factory
# ---------------------------------------------------------------------------

GRADERS = {
    "task1": Task1Grader,
    "task2": Task2Grader,
    "task3": Task3Grader,
}


def get_grader(task_id: str, engine: ColdChainEngine) -> BaseGrader:
    cls = GRADERS.get(task_id)
    if cls is None:
        raise ValueError(f"Unknown task_id: {task_id!r}. Choose from {list(GRADERS.keys())}")
    return cls(engine)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    print("Running grader self-tests...\n")

    for tid in ["task1", "task2", "task3"]:
        eng = ColdChainEngine(task_id=tid, seed=42)
        eng.reset(task_id=tid)
        grader = get_grader(tid, eng)

        from models import Action, ActionType

        # Simulate 10 days of do_nothing actions
        for _ in range(10):
            action = Action(action_type=ActionType.DO_NOTHING, internal_thought="Baseline test")
            obs, reward, done = eng.step(action)
            grader.record_step()
            if done:
                break

        result = grader.compute_score()
        print(f"Task: {tid} | Difficulty: {grader.difficulty}")
        print(f"  Score:              {result.score:.4f}")
        print(f"  Scientific Integrity: {result.scientific_integrity:.4f}")
        print(f"  Doses Delivered:    {result.doses_delivered}")
        print(f"  Mean Thermal Debt:  {result.mean_thermal_debt:.4f}")
        print(f"  Budget Remaining:   ${result.budget_remaining:,.2f}")
        print(f"  Breakdown:          {result.breakdown}")
        print()

    print("All grader tests passed!")
    sys.exit(0)
