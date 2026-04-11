"""
OpenEnv-FragileChain: Cold-Chain Simulation Engine

Manages the pharmaceutical supply network simulation including:
- Site network topology
- Temperature / Thermal Debt dynamics
- Daily demand consumption
- Budget tracking
- Event generation (alerts, disruptions)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from models import (
    Action,
    ActionType,
    AlertType,
    GlobalAlert,
    Observation,
    Reward,
    ShippingMode,
    SiteStatus,
    TrialPhase,
)


# ---------------------------------------------------------------------------
# Site configuration constants
# ---------------------------------------------------------------------------

SHIPPING_COST_PER_VIAL: Dict[str, float] = {
    ShippingMode.STANDARD: 12.0,
    ShippingMode.EXPRESS: 35.0,
    ShippingMode.BIO_HAZARD: 80.0,
}

SHIPPING_DAYS: Dict[str, int] = {
    ShippingMode.STANDARD: 3,
    ShippingMode.EXPRESS: 1,
    ShippingMode.BIO_HAZARD: 1,
}

PRIORITY_WEIGHTS: Dict[str, float] = {
    TrialPhase.PHASE_I: 0.5,
    TrialPhase.PHASE_II: 0.75,
    TrialPhase.PHASE_III: 1.0,
}

SCOUT_COST = 500.0
TARGET_TEMP_C = -40.0  # midpoint of -20 to -80 range
TEMP_TOLERANCE = 20.0  # acceptable deviation in °C


@dataclass
class SiteState:
    """Mutable internal site state."""

    site_id: str
    vials: int
    patient_load: int
    trial_phase: str
    temp_c: float
    thermal_debt: float
    is_hub: bool
    is_isolated: bool = False
    alert: str = AlertType.NONE

    # Noise and degradation parameters
    base_temp_c: float = field(init=False)
    temp_noise_std: float = 1.5
    thermal_decay_rate: float = 0.01   # debt accumulation per day at ambient

    def __post_init__(self):
        self.base_temp_c = self.temp_c

    @property
    def daily_demand(self) -> int:
        """Vials consumed per day (1 per 3 patients, minimum 1)."""
        return max(1, self.patient_load // 3)

    @property
    def days_until_stockout(self) -> Optional[int]:
        if self.daily_demand == 0:
            return None
        return max(0, self.vials // self.daily_demand)

    def priority_weight(self) -> float:
        return PRIORITY_WEIGHTS.get(self.trial_phase, 0.75)

    def to_model(self) -> SiteStatus:
        return SiteStatus(
            site_id=self.site_id,
            vials_in_stock=max(0, self.vials),
            patient_load=self.patient_load,
            trial_phase=self.trial_phase,
            current_temp_c=round(self.temp_c, 2),
            avg_thermal_debt=round(max(0.0, min(1.0, self.thermal_debt)), 4),
            days_until_stockout=self.days_until_stockout,
            is_hub=self.is_hub,
            is_isolated=self.is_isolated,
            alert=self.alert,
        )


# ---------------------------------------------------------------------------
# Routing graph
# ---------------------------------------------------------------------------

# site_id -> list of directly reachable site_ids
DEFAULT_NETWORK: Dict[str, List[str]] = {
    "HUB_CENTRAL": ["SITE_ALPHA", "SITE_BETA", "SITE_GAMMA", "HUB_COAST"],
    "HUB_COAST":   ["SITE_DELTA", "SITE_EPSILON", "HUB_CENTRAL"],
    "SITE_ALPHA":  ["HUB_CENTRAL"],
    "SITE_BETA":   ["HUB_CENTRAL"],
    "SITE_GAMMA":  ["HUB_CENTRAL"],
    "SITE_DELTA":  ["HUB_COAST"],
    "SITE_EPSILON":["HUB_COAST"],
}

DEFAULT_SITES_CONFIG = [
    {
        "site_id": "HUB_CENTRAL",
        "vials": 500,
        "patient_load": 20,
        "trial_phase": TrialPhase.PHASE_II,
        "temp_c": -42.0,
        "thermal_debt": 0.0,
        "is_hub": True,
    },
    {
        "site_id": "HUB_COAST",
        "vials": 300,
        "patient_load": 15,
        "trial_phase": TrialPhase.PHASE_II,
        "temp_c": -38.0,
        "thermal_debt": 0.0,
        "is_hub": True,
    },
    {
        "site_id": "SITE_ALPHA",
        "vials": 80,
        "patient_load": 30,
        "trial_phase": TrialPhase.PHASE_III,
        "temp_c": -40.0,
        "thermal_debt": 0.0,
        "is_hub": False,
    },
    {
        "site_id": "SITE_BETA",
        "vials": 60,
        "patient_load": 20,
        "trial_phase": TrialPhase.PHASE_I,
        "temp_c": -41.0,
        "thermal_debt": 0.0,
        "is_hub": False,
    },
    {
        "site_id": "SITE_GAMMA",
        "vials": 70,
        "patient_load": 25,
        "trial_phase": TrialPhase.PHASE_II,
        "temp_c": -39.0,
        "thermal_debt": 0.0,
        "is_hub": False,
    },
    {
        "site_id": "SITE_DELTA",
        "vials": 90,
        "patient_load": 35,
        "trial_phase": TrialPhase.PHASE_III,
        "temp_c": -40.0,
        "thermal_debt": 0.0,
        "is_hub": False,
    },
    {
        "site_id": "SITE_EPSILON",
        "vials": 50,
        "patient_load": 18,
        "trial_phase": TrialPhase.PHASE_I,
        "temp_c": -40.0,
        "thermal_debt": 0.0,
        "is_hub": False,
    },
]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ColdChainEngine:
    """
    Discrete-time simulation of a pharmaceutical cold-chain network.

    Each step() corresponds to taking a logistical action.
    advance_day() is called at the end of each step to simulate natural decay.
    """

    def __init__(
        self,
        task_id: str = "task1",
        seed: Optional[int] = None,
        max_days: int = 30,
        max_budget: float = 500_000.0,
    ):
        self.task_id = task_id
        self.seed = seed
        self.max_days = max_days
        self.max_budget = max_budget
        self._rng = random.Random(seed)

        # Populated by reset()
        self.sites: Dict[str, SiteState] = {}
        self.network: Dict[str, List[str]] = {}
        self.current_day: int = 0
        self.budget: float = max_budget
        self.global_alerts: List[GlobalAlert] = []
        self.in_transit: List[Dict] = []

        # Cumulative tracking
        self.total_doses_delivered: int = 0
        self.total_demand: int = 0
        self.shipping_cost_accumulated: float = 0.0

        # Previous PSL for delta computation
        self._prev_psl: float = 1.0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None) -> None:
        """Re-initialise all state for a new episode."""
        if task_id:
            self.task_id = task_id
        self._rng = random.Random(self.seed)
        self.current_day = 0
        self.budget = self.max_budget
        self.global_alerts = []
        self.in_transit = []
        self.total_doses_delivered = 0
        self.total_demand = 0
        self.shipping_cost_accumulated = 0.0

        # Build fresh sites
        self.network = {k: list(v) for k, v in DEFAULT_NETWORK.items()}
        self.sites = {}
        for cfg in DEFAULT_SITES_CONFIG:
            s = SiteState(
                site_id=cfg["site_id"],
                vials=cfg["vials"],
                patient_load=cfg["patient_load"],
                trial_phase=cfg["trial_phase"],
                temp_c=cfg["temp_c"],
                thermal_debt=cfg["thermal_debt"],
                is_hub=cfg["is_hub"],
            )
            self.sites[s.site_id] = s

        # Apply task-specific initial conditions
        self._apply_task_scenario()
        self._prev_psl = self._compute_psl()

    def _apply_task_scenario(self) -> None:
        """Inject task-specific starting conditions."""
        if self.task_id == "task1":
            # Steady state – clean start, no alerts
            pass

        elif self.task_id == "task2":
            # Thermal Anomaly – SITE_ALPHA's fridge starts malfunctioning
            self.sites["SITE_ALPHA"].alert = AlertType.FRIDGE_MALFUNCTION
            self.sites["SITE_ALPHA"].temp_noise_std = 5.0
            self.sites["SITE_ALPHA"].thermal_decay_rate = 0.08  # fast decay

        elif self.task_id == "task3":
            # Black Swan – HUB_COAST closure + hurricane heading to SITE_DELTA/EPSILON
            self.sites["HUB_COAST"].is_isolated = True
            self.sites["HUB_COAST"].alert = AlertType.HUB_CLOSURE
            self.sites["SITE_DELTA"].alert = AlertType.HURRICANE
            self.sites["SITE_EPSILON"].alert = AlertType.HURRICANE
            # Reduce coastal site stock to create urgency
            self.sites["SITE_DELTA"].vials = 20
            self.sites["SITE_EPSILON"].vials = 15
            self.global_alerts = [
                GlobalAlert(
                    alert_type=AlertType.HUB_CLOSURE,
                    affected_sites=["HUB_COAST"],
                    severity=0.9,
                    description="HUB_COAST closed due to transport strike.",
                ),
                GlobalAlert(
                    alert_type=AlertType.HURRICANE,
                    affected_sites=["SITE_DELTA", "SITE_EPSILON"],
                    severity=0.85,
                    description="Category-3 hurricane landfall expected in 4 days.",
                ),
            ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, action: Action) -> Tuple[Observation, Reward, bool]:
        """
        Execute one action and advance the simulation by one day.

        Returns:
            (observation, reward, done)
        """
        reward_breakdown = self._execute_action(action)
        self._advance_day()
        self._maybe_inject_random_alert()

        done = (self.current_day >= self.max_days) or self._is_catastrophic_failure()
        obs = self._build_observation(done=done, reward=reward_breakdown.total)
        return obs, reward_breakdown, done

    def get_observation(self) -> Observation:
        """Return current observation without advancing state."""
        return self._build_observation(done=False, reward=0.0)

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute_action(self, action: Action) -> Reward:
        psl_before = self._compute_psl()

        shipping_cost = 0.0
        fiscal_cost = 0.0
        thermal_penalty = 0.0
        stockout_penalty = 0.0

        atype = action.action_type if isinstance(action.action_type, str) else action.action_type.value

        if atype == ActionType.REBALANCE or atype == "rebalance":
            shipping_cost, fiscal_cost = self._do_rebalance(action)

        elif atype == ActionType.REROUTE_HUB or atype == "reroute_hub":
            fiscal_cost = self._do_reroute_hub(action)

        elif atype == ActionType.SCOUT or atype == "scout":
            fiscal_cost = self._do_scout(action)

        elif atype == ActionType.DO_NOTHING or atype == "do_nothing":
            pass  # No cost, but also no benefit

        psl_after = self._compute_psl()
        delta_psl = psl_after - psl_before

        # Thermal debt aggregation penalty
        avg_debt = sum(s.thermal_debt for s in self.sites.values()) / max(1, len(self.sites))
        thermal_penalty = -avg_debt * 0.15

        # Stockout penalty
        stockouts = sum(1 for s in self.sites.values() if s.vials <= 0)
        stockout_penalty = -stockouts * 0.05

        # Reward formula:  R = 0.2*PSL_after + 0.3*ΔPSL - 0.2*(fiscal_cost/max_budget) - 0.2*shipping_cost_norm + thermal + stockout
        # Incorporates absolute positive reward for maintaining high patient service levels over time.
        total = (
            0.2 * psl_after
            + 0.3 * delta_psl
            - 0.2 * (fiscal_cost / self.max_budget)
            - 0.2 * (shipping_cost / max(1.0, self.max_budget * 0.01))
            + thermal_penalty
            + stockout_penalty
        )
        total = max(-1.0, min(1.0, total))  # clip

        self._prev_psl = psl_after

        return Reward(
            total=round(total, 4),
            patient_service_level_delta=round(delta_psl, 4),
            fiscal_cost_penalty=round(-0.2 * (fiscal_cost / self.max_budget), 4),
            shipping_cost_penalty=round(-0.2 * (shipping_cost / max(1.0, self.max_budget * 0.01)), 4),
            thermal_debt_penalty=round(thermal_penalty, 4),
            stockout_penalty=round(stockout_penalty, 4),
        )

    def _do_rebalance(self, action: Action) -> Tuple[float, float]:
        """Move vials from source to target."""
        src_id = action.source_id
        tgt_id = action.target_id
        amount = action.amount or 0
        mode = action.mode if isinstance(action.mode, str) else action.mode.value

        if not src_id or not tgt_id:
            return 0.0, 0.0
        if src_id not in self.sites or tgt_id not in self.sites:
            return 0.0, 0.0
        if amount <= 0:
            return 0.0, 0.0

        src = self.sites[src_id]
        tgt = self.sites[tgt_id]

        # Check connectivity (skip check if target is isolated – can't receive)
        if tgt.is_isolated:
            return 0.0, 0.0

        if src_id == tgt_id:
            return 0.0, 0.0

        # Clamp to available stock
        actual_amount = min(amount, src.vials)
        if actual_amount <= 0:
            return 0.0, 0.0

        src.vials -= actual_amount

        shipping_days = SHIPPING_DAYS.get(mode, SHIPPING_DAYS.get(ShippingMode.STANDARD, 3))
        self.in_transit.append({
            "target_id": tgt_id,
            "amount": actual_amount,
            "arrival_day": self.current_day + shipping_days
        })

        # Cost calculation
        cost_per_vial = SHIPPING_COST_PER_VIAL.get(mode, SHIPPING_COST_PER_VIAL[ShippingMode.STANDARD])
        shipping_cost = actual_amount * cost_per_vial
        self.budget -= shipping_cost
        self.shipping_cost_accumulated += shipping_cost

        return shipping_cost, shipping_cost

    def _do_reroute_hub(self, action: Action) -> float:
        """Reroute a set of sites to connect through a different hub."""
        tgt_hub = action.target_id
        sites_to_reroute = action.affected_sites or []
        cost = 2000.0  # fixed rerouting admin cost

        if not tgt_hub or tgt_hub not in self.sites:
            return 0.0

        if not sites_to_reroute:
            return 0.0

        hub = self.sites[tgt_hub]
        if not hub.is_hub or hub.is_isolated:
            return 0.0

        for sid in sites_to_reroute:
            if sid in self.network:
                # Replace any isolated hub connections with the new hub
                new_connections = [n for n in self.network[sid] if not self.sites.get(n, SiteState("", 0, 0, TrialPhase.PHASE_II, 0, 0, False)).is_isolated]
                if tgt_hub not in new_connections:
                    new_connections.append(tgt_hub)
                self.network[sid] = new_connections
                # Also un-isolate the site if it was isolated due to hub closure
                # EXPLOIT FIX: Do not un-isolate the closed hub itself!
                if self.sites[sid].is_isolated and self.sites[sid].alert not in [AlertType.HUB_CLOSURE, "hub_closure"]:
                    self.sites[sid].is_isolated = False

        self.budget -= cost
        return cost

    def _do_scout(self, action: Action) -> float:
        """Pay to get accurate readings for a site (already accurate in simulation)."""
        # In real env this would unveil hidden state; here it's already transparent
        self.budget -= SCOUT_COST
        return SCOUT_COST

    # ------------------------------------------------------------------
    # Day advancement
    # ------------------------------------------------------------------

    def _advance_day(self) -> None:
        """Apply natural daily dynamics: demand consumption, temperature drift, thermal debt."""
        self.current_day += 1

        # Process shipping arrivals
        arrived = [ship for ship in self.in_transit if ship["arrival_day"] <= self.current_day]
        self.in_transit = [ship for ship in self.in_transit if ship["arrival_day"] > self.current_day]
        for ship in arrived:
            tgt = self.sites.get(ship["target_id"])
            if tgt:
                tgt.vials += ship["amount"]

        for site in self.sites.values():
            # Consume daily demand
            daily_consumed = min(site.vials, site.daily_demand)
            site.vials -= daily_consumed
            self.total_doses_delivered += daily_consumed
            self.total_demand += site.daily_demand

            # Temperature drift (random walk around base)
            if site.alert == AlertType.FRIDGE_MALFUNCTION or site.alert == "fridge_malfunction":
                # Temperature creeps toward ambient (+20°C drift per day)
                drift = self._rng.gauss(2.5, site.temp_noise_std)
            elif site.alert == AlertType.HURRICANE or site.alert == "hurricane":
                drift = self._rng.gauss(1.0, 3.0)
            else:
                drift = self._rng.gauss(0.0, site.temp_noise_std)

            site.temp_c += drift
            # Keep some realism: clamp to physical bounds
            site.temp_c = max(-90.0, min(25.0, site.temp_c))

            # Thermal debt accumulation
            temp_excess = max(0.0, site.temp_c - (TARGET_TEMP_C + TEMP_TOLERANCE))
            if site.alert == AlertType.FRIDGE_MALFUNCTION or site.alert == "fridge_malfunction":
                debt_delta = site.thermal_decay_rate * (1.0 + temp_excess / 20.0)
            else:
                debt_delta = site.thermal_decay_rate * max(0.0, temp_excess / 10.0)
            site.thermal_debt = min(1.0, site.thermal_debt + debt_delta)

    def _maybe_inject_random_alert(self) -> None:
        """Occasionally inject a minor random alert (only task1 and task2)."""
        if self.task_id == "task3":
            return  # task3 already has major alerts
        if self._rng.random() < 0.05:  # 5% chance per day
            sites = list(self.sites.values())
            target = self._rng.choice(sites)
            target.alert = AlertType.WEATHER_EVENT
            self.global_alerts = [
                GlobalAlert(
                    alert_type=AlertType.WEATHER_EVENT,
                    affected_sites=[target.site_id],
                    severity=self._rng.uniform(0.1, 0.4),
                    description=f"Minor weather event at {target.site_id}.",
                )
            ]
        else:
            # Clear any transient weather alerts
            self.global_alerts = [
                a for a in self.global_alerts
                if a.alert_type not in [AlertType.WEATHER_EVENT, "weather_event"]
            ]

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_psl(self) -> float:
        """
        Patient Service Level: weighted average of per-site fulfillment ratios.

        PSL = mean(min(1.0, site.vials / site.daily_demand) for each site)
        
        This provides continuous partial credit: a site at 50% supply gets 0.5 credit,
        not binary 0 or 1. Enables smooth gradient for per-step reward learning.
        """
        if not self.sites:
            return 0.0
        per_site_ratios = [
            min(1.0, s.vials / s.daily_demand) if s.daily_demand > 0 else 1.0
            for s in self.sites.values()
        ]
        return sum(per_site_ratios) / len(per_site_ratios)

    def _is_catastrophic_failure(self) -> bool:
        """End episode early if budget exhausted or all phase-III sites gone."""
        if self.budget < 0:
            return True
        phase3 = [s for s in self.sites.values() if s.trial_phase == TrialPhase.PHASE_III or s.trial_phase == "Phase III"]
        if phase3 and all(s.vials == 0 for s in phase3):
            return True
        return False

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self, done: bool, reward: float) -> Observation:
        protocol_snippet = self._get_protocol_snippet()
        return Observation(
            sites=[s.to_model() for s in self.sites.values()],
            current_day=self.current_day,
            remaining_budget=round(self.budget, 2),
            global_alerts=list(self.global_alerts),
            protocol_metadata=protocol_snippet,
            done=done,
            reward=reward,
            info={
                "psl": round(self._compute_psl(), 4),
                "total_doses_delivered": self.total_doses_delivered,
                "total_demand_so_far": self.total_demand,
                "shipping_cost_accumulated": round(self.shipping_cost_accumulated, 2),
                "task_id": self.task_id,
                "in_transit_shipments": len(self.in_transit),
                "in_transit_vials_total": sum(ship["amount"] for ship in self.in_transit)
            },
        )

    def _get_protocol_snippet(self) -> str:
        """Return a simulated protocol document excerpt."""
        if self.task_id == "task1":
            return (
                "PROTOCOL v2.3 §4.1: Standard replenishment cycle. "
                "Minimum stock level: 7-day supply per site. "
                "Temperature excursion threshold: >-20°C for >15 min is a reportable event."
            )
        elif self.task_id == "task2":
            malfunctioning = [s.site_id for s in self.sites.values() if s.alert in [AlertType.FRIDGE_MALFUNCTION, "fridge_malfunction"]]
            site_name = malfunctioning[0] if malfunctioning else "UNKNOWN_SITE"
            return (
                f"PROTOCOL v2.3 §6.7 AMENDMENT: Thermal excursion detected at {site_name}. "
                "All vials with Thermal Debt >0.7 are to be quarantined immediately. "
                "Phase III patients must be prioritised for re-supply. "
                "Contact QA within 24h of first excursion event."
            )
        else:
            phase3_sites = [s.site_id for s in self.sites.values() if s.trial_phase in [TrialPhase.PHASE_III, "Phase III"]]
            closed_hubs = [s.site_id for s in self.sites.values() if s.alert in [AlertType.HUB_CLOSURE, "hub_closure"]]
            
            p3_str = ", ".join(phase3_sites) if phase3_sites else "None"
            hub_str = ", ".join(closed_hubs) if closed_hubs else "None"
            return (
                "EMERGENCY PROTOCOL §9.1: Mass disruption event activated. "
                f"Phase III clinical sites ({p3_str}) have CRITICAL priority. "
                "Bio-hazard shipping pre-approved for Phase III sites. "
                "Budget exception: additional $100,000 approved for emergency logistics. "
                f"{hub_str} route suspended until further notice."
            )
