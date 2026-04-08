"""
OpenEnv-FragileChain: Server-side Environment

Implements the OpenEnv Environment base class with:
- reset()   → initial Observation
- step()    → Observation  (+ updates internal reward/done)
- state     → State property

This is the class hosted by the FastAPI server.
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, Optional
from uuid import uuid4

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.types import (
        Action as BaseAction,
        Observation as BaseObservation,
        State as BaseState,
    )
    from openenv.core.env_server.environment import Environment
except ImportError:
    # Fallback: define lightweight stubs so the code still runs without openenv-core
    class BaseAction:
        pass
    class BaseObservation:
        pass
    class BaseState:
        pass
    class Environment:
        pass

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import ColdChainEngine
from graders import get_grader, BaseGrader
from models import (
    Action,
    Observation,
    State,
)


class FragileChainEnvironment(Environment):
    """
    OpenEnv-compatible environment for the pharmaceutical cold-chain scenario.

    Supports three task modes (task_id: task1 / task2 / task3).
    Each episode runs for up to max_days simulation days.
    """

    def __init__(self, task_id: str = "task1", seed: Optional[int] = None, max_days: int = 30):
        self._task_id = task_id
        self._seed = seed
        self._max_days = max_days

        self._engine: ColdChainEngine = ColdChainEngine(
            task_id=task_id, seed=seed, max_days=max_days
        )
        self._grader: Optional[BaseGrader] = None
        self._state: State = State(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=task_id,
            current_day=0,
            max_days=max_days,
            total_reward=0.0,
            task_score=0.0,
        )

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Reset the environment and return the initial observation."""
        tid = task_id or self._task_id
        self._task_id = tid

        self._engine = ColdChainEngine(
            task_id=tid,
            seed=seed or self._seed,
            max_days=self._max_days,
        )
        self._engine.reset(task_id=tid)
        self._grader = get_grader(tid, self._engine)

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=tid,
            current_day=0,
            max_days=self._max_days,
            total_reward=0.0,
            task_score=0.0,
        )

        obs = self._engine.get_observation()
        return obs

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute an action and return the resulting observation."""
        obs, reward_obj, done = self._engine.step(action)

        self._state.step_count += 1
        self._state.current_day = self._engine.current_day
        self._state.total_reward += reward_obj.total

        # Record step in grader
        if self._grader is not None:
            self._grader.record_step()

        # Compute current task score
        if self._grader is not None:
            task_result = self._grader.compute_score()
            self._state.task_score = task_result.score
            obs.info["task_score"] = task_result.score
            obs.info["task_result"] = task_result.dict()

        return obs

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Async wrapper for step."""
        return self.step(action, timeout_s, **kwargs)

    @property
    def state(self) -> State:
        """Return current episode state."""
        return self._state

    # Alias for compatibility with some OpenEnv versions
    def get_state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Additional methods required by openenv-core http_server
    # ------------------------------------------------------------------

    async def reset_async(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """Async version of reset – delegates to synchronous reset."""
        return self.reset(seed=seed, episode_id=episode_id, task_id=task_id, **kwargs)

    def close(self) -> None:
        """Clean up resources. No-op for this environment."""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Return environment metadata for the /metadata endpoint."""
        return {
            "env_name": "fragilechain",
            "version": "0.1.0",
            "task_id": self._task_id,
            "max_days": self._max_days,
            "tasks": [
                {
                    "task_id": "task1",
                    "difficulty": "easy",
                    "description": "Steady State – maintain supply above demand for 30 days",
                    "max_days": 30,
                },
                {
                    "task_id": "task2",
                    "difficulty": "medium",
                    "description": "Thermal Anomaly – detect and mitigate fridge failure at SITE_ALPHA",
                    "max_days": 30,
                },
                {
                    "task_id": "task3",
                    "difficulty": "hard",
                    "description": "Black Swan – hub closure + hurricane, prioritise Phase III",
                    "max_days": 30,
                },
            ],
        }
