"""
OpenEnv-FragileChain: EnvClient

Provides a typed client for connecting to a running FragileChain server.
Requires openenv-core to be installed.

Usage (async):
    async with FragileChainEnv(base_url="http://localhost:8000") as env:
        obs = await env.reset(task_id="task1")
        result = await env.step(Action(action_type="do_nothing"))

Usage (sync):
    with FragileChainEnv(base_url="http://localhost:8000").sync() as env:
        obs = env.reset(task_id="task1")
        result = env.step(Action(action_type="do_nothing"))
"""

from __future__ import annotations

from typing import Optional
from models import Action, Observation, State

try:
    from openenv.core.env_client import EnvClient

    class FragileChainEnv(EnvClient):
        """
        Client for the FragileChain environment.

        Wraps EnvClient with typed Action/Observation interfaces.
        """

        def __init__(self, base_url: str = "http://localhost:8000", **kwargs):
            super().__init__(
                base_url=base_url,
                action_type=Action,
                observation_type=Observation,
                **kwargs,
            )

        async def reset(
            self,
            task_id: str = "task1",
            seed: Optional[int] = None,
            **kwargs,
        ) -> Observation:
            return await super().reset(task_id=task_id, seed=seed, **kwargs)

        async def step(self, action: Action, **kwargs) -> Observation:
            return await super().step(action, **kwargs)

        async def state(self) -> State:
            return await super().state()

except ImportError:
    # Fallback: sync HTTP client without openenv-core
    import requests

    class FragileChainEnv:  # type: ignore[no-redef]
        """
        Lightweight sync HTTP client for FragileChain (no openenv-core required).
        """

        def __init__(self, base_url: str = "http://localhost:8000"):
            self.base_url = base_url.rstrip("/")
            self._session = requests.Session()

        def reset(self, task_id: str = "task1", seed: Optional[int] = None) -> Observation:
            params = {"task_id": task_id}
            if seed is not None:
                params["seed"] = seed
            resp = self._session.post(f"{self.base_url}/reset", params=params)
            resp.raise_for_status()
            return Observation(**resp.json())

        def step(self, action: Action) -> Observation:
            resp = self._session.post(
                f"{self.base_url}/step",
                json=action.dict(),
            )
            resp.raise_for_status()
            return Observation(**resp.json())

        def state(self) -> State:
            resp = self._session.get(f"{self.base_url}/state")
            resp.raise_for_status()
            return State(**resp.json())

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self._session.close()

        def sync(self):
            return self
