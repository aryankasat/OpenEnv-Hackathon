"""
OpenEnv-FragileChain: FastAPI Server App

Creates the HTTP/WebSocket server for the cold-chain environment.
Compatible with openenv-core create_app() factory.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
    OR
    python server/app.py
"""

import os
import sys

# Ensure parent directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from models import Action, Observation, State
from server.fragilechain_environment import FragileChainEnvironment

# ---------------------------------------------------------------------------
# Try to use openenv's create_app; fall back to manual FastAPI wiring
# ---------------------------------------------------------------------------

class ActionWrapperMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] == "/step" and scope["method"] == "POST":
            # intercept body
            body = b""
            more_body = True
            messages = []
            while more_body:
                message = await receive()
                messages.append(message)
                body += message.get("body", b"")
                more_body = message.get("more_body", False)
            
            try:
                import json
                data = json.loads(body)
                if isinstance(data, dict) and "action" not in data and "action_type" in data:
                    # wrap it
                    new_body = json.dumps({"action": data}).encode("utf-8")
                    messages = [{"type": "http.request", "body": new_body, "more_body": False}]
            except Exception:
                pass
                
            # create a playback receiver
            async def new_receive():
                if messages:
                    return messages.pop(0)
                return {"type": "http.request", "body": b"", "more_body": False}
                
            return await self.app(scope, new_receive, send)
            
        return await self.app(scope, receive, send)

try:
    from openenv.core.env_server.http_server import create_app
    app = create_app(
        FragileChainEnvironment,
        Action,
        Observation,
        env_name="fragilechain",
    )
    app.add_middleware(ActionWrapperMiddleware)
    _using_openenv_core = True
except ImportError:
    _using_openenv_core = False
    app = FastAPI(
        title="OpenEnv-FragileChain",
        description=(
            "Pharmaceutical cold-chain logistics environment for AI agents. "
            "Simulates clinical trial supply chain management under thermal stress "
            "and mass-disruption events."
        ),
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Singleton environment (stateless endpoint style)
    _env = FragileChainEnvironment()

    @app.get("/health")
    async def health():
        return {"status": "ok", "env": "fragilechain"}

    @app.post("/reset")
    async def reset(
        task_id: str = Query("task1", enum=["task1", "task2", "task3"]),
        seed: int = Query(None),
    ):
        obs = _env.reset(task_id=task_id, seed=seed)
        return obs.dict()

    @app.post("/step")
    async def step(action: Action):
        obs = _env.step(action)
        return obs.dict()

    @app.get("/state")
    async def state():
        return _env.state.dict()




@app.get("/tasks")
async def list_tasks():
    return {
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
        ]
    }


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
