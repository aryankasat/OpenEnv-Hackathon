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

from fastapi import FastAPI, Query, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import json
import traceback
from typing import List, Optional, Any, Dict

from models import Action, Observation, State
from server.fragilechain_environment import FragileChainEnvironment

# ---------------------------------------------------------------------------
# Try to use openenv's create_app; fall back to manual FastAPI wiring
# ---------------------------------------------------------------------------

class ActionWrapperMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        # We only care about POST /step and /reset
        if scope["type"] == "http" and scope["method"] == "POST" and scope["path"] in ["/step", "/reset"]:
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
                data = json.loads(body)
                if scope["path"] == "/step":
                    if isinstance(data, dict) and "action" not in data and "action_type" in data:
                        # wrap it: {"action": {...}}
                        new_body = json.dumps({"action": data}).encode("utf-8")
                        messages = [{"type": "http.request", "body": new_body, "more_body": False}]
                elif scope["path"] == "/reset":
                    # If reset call has body but NO action key, it's probably standard OpenEnv
                    pass
            except Exception:
                pass
                
            # create a playback receiver
            async def new_receive():
                if messages:
                    return messages.pop(0)
                return {"type": "http.request", "body": b"", "more_body": False}
                
            return await self.app(scope, new_receive, send)
            
        return await self.app(scope, receive, send)

def add_standard_routes(app_obj, env_instance):
    """Register endpoints that should always be present."""
    
    @app_obj.get("/tasks")
    async def list_tasks():
        return {
            "tasks": [
                {
                    "task_id": "task1",
                    "difficulty": "easy",
                    "description": "Steady State – maintain supply above demand for 30 days",
                    "max_days": 30,
                    "has_grader": True,
                },
                {
                    "task_id": "task2",
                    "difficulty": "medium",
                    "description": "Thermal Anomaly – detect and mitigate fridge failure at SITE_ALPHA",
                    "max_days": 30,
                    "has_grader": True,
                },
                {
                    "task_id": "task3",
                    "difficulty": "hard",
                    "description": "Black Swan – hub closure + hurricane, prioritise Phase III",
                    "max_days": 30,
                    "has_grader": True,
                },
            ]
        }

    @app_obj.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        print(f"CRITICAL ERROR: {exc}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error", "error": str(exc), "traceback": traceback.format_exc()}
        )

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

    # Override OpenAPI to show the simplified flat payload example
    def wrapped_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        # Build the initial schema
        from fastapi.openapi.utils import get_openapi
        schema = get_openapi(
            title=app.title,
            version=app.version,
            openapi_version=app.openapi_version,
            description=app.description,
            routes=app.routes,
        )
        
        # Inject the custom example into the /step endpoint
        if "/step" in schema.get("paths", {}):
            try:
                # Find the request body content
                step_post = schema["paths"]["/step"]["post"]
                content = step_post["requestBody"]["content"]["application/json"]
                
                # Update the example to the user's requested format
                content["example"] = {
                    "action_type": "do_nothing",
                    "internal_thought": "testing step"
                }
                
                # Also update the schema itself so it shows the flat keys
                # instead of the nested 'action' structure
                content["schema"] = {
                    "title": "Action",
                    "type": "object",
                    "properties": {
                        "action_type": {"title": "Action Type", "type": "string"},
                        "internal_thought": {"title": "Internal Thought", "type": "string"},
                        "source_id": {"type": "string", "nullable": True},
                        "target_id": {"type": "string", "nullable": True},
                        "amount": {"type": "integer", "nullable": True}
                    },
                    "required": ["action_type"]
                }
            except Exception:
                pass
                
        app.openapi_schema = schema
        return app.openapi_schema
    
    app.openapi = wrapped_openapi
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
    _env.reset() # Initial reset to populate sites

    add_standard_routes(app, _env)

    @app.get("/health")
    async def health():
        return {"status": "ok", "env": "fragilechain"}

    @app.post("/reset")
    async def reset(
        task_id: Optional[str] = Query(None, enum=["task1", "task2", "task3"]),
        seed: Optional[int] = Query(None),
        body: dict = Body(None)
    ):
        # Merge query and body params
        tid = (body.get("task_id") if body else None) or task_id or "task1"
        s = (body.get("seed") if body else None) or seed
        e_id = (body.get("episode_id") if body else None)
        
        obs = _env.reset(task_id=tid, seed=s, episode_id=e_id)
        return obs.dict()

    @app.post("/step")
    async def step(action: Action):
        obs = _env.step(action)
        return obs.dict()

    @app.get("/state")
    async def state():
        return _env.state.dict()

    @app.get("/metadata")
    async def metadata():
        return _env.get_metadata()




@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {
                "task_id": "task1",
                "difficulty": "easy",
                "description": "Steady State – maintain supply above demand for 30 days",
                "max_days": 30,
                "has_grader": True,
            },
            {
                "task_id": "task2",
                "difficulty": "medium",
                "description": "Thermal Anomaly – detect and mitigate fridge failure at SITE_ALPHA",
                "max_days": 30,
                "has_grader": True,
            },
            {
                "task_id": "task3",
                "difficulty": "hard",
                "description": "Black Swan – hub closure + hurricane, prioritise Phase III",
                "max_days": 30,
                "has_grader": True,
            },
        ]
    }


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
