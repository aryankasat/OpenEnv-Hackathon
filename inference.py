"""
OpenEnv-FragileChain — Inference Script
========================================
Mandatory submission script.

Environment variables (REQUIRED):
    HF_TOKEN           Your Hugging Face / API key  (also accepts API_KEY)

Environment variables (with defaults):
    API_BASE_URL       LLM endpoint
                       default: https://api.groq.com/openai/v1
    MODEL_NAME         Model identifier
                       default: llama-3.3-70b-versatile
    LOCAL_IMAGE_NAME   Docker image name when using from_docker_image()
                       (not used in direct-engine mode)
    FRAGILECHAIN_TASK  Which task to run: task1 | task2 | task3
                       default: task1
    FRAGILECHAIN_SEED  RNG seed for reproducibility
                       default: 42

STDOUT FORMAT (emitted exactly as specified):
    [START] task=<task_name> env=fragilechain model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Usage:
    export HF_TOKEN=gsk_...           # or API_KEY=...
    python inference.py               # runs task1 by default
    FRAGILECHAIN_TASK=task3 python inference.py
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

# ── project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from engine import ColdChainEngine
from graders import get_grader
from models import Action, ActionType, Observation, ShippingMode

# ── OpenAI client (works with Groq, HF router, or any compatible endpoint) ───
try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai not installed. Run: pip install openai", flush=True)
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")
TASK_NAME    = os.getenv("FRAGILECHAIN_TASK",  "task1")
BENCHMARK    = "fragilechain"
SEED         = int(os.getenv("FRAGILECHAIN_SEED", "42"))
MAX_STEPS    = 35           # ≥ max_days to allow multi-action days
SUCCESS_SCORE_THRESHOLD = 0.4   # [0, 1] — any score above counts as success

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
You are an expert pharmaceutical logistics coordinator managing a clinical trial cold-chain supply network.
Your goal: keep drugs cold and patients supplied across 7 clinical trial sites.

AVAILABLE ACTIONS — respond with ONLY a valid JSON object, no other text:
1. Move vials between sites:
   {"action_type":"rebalance","source_id":"<SITE>","target_id":"<SITE>","amount":<int>,"mode":"standard|express|bio_hazard","internal_thought":"<reason>"}

2. Reroute sites through a different hub:
   {"action_type":"reroute_hub","target_id":"<HUB>","affected_sites":["<SITE>",...],"internal_thought":"<reason>"}

3. Scout a site for detailed readings ($500):
   {"action_type":"scout","source_id":"<SITE>","internal_thought":"<reason>"}

4. Do nothing (let one day pass):
   {"action_type":"do_nothing","internal_thought":"<reason>"}

Sites: HUB_CENTRAL, HUB_COAST, SITE_ALPHA, SITE_BETA, SITE_GAMMA, SITE_DELTA, SITE_EPSILON
Hubs:  HUB_CENTRAL, HUB_COAST

Priority: Phase III (SITE_ALPHA, SITE_DELTA) = CRITICAL > Phase II = HIGH > Phase I = STANDARD
Thermal Debt: 0.0=pristine → 1.0=fully degraded (act immediately if >0.5!)
Shipping cost per vial: standard=$12, express=$35, bio_hazard=$80

Return ONLY the JSON action. No prose.
""").strip()


def _build_obs_prompt(obs: Observation, step: int) -> str:
    lines = [
        f"=== DAY {obs.current_day} | STEP {step} | BUDGET ${obs.remaining_budget:,.0f} ===",
        "",
        "PROTOCOL:",
        obs.protocol_metadata,
        "",
        "GLOBAL ALERTS:",
    ]
    if obs.global_alerts:
        for a in obs.global_alerts:
            lines.append(f"  [{a.alert_type.upper()}] {a.description} (severity={a.severity:.2f})")
    else:
        lines.append("  None")

    lines += ["", "SITES:"]
    for s in obs.sites:
        flags = []
        if s.is_hub:      flags.append("HUB")
        if s.is_isolated: flags.append("ISOLATED")
        if s.alert and s.alert != "none": flags.append(f"ALERT:{s.alert.upper()}")
        flag_str = f" [{','.join(flags)}]" if flags else ""
        bar  = "█" * int(s.avg_thermal_debt * 10) + "░" * (10 - int(s.avg_thermal_debt * 10))
        lines.append(
            f"  {s.site_id}{flag_str}: vials={s.vials_in_stock} "
            f"stockout={s.days_until_stockout}d phase={s.trial_phase} "
            f"temp={s.current_temp_c:.1f}°C debt={s.avg_thermal_debt:.3f}[{bar}]"
        )

    lines += ["", "Return your JSON action now."]
    return "\n".join(lines)


# ── Structured logging helpers ────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = json.dumps(error) if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM call ─────────────────────────────────────────────────────────────────

def _call_llm(client: OpenAI, prompt: str) -> str:
    """Call the LLM and return raw text, or empty string on failure."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return ""


def _parse_action(raw: str) -> tuple[Action, Optional[str]]:
    """
    Parse the LLM response into an Action.
    Returns (action, error_string_or_None).
    Strips markdown fences if present.
    """
    # Strip ```json ... ``` fences
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        ).strip()

    try:
        d = json.loads(text)
        return Action(**d), None
    except Exception as e:
        return Action(
            action_type=ActionType.DO_NOTHING,
            internal_thought=f"[parse-error] {e}",
        ), str(e)


# ── Main episode loop ─────────────────────────────────────────────────────────

def run_episode() -> None:
    if not API_KEY:
        print("[ERROR] Set HF_TOKEN or API_KEY environment variable.", flush=True)
        sys.exit(1)

    client  = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    engine  = ColdChainEngine(task_id=TASK_NAME, seed=SEED, max_days=30)
    engine.reset(task_id=TASK_NAME)
    grader  = get_grader(TASK_NAME, engine)

    obs          = engine.get_observation()
    rewards: List[float] = []
    steps_taken  = 0
    score        = 0.0
    success      = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            prompt   = _build_obs_prompt(obs, step)
            raw      = _call_llm(client, prompt)
            action, parse_err = _parse_action(raw)

            obs, reward_obj, done = engine.step(action)
            grader.record_step()

            reward = reward_obj.total
            rewards.append(reward)
            steps_taken = step

            # action repr — compact single-line JSON
            action_repr = json.dumps({
                "action_type": action.action_type
                    if isinstance(action.action_type, str) else action.action_type.value,
                **({"source_id": action.source_id} if action.source_id else {}),
                **({"target_id": action.target_id} if action.target_id else {}),
                **({"amount": action.amount}        if action.amount      else {}),
            }, separators=(",", ":"))

            log_step(
                step   = step,
                action = action_repr,
                reward = reward,
                done   = done,
                error  = parse_err,
            )

            if done:
                break

        # ── Final grader score ────────────────────────────────────────────
        task_result = grader.compute_score()
        score   = task_result.score                     # already in [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    run_episode()
