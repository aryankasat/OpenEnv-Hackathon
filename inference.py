"""
OpenEnv-FragileChain — Inference Script
========================================
Mandatory submission script for the OpenEnv Hackathon.

Runs a full episode against the FragileChain environment server using an LLM
to decide logistics actions. Prints structured logs consumed by the judge.

Required environment variables:
    HF_TOKEN           Your Hugging Face / API key  (or set API_KEY)

Optional environment variables:
    ENV_URL            Environment server URL (defaults to http://localhost:8000)
    API_BASE_URL       LLM endpoint (defaults to https://api.groq.com/openai/v1)
    MODEL_NAME         Model identifier (defaults to llama-3.3-70b-versatile)

STDOUT FORMAT (machine-parsed by judge):
    [START] task=<task_name> env=fragilechain model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import argparse
import os
import sys
import json
import textwrap
from typing import List, Optional
from pathlib import Path

# ── Ensure project root is on path ───────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Import environment client ────────────────────────────────────────────────
from client import FragileChainEnv
from models import Action, ActionType, Observation

# ── Import OpenAI client ─────────────────────────────────────────────────────
try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai not installed. Run: pip install openai", flush=True)
    sys.exit(1)

# ── Configuration Defaults ───────────────────────────────────────────────────
DEFAULT_ENV_URL      = os.getenv("ENV_URL", "http://localhost:8000")
DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
DEFAULT_MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
DEFAULT_HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

BENCHMARK    = "fragilechain"
MAX_STEPS    = 35
SUCCESS_THRESHOLD = 0.4


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


# ── Mandatory logging helpers (judge-parsed format) ───────────────────────────
def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    error_val = json.dumps(error) if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ── LLM call ──────────────────────────────────────────────────────────────────
def call_llm(client: OpenAI, model: str, prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", file=sys.stderr, flush=True)
        return ""


def parse_action(raw: str) -> tuple[Action, Optional[str]]:
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```")).strip()
    try:
        d = json.loads(text)
        return Action(**d), None
    except Exception as e:
        return Action(action_type=ActionType.DO_NOTHING, internal_thought=f"[parse-error] {e}"), str(e)


# ── Episode runner ────────────────────────────────────────────────────────────
def run_episode(task_id: str, env_url: str, api_key: str, api_base: str, model_name: str, seed: int):
    client = OpenAI(api_key=api_key, base_url=api_base)
    
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env_name=BENCHMARK, model=model_name)

    try:
        with FragileChainEnv(base_url=env_url).sync() as env:
            obs = env.reset(task_id=task_id, seed=seed)
            
            for step in range(1, MAX_STEPS + 1):
                if obs.done:
                    break
                
                prompt = _build_obs_prompt(obs, step)
                raw_action = call_llm(client, model_name, prompt)
                action, parse_err = parse_action(raw_action)
                
                obs = env.step(action)
                reward = obs.reward
                rewards.append(reward)
                steps_taken = step

                action_repr = json.dumps({
                    "action_type": action.action_type if isinstance(action.action_type, str) else action.action_type.value,
                    **({"source_id": action.source_id} if action.source_id else {}),
                    **({"target_id": action.target_id} if action.target_id else {}),
                    **({"amount": action.amount} if action.amount else {}),
                }, separators=(",", ":"))

                log_step(step=step, action=action_repr, reward=reward, done=obs.done, error=parse_err)

                if obs.done:
                    break
            
            # Get final score from environment state
            try:
                state_obj = env.state()
                score = state_obj.task_score
            except:
                # Fallback to mean reward if state() fails
                score = sum(rewards) / len(rewards) if rewards else 0.0
            
            success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode {task_id} failed: {e}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FragileChain Inference Script")
    parser.add_argument("--task", type=str, default="all", help="Task ID (task1|task2|task3|all)")
    parser.add_argument("--url", type=str, default=DEFAULT_ENV_URL, help="Environment server URL")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="LLM model name")
    parser.add_argument("--api-base", type=str, default=DEFAULT_API_BASE_URL, help="LLM API base URL")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    api_key = DEFAULT_HF_TOKEN
    if not api_key:
        print("[ERROR] HF_TOKEN or API_KEY environment variable required.", file=sys.stderr)
        sys.exit(1)

    tasks = ["task1", "task2", "task3"] if args.task == "all" else [args.task]
    
    for t in tasks:
        run_episode(
            task_id=t,
            env_url=args.url,
            api_key=api_key,
            api_base=args.api_base,
            model_name=args.model,
            seed=args.seed
        )
