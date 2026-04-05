"""
OpenEnv-FragileChain: Baseline Inference Script

Runs a ReAct-style agent using the Groq API (via the OpenAI-compatible
client) against all 3 tasks and reports reproducible baseline scores.

Groq exposes an OpenAI-compatible endpoint, so the `openai` Python
package is used as-is — only the base_url and api_key change.

Usage:
    export GROQ_API_KEY=gsk_...
    python baseline.py

    # Optional overrides:
    export GROQ_MODEL=llama-3.3-70b-versatile   # default
    export GROQ_BASE_URL=https://api.groq.com/openai/v1  # default

Results are printed to stdout and saved to outputs/baseline_scores.json.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Env setup: add project root to path
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from engine import ColdChainEngine
from graders import get_grader, BaseGrader
from models import Action, ActionType, Observation, ShippingMode

# ---------------------------------------------------------------------------
# Groq client  (uses openai package pointed at Groq's compatible endpoint)
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai package not installed. Run: pip install openai")
    sys.exit(1)

# Groq credentials — set GROQ_API_KEY in your environment
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
# Recommended fast Groq models:
#   llama-3.3-70b-versatile  (default, best quality)
#   llama3-8b-8192           (fastest / cheapest)
#   mixtral-8x7b-32768       (long-context)
MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

MAX_STEPS_PER_TASK = 35  # slightly more than max_days to account for multi-action days
SEED = 42


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert pharmaceutical logistics coordinator managing a clinical trial cold-chain supply network.

Your goal is to keep drugs properly refrigerated and patients supplied across 7 clinical trial sites.
You must balance budget, temperature maintenance, and patient service levels.

AVAILABLE ACTIONS (return JSON exactly as shown):
1. Rebalance vials between sites:
   {"action_type": "rebalance", "source_id": "<SITE_ID>", "target_id": "<SITE_ID>", "amount": <int>, "mode": "standard|express|bio_hazard", "internal_thought": "<your reasoning>"}

2. Reroute sites through a different hub:
   {"action_type": "reroute_hub", "target_id": "<HUB_ID>", "affected_sites": ["<SITE_ID>", ...], "internal_thought": "<your reasoning>"}

3. Scout a site for accurate readings (costs $500):
   {"action_type": "scout", "source_id": "<SITE_ID>", "internal_thought": "<your reasoning>"}

4. Do nothing (advance time):
   {"action_type": "do_nothing", "internal_thought": "<your reasoning>"}

Site IDs: HUB_CENTRAL, HUB_COAST, SITE_ALPHA, SITE_BETA, SITE_GAMMA, SITE_DELTA, SITE_EPSILON
Hub IDs: HUB_CENTRAL, HUB_COAST

PRIORITIES:
- Phase III sites (SITE_ALPHA, SITE_DELTA): CRITICAL priority
- Phase II sites: HIGH priority  
- Phase I sites: STANDARD priority

THERMAL DEBT: 0.0 = pristine, 1.0 = fully degraded (quarantine immediately!)
BUDGET: Each shipping costs per vial: standard=$12, express=$35, bio_hazard=$80

Respond with ONLY a valid JSON action object. No other text.
""").strip()


def build_user_prompt(obs: Observation, day: int, task_id: str) -> str:
    """Format the current observation as a user prompt."""
    lines = [
        f"=== DAY {obs.current_day} | TASK: {task_id.upper()} | BUDGET: ${obs.remaining_budget:,.0f} ===",
        "",
        "PROTOCOL NOTE:",
        obs.protocol_metadata,
        "",
        "GLOBAL ALERTS:",
    ]
    if obs.global_alerts:
        for alert in obs.global_alerts:
            lines.append(f"  • [{alert.alert_type.upper()}] {alert.description} (severity={alert.severity:.2f})")
    else:
        lines.append("  None")
    lines.append("")
    lines.append("SITE STATUS:")
    for site in obs.sites:
        status_flags = []
        if site.is_hub:
            status_flags.append("HUB")
        if site.is_isolated:
            status_flags.append("ISOLATED")
        if site.alert and site.alert != "none":
            status_flags.append(f"ALERT:{site.alert.upper()}")
        flag_str = f" [{', '.join(status_flags)}]" if status_flags else ""

        debt_bar = "█" * int(site.avg_thermal_debt * 10) + "░" * (10 - int(site.avg_thermal_debt * 10))
        lines.append(
            f"  {site.site_id}{flag_str}: "
            f"Vials={site.vials_in_stock} | Demand={site.days_until_stockout}d | "
            f"Phase={site.trial_phase} | Temp={site.current_temp_c:.1f}°C | "
            f"ThermalDebt={site.avg_thermal_debt:.3f} [{debt_bar}]"
        )
    lines.append("")
    lines.append("What is your next action? Return ONLY the JSON action.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent_episode(
    client: OpenAI,
    task_id: str,
    seed: int = SEED,
    verbose: bool = True,
) -> Dict:
    """Run one full episode and return a results dict."""
    engine = ColdChainEngine(task_id=task_id, seed=seed, max_days=30)
    engine.reset(task_id=task_id)
    grader = get_grader(task_id, engine)

    obs = engine.get_observation()
    total_reward = 0.0
    step_count = 0
    errors = 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"Running Task: {task_id.upper()}")
        print(f"{'='*60}")

    for step in range(MAX_STEPS_PER_TASK):
        if obs.done:
            break

        user_prompt = build_user_prompt(obs, engine.current_day, task_id)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=256,
            )
            raw = response.choices[0].message.content.strip()

            # Parse JSON action
            action_dict = json.loads(raw)
            action = Action(**action_dict)

        except (json.JSONDecodeError, Exception) as e:
            # Default to do_nothing on parse errors
            action = Action(
                action_type=ActionType.DO_NOTHING,
                internal_thought=f"Parse error: {e}. Defaulting to wait.",
            )
            errors += 1

        obs, reward_obj, done = engine.step(action)
        grader.record_step()
        total_reward += reward_obj.total
        step_count += 1

        if verbose:
            print(
                f"  Day {engine.current_day:02d} | Action={action.action_type} "
                f"| R={reward_obj.total:+.4f} | PSL={obs.info.get('psl', '?'):.3f}"
            )

        if done:
            break

    task_result = grader.compute_score()

    if verbose:
        print(f"\nTask {task_id} complete:")
        print(f"  Final Score:      {task_result.score:.4f}")
        print(f"  Sci. Integrity:   {task_result.scientific_integrity:.4f}")
        print(f"  Total Reward:     {total_reward:.4f}")
        print(f"  Doses Delivered:  {task_result.doses_delivered}")
        print(f"  Thermal Debt:     {task_result.mean_thermal_debt:.4f}")
        print(f"  Budget Remaining: ${task_result.budget_remaining:,.2f}")
        print(f"  Parse Errors:     {errors}")

    return {
        "task_id": task_id,
        "score": task_result.score,
        "scientific_integrity": task_result.scientific_integrity,
        "total_reward": round(total_reward, 4),
        "doses_delivered": task_result.doses_delivered,
        "total_demand": task_result.total_demand,
        "mean_thermal_debt": task_result.mean_thermal_debt,
        "budget_remaining": task_result.budget_remaining,
        "steps": step_count,
        "parse_errors": errors,
        "breakdown": task_result.breakdown,
        "model": MODEL,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not GROQ_API_KEY:
        print("[ERROR] GROQ_API_KEY environment variable not set.")
        print("       Get a free key at https://console.groq.com/keys")
        sys.exit(1)

    # OpenAI client pointed at Groq's compatible endpoint — no other changes needed
    client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

    print(f"OpenEnv-FragileChain Baseline Runner  [Groq API]")
    print(f"Model: {MODEL} | Seed: {SEED}")
    print(f"Base URL: {GROQ_BASE_URL}")
    print(f"{'='*60}")

    results = []
    for task_id in ["task1", "task2", "task3"]:
        result = run_agent_episode(client, task_id, seed=SEED, verbose=True)
        results.append(result)
        time.sleep(1)  # Rate limiting

    # Summary table
    print(f"\n{'='*60}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Task':<10} {'Difficulty':<12} {'Score':<10} {'SI Score':<12} {'Total Reward':<15}")
    print("-" * 60)
    difficulties = {"task1": "Easy", "task2": "Medium", "task3": "Hard"}
    for r in results:
        print(
            f"{r['task_id']:<10} {difficulties[r['task_id']]:<12} "
            f"{r['score']:<10.4f} {r['scientific_integrity']:<12.4f} {r['total_reward']:<15.4f}"
        )

    avg = sum(r["score"] for r in results) / len(results)
    print(f"\nAverage Score: {avg:.4f}")

    # Save results
    output_dir = ROOT / "outputs" / "evals"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline_scores.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "metadata": {
                    "model": MODEL,
                    "seed": SEED,
                    "average_score": round(avg, 4),
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
