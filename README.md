# 🧊 OpenEnv-FragileChain

> A pharmaceutical cold-chain logistics environment for AI agents — OpenEnv hackathon submission

[![OpenEnv](https://img.shields.io/badge/openenv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 🧬 Environment Description & Motivation

The pharmaceutical industry wastes an estimated **$35 billion per year** in drug spoilage. A significant fraction occurs in clinical supply chains, where biologics must be kept between −20 °C and −80 °C while being shipped to geographically distributed sites.

**FragileChain** simulates this. An AI agent controls a network of 7 clinical sites (2 distribution hubs + 5 patient-facing sites) and must:

- Maintain vial stock above daily patient demand at every site
- Keep drugs within their thermal tolerance window (tracking **Cumulative Thermal Debt**)
- Respond to disruption events — fridge failures, hub closures, hurricanes
- Prioritise **Phase III** patients when resources are scarce
- Stay within a logistics budget

This environment tests **Resource Allocation under Semantic Uncertainty**: the agent receives both structured sensor data and unstructured protocol document excerpts and must interpret them together to act well.

---

## 🗺️ Site Network

```
                    HUB_CENTRAL
                   /     |     \
          SITE_ALPHA SITE_BETA SITE_GAMMA
                        |
                    HUB_COAST
                   /         \
            SITE_DELTA    SITE_EPSILON
```

| Site ID          | Type | Phase               | Notes                                 |
| ---------------- | ---- | ------------------- | ------------------------------------- |
| `HUB_CENTRAL`  | Hub  | Phase II            | Main distribution hub — 500 vials    |
| `HUB_COAST`    | Hub  | Phase II            | Coastal distribution hub — 300 vials |
| `SITE_ALPHA`   | Site | **Phase III** | Critical priority, high patient load  |
| `SITE_BETA`    | Site | Phase I             | Standard priority                     |
| `SITE_GAMMA`   | Site | Phase II            | High priority                         |
| `SITE_DELTA`   | Site | **Phase III** | Critical priority, coastal            |
| `SITE_EPSILON` | Site | Phase I             | Standard priority, coastal            |

---

## 🔭 Observation Space

Each `step()` / `reset()` returns an `Observation`:

| Field                 | Type                  | Description                        |
| --------------------- | --------------------- | ---------------------------------- |
| `sites`             | `List[SiteStatus]`  | Per-site status snapshot (7 items) |
| `current_day`       | `int`               | Current simulation day (0–30)     |
| `remaining_budget`  | `float`             | Remaining logistics budget in USD  |
| `global_alerts`     | `List[GlobalAlert]` | Active disruption events           |
| `protocol_metadata` | `str`               | Unstructured protocol excerpt      |
| `done`              | `bool`              | Episode termination flag           |
| `reward`            | `float`             | Scalar reward from last action     |

### SiteStatus fields

| Field                   | Type      | Range          | Description                                |
| ----------------------- | --------- | -------------- | ------------------------------------------ |
| `site_id`             | `str`   | —             | Unique identifier                          |
| `vials_in_stock`      | `int`   | ≥ 0           | Current vial inventory                     |
| `patient_load`        | `int`   | ≥ 0           | Number of active patients                  |
| `phase`               | `str`   | Phase I/II/III | Priority weight                            |
| `current_temp_c`      | `float` | −90 to +25    | Storage temperature in °C                 |
| `avg_thermal_debt`    | `float` | 0.0–1.0       | Cumulative thermal stress (1.0 = unusable) |
| `days_until_stockout` | `int?`  | ≥ 0           | Days until vials run out                   |
| `is_hub`              | `bool`  | —             | Distribution hub flag                      |
| `is_isolated`         | `bool`  | —             | Cut off from normal routes                 |
| `alert`               | `str`   | enum           | Active alert at this site                  |

---

## 🎮 Action Space

Each step the agent sends one `Action`:

| Field                | Type           | Description                                                          |
| -------------------- | -------------- | -------------------------------------------------------------------- |
| `action_type`      | `str`        | `rebalance` · `reroute_hub` · `scout` · `do_nothing`      |
| `source_id`        | `str?`       | Source site (for `rebalance`)                                      |
| `target_id`        | `str?`       | Target site or hub                                                   |
| `amount`           | `int?`       | Vials to move                                                        |
| `mode`             | `str`        | `standard` ($12/vial) · `express` ($35) · `bio_hazard` ($80) |
| `affected_sites`   | `List[str]?` | Sites to reroute (for `reroute_hub`)                               |
| `internal_thought` | `str`        | Agent reasoning (logged, not used in sim)                            |

### Semantics

| Action          | Effect                                                                      | Cost                |
| --------------- | --------------------------------------------------------------------------- | ------------------- |
| `rebalance`   | Transfer `amount` vials from `source_id` to `target_id`               | Amount × mode rate |
| `reroute_hub` | Re-connect `affected_sites` through `target_id` hub; resolves isolation | $2,000 fixed        |
| `scout`       | Reveal detailed sensor data for a site                                      | $500                |
| `do_nothing`  | Advance one day                                                             | $0                  |

---

## 📏 Unified Reward Function

**All tasks use the same per-step reward function:**

```
R_t = 0.4 × ΔPSL
    − 0.2 × (FiscalCost / MaxBudget)
    − 0.2 × (ShippingCost / MaxBudget × 0.01)
    − 0.15 × MeanThermalDebt
    − 0.05 × StockoutCount
```

Where:
- **ΔPSL** = Patient Service Level delta (change from previous step)
- **PSL** = **weighted average** of per-site fulfillment ratios: `mean(min(1.0, site.vials / site.daily_demand))`
  - This provides **continuous partial credit**: a site at 50% supply gets 0.5 contribution, not binary 0 or 1
  - Enables smooth learning gradient from sparse initial policies to optimal ones
- **FiscalCost** = cost of actions (rebalance, reroute, scout)
- **ShippingCost** = cost of shipping mode (standard, express, bio_hazard)
- **MeanThermalDebt** = average thermal debt across all sites
- **StockoutCount** = number of sites with `vials ≤ 0`

**Clipped to [−1.0, 1.0]** to prevent extreme outliers.

This **unified function** provides continuous learning signal across all three tasks:
- Positive reward for improving PSL and staying cool (via partial site fulfillment)
- Gradual penalties for budget burn, thermal stress, and stockouts
- Works identically in task1, task2, task3
- Enables agents to see progress from 0 → 1 supply via smooth ΔPSL signal

### Task-Specific Final Scoring (Graders Only)

Task-specific nuances appear **only in the final TaskResult score** (computed after episode end). Per-step rewards are identical:

| Task | Final Score Formula | Focus |
|------|---------|---------|
| **Task 1** | `(per_site_ratio) × (1 − 0.5·MeanDebt) × BudgetFactor` | Service consistency (now with per-site partial credit) |
| **Task 2** | `0.4·SpeedScore + 0.4·VialsSavedRatio + 0.2·ThermalIntegrity − ContinuousPenalty` | Mitigation quality (penalty now proportional not cliff-based) |
| **Task 3** | `SI − Phase3StockoutPenalty + RerouteBonus` where SI uses actual delivery | Scientific Integrity (uses real vs. estimated doses) |

---

## 📋 Tasks

All tasks use the **same unified per-step reward function** (above). Task differences are in the **final grader scoring only:**

### Task 1 — Steady State *(Easy)*

Maintain supply above daily demand at all 7 sites for 30 days under normal conditions.

**Final Grader Score:**
```
Score = (per_site_ratio) × (1 − 0.5·MeanDebt) × (0.8 + 0.2·BudgetFraction)
```

### Task 2 — Thermal Anomaly *(Medium)*

SITE_ALPHA's fridge is leaking. `thermal_debt` rises at 8 %/day. Evacuate stock before efficacy hits zero.

**Final Grader Score:**
```
Score = 0.4·SpeedScore + 0.4·VialsSavedRatio + 0.2·ThermalIntegrity − ContinuousPenalty
```

### Task 3 — Black Swan *(Hard)*

Simultaneous HUB_COAST closure + hurricane at SITE_DELTA/EPSILON. Must reroute and protect Phase III patients.

**Final Grader Score:**
```
SI    = (Σ doses·priority_weight / total_demand) × (1 − mean_thermal_debt)
Score = SI − Phase3StockoutPenalty + RerouteBonus
```

---

## 🔧 Critical Fixes & Improvements

The **per-step reward function is unified and identical across all tasks.** The following fixes enhance both per-step learning and final grader scoring:

### Fix 1: Partial Credit for Service Ratio (Per-Step + Final Grader)

**Problem:** Original binary all-or-nothing rule: only gave reward if ALL 7 sites met demand. A single shortfall meant zero credit for that step, making reward extremely sparse and hard to learn from.

**Solution:** 
- **Per-step reward:** Update `_compute_psl()` to use weighted average of per-site fulfillment:
  ```python
  PSL = mean(min(1.0, site.vials / site.daily_demand) for each site)
  ```
  A site at 50% supply now contributes 0.5 to PSL, not binary 0 or 1.

- **Final grader (Task 1):** Calculate per-site partial ratio in final score:
  ```python
  per_site_ratio = sum(min(1.0, s.vials / s.daily_demand) for s in sites) / len(sites)
  ```

**Impact:** Continuous dense reward signal at every step; agents see progress from 0 → 1 supply smoothly rather than waiting for all-or-nothing flip.

---

### Fix 2: Continuous Compromise Penalty (Final Grader Only)

**Problem:** Hard cliff at `thermal_debt >= 1.0` in grader final score: either 0.0 or −0.3 penalty, discontinuous.

**Solution:** In Task2Grader's final scoring, use proportional penalty function:
```python
compromise_penalty = min(0.3, max(0.0, alpha_debt - 1.0) * 0.05) if alpha_debt > 1.0 else 0.0
```

Penalty grows smoothly: at debt=1.0 → 0, debt=1.5 → −0.025, debt=7.0 → −0.3 (capped).

**Impact:** Final grader score has smooth gradient. Per-step reward unchanged (unified).

---

### Fix 3: Actual vs. Estimated Delivery (Final Grader Only)

**Problem:** SI calculation in grader used estimated delivery (daily_demand × current_day), ignoring actual stockouts. Final score nearly same whether agent performed well or randomly.

**Solution:** In Task3Grader's `_on_record_step()`, track actual cumulative doses per site:
```python
for site in self.engine.sites.values():
    actual_delivered = min(site.vials, site.daily_demand)
    self._per_site_delivered[site.site_id] += actual_delivered
```

Then use real values in final SI computation:
```python
weighted_delivered = sum(
    self._per_site_delivered[site.site_id] * site.priority_weight()
    for site in self.engine.sites.values()
)
```

**Impact:** Final grader score now reflects actual patient-days served. Per-step reward unchanged (unified).

---

## 📊 Baseline Scores

Measured with `llama-3.3-70b-versatile` via Groq API, seed=42:

| Task              | Difficulty | Score          | SI Score | Notes                                             |
| ----------------- | ---------- | -------------- | -------- | ------------------------------------------------- |
| task1             | Easy       | **0.50** | 0.50     | Agent waits, then rebalances too late             |
| task2             | Medium     | **0.09** | 0.00     | Over-scouts, misses the thermal evacuation window |
| task3             | Hard       | **0.67** | 0.79     | Phase III directive followed well                 |
| **Average** |            | **0.42** |          |                                                   |

Scores saved to `outputs/evals/baseline_scores.json` after running `baseline.py`.

---

## 🚀 Setup & Usage

### Option 1 — Local Python (fastest)

```bash
git clone https://github.com/YOUR_ORG/OpenEnv-Railway-Mass-Disruption
cd OpenEnv-Railway-Mass-Disruption

python3 -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn pydantic requests openai

# Smoke-test graders
python3 graders.py

# Start the API server
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Option 2 — Docker

```bash
docker build -t fragilechain .
docker run -p 8000:8000 fragilechain

# Health check
curl http://localhost:8000/health
```

### Option 3 — Python client

```python
from client import FragileChainEnv
from models import Action, ActionType

with FragileChainEnv(base_url="http://localhost:8000").sync() as env:
    obs = env.reset(task_id="task2")
    while not obs.done:
        action = Action(action_type=ActionType.DO_NOTHING)
        obs = env.step(action)
        print(f"Day {obs.current_day}: reward={obs.reward:.3f}")
```

---

## 🤖 Inference Script (`inference.py`)

The mandatory submission script. Uses the OpenAI client pointed at any compatible LLM endpoint (Groq, HF Router, local Ollama, etc.).

### Environment variables

| Variable                   | Required | Default                            | Description                                 |
| -------------------------- | -------- | ---------------------------------- | ------------------------------------------- |
| `HF_TOKEN` / `API_KEY` | ✅       | —                                 | API key                                     |
| `API_BASE_URL`           | —       | `https://api.groq.com/openai/v1` | LLM endpoint                                |
| `MODEL_NAME`             | —       | `llama-3.3-70b-versatile`        | Model identifier                            |
| `FRAGILECHAIN_TASK`      | —       | `task1`                          | Task to run (`task1`/`task2`/`task3`) |
| `FRAGILECHAIN_SEED`      | —       | `42`                             | RNG seed                                    |

### Run

```bash
# Groq (recommended)
export HF_TOKEN=gsk_...
python3 inference.py

# HuggingFace router
export HF_TOKEN=hf_...
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python3 inference.py

# Run a specific task
FRAGILECHAIN_TASK=task3 python3 inference.py
```

### Stdout format

```
[START] task=task1 env=fragilechain model=llama-3.3-70b-versatile
[STEP]  step=1 action={"action_type":"do_nothing"} reward=0.00 done=false error=null
[STEP]  step=2 action={"action_type":"rebalance","source_id":"HUB_CENTRAL","target_id":"SITE_ALPHA","amount":30} reward=-0.01 done=false error="parsing failed"
...
[END]   success=true steps=14 score=0.500 rewards=0.00,-0.01,...
```

### Inference Script Fixes

Two critical fixes ensure robust execution:

1. **Error Field Handling** — Error messages are now properly JSON-quoted in `[STEP]` output:
   ```python
   err = json.dumps(error) if error else "null"
   ```
   This ensures valid JSON even if the error message contains special characters.

2. **Resource Cleanup** — Engine is properly closed in finally block:
   ```python
   finally:
       engine.close()
       log_end(...)
   ```
   Prevents resource leaks and ensures proper cleanup even if exceptions occur.

---

## ✅ Pre-Submission Validation

```bash
# Install prerequisite
pip install openenv-core

# Run all 3 checks (HF Space liveness + Docker build + openenv validate)
./scripts/validate-submission.sh https://YOUR-SPACE.hf.space
```

The script runs:

1. **Ping** — `POST /reset` returns 200 on your live HF Space
2. **Docker** — `docker build` succeeds within 600 s
3. **Spec** — `openenv validate` passes

---

## ⚡ Groq Baseline (`baseline.py`)

```bash
export GROQ_API_KEY=gsk_...
# optional: export GROQ_MODEL=llama3-8b-8192

python3 baseline.py
# → outputs/evals/baseline_scores.json
```

Supported Groq models:

| Model                       | Speed      | Quality              |
| --------------------------- | ---------- | -------------------- |
| `llama-3.3-70b-versatile` | ★★★     | ★★★★★ (default) |
| `llama3-8b-8192`          | ★★★★★ | ★★★               |
| `mixtral-8x7b-32768`      | ★★★★   | ★★★★             |

---

## 🐳 HuggingFace Spaces Deployment

1. Create a Space: **SDK → Docker**, tag with `openenv`
2. Push this repository as the Space source
3. The `Dockerfile` handles all dependencies

```bash
# Confirm readiness first
./scripts/validate-submission.sh https://YOUR-SPACE.hf.space
```

---

## 📁 File Structure

```
OpenEnv-Railway-Mass-Disruption/
├── inference.py                    ← Mandatory submission inference script
├── openenv.yaml                    ← OpenEnv spec manifest
├── pyproject.toml                  ← Package config + dependencies
├── Dockerfile                      ← Container definition
├── .dockerignore
├── README.md
├── __init__.py                     ← Package exports
├── models.py                       ← Pydantic: Action, Observation, State, Reward
├── engine.py                       ← Cold-chain simulation engine
├── graders.py                      ← Task graders (easy / medium / hard)
├── client.py                       ← EnvClient wrapper
├── baseline.py                     ← Groq baseline inference script
├── tasks/
│   ├── task1_steady_state.json
│   ├── task2_thermal_anomaly.json
│   └── task3_black_swan.json
├── scripts/
│   └── validate-submission.sh      ← Pre-submission validator
├── server/
│   ├── __init__.py
│   ├── fragilechain_environment.py ← OpenEnv Environment class
│   ├── app.py                      ← FastAPI application
│   └── requirements.txt
└── outputs/                        ← Runtime logs & eval results (gitignored)
    ├── logs/
    └── evals/
```

---

## 🧪 Quick Tests

```bash
# Models load
python3 -c "from models import Action, Observation, State; print('OK')"

# Engine + graders
python3 graders.py

# API server round-trip
uvicorn server.app:app --port 8000 &
curl -X POST http://localhost:8000/reset?task_id=task1
curl -X POST http://localhost:8000/step \
     -H "Content-Type: application/json" \
     -d '{"action_type":"do_nothing","internal_thought":"test"}'
```

---

## 📜 License

MIT — see [LICENSE](LICENSE)

---

## 🏆 Hackathon

Built for the **OpenEnv Hackathon** — creating real-world agentic RL environments compliant with the [OpenEnv spec](https://github.com/meta-pytorch/OpenEnv).
