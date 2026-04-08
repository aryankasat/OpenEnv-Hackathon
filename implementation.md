# Deployment to Hugging Face Spaces

This guide walks through deploying FragileChain to Hugging Face Spaces for submission and evaluation.

---

## Prerequisites

- [Hugging Face account](https://huggingface.co) with write access
- Git installed locally
- API credentials ready:
  - `HF_TOKEN` (from https://huggingface.co/settings/tokens)
  - `GROQ_API_KEY` or equivalent LLM provider key

---

## Step 1: Create the Space

1. Go to [huggingface.co/spaces/new](https://huggingface.co/spaces/new)
2. Fill in:
   - **Owner:** Your username or organization
   - **Space name:** `fragilechain` (or any name)
   - **License:** MIT
   - **SDK:** Docker 🐳
   - **Visibility:** Public (for evaluation)
   - **Tags:** Add `openenv` tag for discoverability
3. Click **Create Space**

You'll land on the Space page. You now have a fresh Git repo at:
```
https://huggingface.co/spaces/{your_username}/fragilechain
```

---

## Step 2: Configure Git Access

### Option A: Use Hugging Face CLI (Recommended)

```bash
pip install huggingface-hub

huggingface-cli login
# Paste your HF_TOKEN when prompted
```

### Option B: Manual Git Configuration

```bash
git config --global credential.helper store
# On next git push, you'll be prompted for token (use it as password)
```

---

## Step 3: Clone or Push the Repository

### If starting fresh:

```bash
cd /tmp
git clone https://huggingface.co/spaces/${YOUR_USERNAME}/fragilechain
cd fragilechain
```

Then copy all OpenEnv-FragileChain files (except `.git`, `__pycache__`, `outputs/`) into this directory.

### If you have an existing repo:

```bash
cd /path/to/OpenEnv-FragileChain

# Add HF Space as remote
git remote add hf https://huggingface.co/spaces/${YOUR_USERNAME}/fragilechain

# Verify
git remote -v

# Push to Space
git push hf main
# (you may need to specify branch: git push hf HEAD:main)
```

---

## Step 4: Set Secrets in Space Settings

The Space will read environment variables from **Settings → Repository secrets**.

1. Go to your Space URL: `https://huggingface.co/spaces/{username}/fragilechain`
2. Click **Settings** (gear icon, top right)
3. Scroll to **Repository secrets**
4. Add these secrets:

| Name | Value | Example |
|------|-------|---------|
| `HF_TOKEN` | Your HF API key | `hf_abc...xyz` |
| `GROQ_API_KEY` | Groq API key (optional, for baseline) | `gsk_...` |
| `API_KEY` | Alternative to HF_TOKEN | — |

**Note:** These are only visible to the Space owner and never logged.

---

## Step 5: Monitor the Build

Once you push:

1. HF automatically triggers Docker build
2. Go to **Settings → Logs** to watch build progress
3. Build takes ~5–10 min (depends on package installs)
4. If build fails:
   - Check **Build logs** for errors
   - Common issues:
     - Missing `requirements.txt` dependencies → update [server/requirements.txt](server/requirements.txt)
     - GPU not available (shouldn't matter for this env) → ignore
     - Port binding → check [server/app.py](server/app.py) default port

Once build succeeds, you'll see:
```
Running on [Space URL] (press Ctrl+C to quit)
```

---

## Step 6: Test the Deployment

### Health check:

```bash
curl https://{username}-fragilechain.hf.space/health
# → {"status":"ok"}
```

### Reset request:

```bash
curl -X POST https://{username}-fragilechain.hf.space/reset?task_id=task1 \
  -H "Content-Type: application/json"
# → Returns initial observation JSON
```

### Step request:

```bash
curl -X POST https://{username}-fragilechain.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "do_nothing",
    "internal_thought": "Testing deployment"
  }'
# → Returns next observation JSON
```

If all return 200 with proper JSON, your Space is live! ✅

---

## Step 7: Run Submission Validation

Once your Space is live, validate it against the OpenEnv spec:

```bash
./scripts/validate-submission.sh https://{username}-fragilechain.hf.space
```

This runs three checks:
1. **Liveness** — Can ping `/reset` endpoint
2. **Docker** — Dockerfile builds locally (confirms reproducibility)
3. **Spec** — Validates response schemas (requires `openenv-core` installed)

Output:
```
[✓] Space is live and responding
[✓] Docker build succeeded
[✓] OpenEnv spec validation passed

Submission ready for evaluation!
```

---

## Step 8: Run Inference via Space

Test the inference script against your live Space:

```bash
export HF_TOKEN=hf_...
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile

FRAGILECHAIN_TASK=task1 python inference.py
```

This will:
1. Initialize OpenAI client
2. Reset environment on your Space
3. Loop: call LLM → parse action → send to Space
4. Log [START], [STEP], [END] to stdout

Expected output:
```
[START] task=task1 env=fragilechain model=llama-3.3-70b-versatile
[STEP] step=1 action={"action_type":"do_nothing"} reward=0.00 done=false error=null
[STEP] step=2 action={"action_type":"rebalance","source_id":"HUB_CENTRAL","target_id":"SITE_ALPHA","amount":20} reward=0.02 done=false error=null
...
[END] success=true steps=14 score=0.523 rewards=0.00,0.02,...
```

---

## Monitoring & Logs

In your Space Settings:

- **Logs** → Build and runtime logs
- **Settings → Persistent Storage** → If you add SQLite logging, data persists across restarts

To keep the Space alive 24/7, use:
- **Settings → Persistent storage** (enable if using databases)
- Default: Space auto-pauses after 48 hours of no requests

To prevent pause: either keep making requests or upgrade to [Pro/PRO tier](https://huggingface.co/spaces/pricing) (keeps always-on).

---

## Troubleshooting

### Build fails: `ModuleNotFoundError: No module named 'X'`

**Solution:** Add package to [server/requirements.txt](server/requirements.txt):
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
requests==2.31.0
openai==1.7.2  # ← add any missing packages
```

Then commit & push:
```bash
git add server/requirements.txt
git commit -m "Add missing dependency"
git push hf main
```

---

### `/health` endpoint returns 404

**Solution:** Ensure `server/app.py` has:
```python
@app.get("/health")
def health():
    return {"status": "ok"}
```

---

### Port binding error: `Address already in use :8000`

**Solution:** Change port in `server/app.py` or Dockerfile. HF Spaces assigns port via `PORT` env var. Check [server/app.py](server/app.py):
```python
if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

---

### Space times out on requests

**Possible causes:**
1. Engine running too long → check if `max_days` is reasonable
2. LLM call hanging → add timeout to `inference.py`
3. Space underfunded → upgrade tier or optimize

**Check:** Look at **Settings → Logs** for actual error. Most likely it's an inference/LLM hang.

---

## Environment Variables at Runtime

The Space's `docker run` command automatically loads secrets:

```bash
# Inside running Space container:
echo $HF_TOKEN        # ← injected from repository secret
echo $GROQ_API_KEY    # ← if set
```

You can reference these in your inference script or in app initialization:

```python
import os
api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
```

---

## Updating the Space

To update code after initial push:

```bash
cd /path/to/OpenEnv-FragileChain

# Make changes
git add .
git commit -m "Fix: update reward function"
git push hf main

# HF automatically rebuilds and redeploys
```

Watch progress in **Settings → Logs**.

---

## Final Submission Checklist

Before submitting to OpenEnv evaluation:

- [ ] Space is public and live (`https://huggingface.co/spaces/{user}/fragilechain`)
- [ ] `/health` returns 200 with `{"status": "ok"}`
- [ ] `/reset` accepts POST and returns valid `Observation` JSON
- [ ] `/step` accepts POST and returns valid step response
- [ ] `./scripts/validate-submission.sh {SPACE_URL}` passes all checks
- [ ] `inference.py` runs successfully and logs [START], [STEP], [END] 
- [ ] Repository is clean (no large binaries, `__pycache__/` excluded)
- [ ] README.md explains environment and baseline results
- [ ] Dockerfile builds locally: `docker build -t fragilechain .`
- [ ] All three tasks (task1, task2, task3) load without error

Once all checks pass, you're ready to submit the Space URL for evaluation! 🚀

---

## Support & Debugging

If deployment fails:

1. **Check build logs:** Settings → Build logs
2. **Test locally first:**
   ```bash
   docker build -t fragilechain .
   docker run -p 8000:8000 fragilechain
   curl http://localhost:8000/health
   ```
3. **Ask HF Community:** [huggingface.co/spaces](https://huggingface.co/spaces) has forums
4. **Review OpenEnv spec:** [github.com/meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv)

---

## Appendix: Docker Notes

The `Dockerfile` defines:

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r server/requirements.txt
CMD ["python", "server/app.py"]
```

**Key points:**
- Uses `python:3.10-slim` (minimal base image, fast build)
- Installs from `server/requirements.txt` (not root-level `pyproject.toml`)
- CMD runs FastAPI server on port 8000
- HF Spaces exposes port 8000 as the public URL

To customize:
- Add GPU/torch: change base to `nvidia/cuda:12.1-runtime-ubuntu22.04`
- Add system deps: insert `RUN apt-get install -y X` before pip install
- Change working dir: modify WORKDIR

Then rebuild by pushing to Space.

---

**Deployment date:** April 2026  
**Tested with:** Python 3.10, FastAPI 0.104, Docker 24.0, HF Spaces  
**Status:** ✅ Production-ready
