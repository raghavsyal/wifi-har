"""
Inference Script — WiFi-HAR OpenEnv Environment
================================================
Runs an LLM agent against all three WiFi-HAR tasks and emits structured
stdout logs in mandatory [START] / [STEP] / [END] format.

Environment variables:
    API_BASE_URL   LLM API endpoint  (default: HuggingFace router)
    MODEL_NAME     Model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       HuggingFace / API key
"""

import os
import sys
import time
import traceback

import subprocess
import sys

# Auto-install required packages if missing
for pkg in ["openai", "openenv-core", "httpx", "pydantic"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

from openai import OpenAI

from wifi_har.environment import WiFiHAREnvironment, TASKS
from models import WiFiHARAction

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

API_KEY      = (
    HF_TOKEN or
    os.getenv("API_KEY") or
    os.getenv("OPENAI_API_KEY") or
    "placeholder"
)

BENCHMARK    = "wifi-har"
SEED         = 42

_client = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    return _client

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI agent analyzing WiFi signal data to recognize human activities.

You will receive structured descriptions of WiFi Channel State Information (CSI) features.
These features capture how human movement perturbs WiFi signals in an indoor environment.

Activity classes and their typical signal signatures:
- static:     very low movement intensity, negligible Doppler (<0.5 m/s), stable signal
- walking:    moderate-high intensity, moderate Doppler (1-3 m/s), variable signal
- transition: moderate intensity, low Doppler (0.3-1.2 m/s), brief duration (sit/stand)
- fall:       very high intensity spike, strong Doppler (>2.5 m/s), extremely variable, very short

You must respond with EXACTLY one word — no punctuation, no explanation:
static, walking, transition, or fall"""

# ── Agent ─────────────────────────────────────────────────────────────────────

def agent_act(observation_text: str, history: list) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-8:])
    messages.append({"role": "user", "content": observation_text})
    try:
        resp = get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=10,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content.strip().lower()
        for label in ["static", "walking", "transition", "fall"]:
            if label in raw:
                return label
        return raw
    except Exception:
        return "static"

# ── Task runner ───────────────────────────────────────────────────────────────

def run_task(task_name: str) -> dict:
    env = WiFiHAREnvironment(task=task_name, seed=SEED)
    obs = env.reset()

    history  = []
    rewards  = []
    steps    = 0
    score    = 0.0
    success  = False
    last_err = None

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        done = False
        while not done:
            #time.sleep(2)
            action_str = agent_act(obs.text, history)

            try:
                action   = WiFiHARAction(activity=action_str)
                next_obs = env.step(action)
                reward   = float(next_obs.reward) if next_obs.reward is not None else 0.0
                done     = next_obs.done
                info     = next_obs.metadata or {}
                last_err = info.get("last_action_error")
            except Exception as e:
                last_err = str(e)[:80]
                reward   = 0.0
                done     = True
                info     = {}

            steps += 1
            rewards.append(reward)
            err_str = last_err if last_err else "null"

            print(
                f"[STEP] step={steps} action={action_str} "
                f"reward={reward:.2f} done={str(done).lower()} error={err_str}",
                flush=True,
            )

            history.append({"role": "user",     "content": obs.text})
            history.append({"role": "assistant", "content": action_str})

            if not done:
                obs = next_obs

        state   = env.state
        score   = state.metadata.get("episode_score", 0.0) if state.metadata else 0.0
        success = score >= 0.5

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        score   = 0.0
        success = False

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )
    return {"task": task_name, "score": score, "steps": steps, "success": success}

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    results = []
    for task in TASKS:
        results.append(run_task(task))

    print("\n=== WiFi-HAR Baseline Summary ===", file=sys.stderr)
    total = 0.0
    for r in results:
        print(f"  {r['task']}: score={r['score']:.3f}  steps={r['steps']}", file=sys.stderr)
        total += r["score"]
    print(f"  Average: {total/len(results):.3f}", file=sys.stderr)

if __name__ == "__main__":
    main()
