---
title: WiFi-HAR OpenEnv
emoji: 📡
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - reinforcement-learning
  - wifi-sensing
  - activity-recognition
  - fall-detection
---

# WiFi-HAR: WiFi-based Human Activity Recognition Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://openenv.dev)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

**WiFi-HAR** is a passive indoor human activity recognition environment for training and evaluating AI agents. Human movement perturbs WiFi signals in measurable ways — this environment simulates those perturbations as structured natural-language observations that LLM agents can reason about directly.

Unlike camera-based systems, WiFi sensing is **privacy-preserving** and requires **no wearable devices**, making it practical for eldercare monitoring, smart home automation, and security applications.

Classical rule-based approaches break down on real CSI signals due to noise and environment variability, making this a genuinely appropriate domain for learned agents.

---

## Problem Statement

Every year, falls among elderly people cause millions of hospitalizations worldwide. Existing detection solutions require either cameras (privacy-invasive) or wearable sensors (forgotten, uncharged, refused). WiFi signals already blanket every indoor space, and human movement perturbs those signals in measurable, classifiable ways. This means fall detection and activity recognition can be done **passively, privately, and with zero additional hardware**.

The challenge is that raw WiFi Channel State Information (CSI) is noisy, non-stationary, and highly environment-dependent. Rule-based thresholds fail the moment furniture moves or a new person enters the space. This is precisely where learned agents have an edge, and why this is a meaningful benchmark domain.

The agent receives structured descriptions of processed CSI features at each timestep and must classify the human activity occurring in the room. The three tasks are ordered by real-world difficulty: recognising a single clean signal, tracking activity through a transition, and detecting a fall under high noise with minimal false alarms.

**Why agents outperform classical methods here:**
- CSI signals are noisy and non-stationary
- The same activity looks different across environments and antenna orientations
- No clean feature engineering generalises across deployments
- Learned temporal representations outperform handcrafted rules

**Why this genuinely challenges LLMs:**
- Task 2 requires temporal consistency — classifying each window independently without tracking context underperforms
- Task 3 requires anticipatory reasoning — the agent must recognise pre-fall walking patterns and detect the event early under high noise

---

## Real-World Applications

- **Eldercare:** Detect falls without cameras, preserving dignity and privacy
- **Smart home:** Passive occupancy detection and activity-aware automation
- **Security:** Intrusion detection using existing WiFi infrastructure
- **Healthcare:** Non-invasive rehabilitation monitoring

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `API_KEY` | LLM API key (injected by validator) | required |
| `HF_TOKEN` | HuggingFace API key (local testing) | optional fallback |
| `API_BASE_URL` | LLM API endpoint | https://router.huggingface.co/v1 |
| `MODEL_NAME` | Model identifier | Qwen/Qwen2.5-72B-Instruct |

The inference script reads `API_KEY` first (validator-injected), then falls back to `HF_TOKEN` for local use.

---

## Observation Space

At each timestep the agent receives a natural-language observation:

```
Current WiFi Signal Observation:
- Movement intensity: high (0.82)
- Doppler profile: strong (2.3 m/s)
- Signal variance: highly variable (0.71)
- Pattern duration: 1.4 seconds
- Environmental noise level: low (0.06)
- Previous observation: walking detected

Based on these WiFi signal features, what human activity is occurring?
Respond with exactly one word: static, walking, transition, or fall
```

| Feature | Description | Range |
|---|---|---|
| movement_intensity | Overall motion energy | [0, 1] |
| doppler_peak | Peak Doppler shift (movement speed) | [0, 6] m/s |
| signal_variance | CSI amplitude variance | [0, 1] |
| pattern_duration | Duration of current signal pattern | [0.1, 6] s |
| noise_estimate | Environmental noise level | [0, 1] |

---

## Action Space

Agent outputs one of four activity labels as a string:

| Label | Description |
|---|---|
| `static` | No movement — person is stationary |
| `walking` | Normal walking detected |
| `transition` | Sitting-to-standing or standing-to-sitting |
| `fall` | Sudden fall event detected |

---

## Tasks

### Task 1 — `single_classify` (Easy)
Classify a single clean CSI segment into one of four activity labels. Low noise (σ=0.05). All four activities including fall can appear.

- **Max steps:** 1
- **Expected LLM score:** 0.75–0.95
- **Grader:** Exact match → ~0.99 | Related category → ~0.30 | Wrong → ~0.01

### Task 2 — `sequence_classify` (Medium)
Classify 10 consecutive CSI windows. Realistic noise (σ=0.15). One activity transition occurs mid-sequence. Agent benefits from tracking context across steps.

- **Max steps:** 10
- **Expected LLM score:** 0.55–0.75
- **Grader:** Mean per-step score + consistency bonus for ≥80% accuracy

### Task 3 — `fall_detection` (Hard)
Monitor a 30-step noisy stream (σ=0.22). A fall event is injected between steps 15–25, preceded by normal walking. Agent must detect the fall early while minimising false alarms.

- **Max steps:** 30
- **Expected LLM score:** 0.30–0.50
- **Grader:** Pre-fall accuracy (up to 0.20) + detection latency score (up to 0.60) + post-fall accuracy (up to 0.20)

---

## Reward Function

Per-step shaped rewards:

| Outcome | Reward |
|---|---|
| Correct classification | +1.0 |
| Related category (static↔transition, walking↔transition) | +0.3 |
| Wrong classification | −0.5 |
| Invalid / no action | −1.0 |

**Episode score** is computed by the task grader at episode end and is always strictly within (0.01, 0.99). This ensures the reward signal varies meaningfully with agent quality — a perfect agent scores ~0.99, a random agent scores ~0.10–0.30 depending on the task.

Fall detection additionally weights detection latency and false alarm rate in the final score, rewarding early detection over late detection.

---

## Setup & Usage

### Docker

```bash
docker build -t wifi-har .
docker run -p 7860:7860 \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  -e HF_TOKEN=your_token \
  wifi-har
```

### Local (uv)

```bash
uv sync
uv run server
```

### Inference baseline

```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

python inference.py
```

Expected stdout format:
```
[START] task=single_classify env=wifi-har model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=walking reward=0.99 done=true error=null
[END] success=true steps=1 score=0.99 rewards=0.99
[START] task=sequence_classify env=wifi-har model=Qwen/Qwen2.5-72B-Instruct
...
[END] success=true steps=10 score=0.65 rewards=...
[START] task=fall_detection env=wifi-har model=Qwen/Qwen2.5-72B-Instruct
...
[END] success=false steps=30 score=0.26 rewards=...
```

### HTTP API

```bash
# Reset to a specific task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" -d '{"task": "fall_detection"}'

# Step with an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" -d '{"activity": "walking"}'

# Get current state
curl http://localhost:7860/state
```

---

## Baseline Scores

Tested with `Qwen/Qwen2.5-72B-Instruct`, seed=42.

When LLM credits are available (full model inference):

| Task | Expected Score |
|---|---|
| single_classify | ~0.85 |
| sequence_classify | ~0.65 |
| fall_detection | ~0.41 |
| **Average** | **~0.64** |

When LLM is unavailable (rule-based fallback only):

| Task | Observed Score |
|---|---|
| single_classify | ~0.30 |
| sequence_classify | ~0.65 |
| fall_detection | ~0.26 |
| **Average** | **~0.40** |

The rule-based fallback uses heuristics on observation text (movement intensity labels, Doppler profile labels) and is intentionally conservative — it demonstrates that the task genuinely requires learned reasoning, not just pattern matching.

---

## Environment Design Notes

- All randomness is seeded — episodes are fully reproducible given the same seed
- Observations are self-describing — no prior domain knowledge required
- Noise increases monotonically across tasks (σ: 0.05 → 0.15 → 0.22)
- All graders are deterministic — no LLM judge in the scoring pipeline
- Episode scores are strictly within (0.01, 0.99) — never exactly 0 or 1
- Per-step rewards can be negative (−0.5 wrong, −1.0 invalid) to penalise bad actions
- Reward signal varies continuously with agent quality, providing a useful training gradient

---

## Project Structure

```
wifi-har/
├── inference.py          # Baseline inference script
├── models.py             # Pydantic action/observation models
├── openenv.yaml          # OpenEnv spec metadata
├── Dockerfile            # Container definition
├── pyproject.toml        # Project dependencies
├── uv.lock               # Locked dependency versions
├── wifi_har/
│   ├── __init__.py
│   ├── environment.py    # Core environment logic
│   ├── generator.py      # Synthetic CSI feature generator
│   └── graders.py        # Deterministic task graders
└── server/
    ├── __init__.py
    └── app.py            # FastAPI server (OpenEnv HTTP interface)
```

---

## License

MIT
