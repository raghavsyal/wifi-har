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

**WiFi-HAR** is a passive indoor human activity recognition environment for training and evaluating AI agents. Human movement perturbs WiFi signals in measurable ways, this environment simulates those perturbations as structured natural-language observations that LLM agents can reason about directly.

Unlike camera-based systems, WiFi sensing is **privacy-preserving** and requires **no wearable devices**, making it practical for eldercare monitoring, smart home automation, and security applications.

Classical rule-based approaches break down on real CSI signals due to noise and environment variability. making this a genuinely appropriate domain for learned agents.

---

## Problem Statement

The agent receives structured descriptions of processed Channel State Information (CSI) features at each timestep and must classify the human activity occurring in the room.

**Why agents outperform classical methods here:**
- CSI signals are noisy and non-stationary
- Same activity looks different across environments and orientations
- No clean feature engineering generalises across deployments
- Learned temporal representations outperform handcrafted rules

**Why this genuinely challenges LLMs:**
- Task 2 requires temporal consistency - classifying each window independently without tracking context underperforms
- Task 3 requires anticipatory reasoning - the agent must recognise pre-fall walking patterns and detect the event early under high noise

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `HF_TOKEN` | HuggingFace API key | required |
| `API_BASE_URL` | LLM API endpoint | https://router.huggingface.co/v1 |
| `MODEL_NAME` | Model identifier | Qwen/Qwen2.5-72B-Instruct |

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
Classify a single clean CSI segment. Low noise (σ=0.05).

- **Max steps:** 1 | **Expected LLM score:** 0.80–0.95
- **Grader:** Exact match → 1.0 | Related category → 0.3 | Wrong → 0.0

### Task 2 — `sequence_classify` (Medium)
Classify 10 consecutive CSI windows. Realistic noise (σ=0.15). One activity transition mid-sequence.

- **Max steps:** 10 | **Expected LLM score:** 0.55–0.75
- **Grader:** Mean per-step score + consistency bonus for ≥80% accuracy

### Task 3 — `fall_detection` (Hard)
Monitor a 30-step stream. Fall injected between steps 15–25, preceded by walking. High noise (σ=0.22).

- **Max steps:** 30 | **Expected LLM score:** 0.30–0.50
- **Grader:** Detection latency score − false alarm penalty + post-fall static bonus

---

## Reward Function

| Outcome | Reward |
|---|---|
| Correct classification | +1.0 |
| Related category (static↔transition, walking↔transition) | +0.3 |
| Wrong classification | −0.5 |
| Invalid / no action | −1.0 |

Fall detection additionally considers detection latency and false alarm rate in the final episode score.

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

Expected stdout:
```
[START] task=single_classify env=wifi-har model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=walking reward=1.00 done=true error=null
[END] success=true steps=1 score=1.00 rewards=1.00
[START] task=sequence_classify env=wifi-har model=Qwen/Qwen2.5-72B-Instruct
...
```

### HTTP API

```bash
# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" -d '{"task": "fall_detection"}'

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" -d '{"activity": "walking"}'

# State
curl http://localhost:7860/state
```

---

## Baseline Scores

Tested with `Qwen/Qwen2.5-72B-Instruct`, seed=42:

| Task | Score |
|---|---|
| single_classify | ~0.85 |
| sequence_classify | ~0.62 |
| fall_detection | ~0.41 |
| **Average** | **~0.63** |

---

## Real-World Applications

- **Eldercare:** Detect falls without cameras, preserving dignity and privacy
- **Smart home:** Passive occupancy detection and activity-aware automation
- **Security:** Intrusion detection using existing WiFi infrastructure
- **Healthcare:** Non-invasive rehabilitation monitoring

---

## Environment Design Notes

- All randomness is seeded — episodes are fully reproducible
- Observations are self-describing — no prior domain knowledge required
- Noise increases monotonically across tasks (0.05 → 0.15 → 0.22)
- All graders are deterministic — no LLM judge in scoring pipeline
- Scores always in [0.0, 1.0], varying meaningfully with agent quality

---

## License

MIT
