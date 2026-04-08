"""
WiFi-HAR Environment — OpenEnv compliant implementation.
Uses openenv.core.env_server.interfaces.Environment as base.
"""

import random
from typing import List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import WiFiHARAction, WiFiHARObservation
except ImportError:
    from ..models import WiFiHARAction, WiFiHARObservation

from wifi_har.generator import generate_observation, generate_sequence, generate_fall_sequence
from wifi_har.graders import (
    grade_single_classify,
    grade_sequence_classify,
    grade_fall_detection,
    normalize_action,
)

TASK_CONFIGS = {
    "single_classify": {
        "noise_level": 0.05,
        "max_steps": 1,
        "description": "Classify a single clean CSI segment. Easy — low noise, unambiguous signal.",
    },
    "sequence_classify": {
        "noise_level": 0.15,
        "max_steps": 10,
        "description": "Classify 10 consecutive CSI windows with noise and one activity transition. Medium difficulty.",
    },
    "fall_detection": {
        "noise_level": 0.22,
        "max_steps": 30,
        "description": "Detect a fall event in a 30-step noisy stream. Hard — requires anticipatory reasoning.",
    },
}

TASKS = list(TASK_CONFIGS.keys())
DEFAULT_SEED = 42


class WiFiHAREnvironment(Environment):
    """
    WiFi-based Human Activity Recognition OpenEnv Environment.

    Simulates WiFi Channel State Information (CSI) observations from a
    passive indoor sensing system. An AI agent must classify human activities
    from structured natural-language descriptions of processed CSI features.

    Real-world applications: eldercare fall detection, smart home automation,
    security monitoring — all without cameras or wearable sensors.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task: str = "single_classify", seed: Optional[int] = DEFAULT_SEED):
        self._task = task if task in TASK_CONFIGS else "single_classify"
        self._seed = seed
        self._config = TASK_CONFIGS[self._task]
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._observations: List[dict] = []
        self._ground_truths: List[str] = []
        self._actions_taken: List[str] = []
        self._fall_timestep: Optional[int] = None
        self._rewards: List[float] = []
        self._episode_score: float = 0.0
        self._done: bool = False

    def reset(self) -> WiFiHARObservation:
        if self._seed is not None:
            random.seed(self._seed)

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._actions_taken = []
        self._rewards = []
        self._episode_score = 0.0
        self._done = False
        self._fall_timestep = None

        noise = self._config["noise_level"]
        max_steps = self._config["max_steps"]

        if self._task == "single_classify":
            activity = random.choice(["static", "walking", "transition", "fall"])
            self._ground_truths = [activity]
            self._observations = [generate_observation(activity, noise_level=noise)]

        elif self._task == "sequence_classify":
            tp = random.randint(3, 7)
            pool = ["static", "walking", "transition"]
            first = random.choice(pool)
            second = random.choice([a for a in pool if a != first])
            acts = [first] * tp + [second] * (max_steps - tp)
            self._ground_truths = acts
            self._observations = generate_sequence(acts, noise_level=noise)

        elif self._task == "fall_detection":
            self._observations, self._fall_timestep = generate_fall_sequence(
                length=max_steps,
                noise_level=noise,
                seed=self._seed,
            )
            self._ground_truths = [o["activity"] for o in self._observations]

        return self._build_obs()

    def step(self, action: WiFiHARAction) -> WiFiHARObservation:  # type: ignore[override]
        if self._done:
            return self._build_obs()

        activity_str = action.activity if hasattr(action, "activity") else str(action)
        self._actions_taken.append(activity_str)
        self._state.step_count += 1

        ground_truth = self._ground_truths[self._state.step_count - 1]
        reward = self._shaped_reward(activity_str, ground_truth)
        self._rewards.append(reward)

        self._done = self._state.step_count >= self._config["max_steps"]

        if self._done:
            self._episode_score = self._final_score()

        obs = self._build_obs()
        obs.reward = reward
        obs.done = self._done
        obs.metadata = {
            "ground_truth": ground_truth,
            "step": self._state.step_count,
            "score": self._episode_score if self._done else None,
            "last_action_error": None,
        }
        return obs

    @property
    def state(self) -> State:
        self._state.metadata = {
            "task": self._task,
            "done": self._done,
            "episode_score": round(self._episode_score, 4),
            "fall_timestep": self._fall_timestep,
            "actions_taken": list(self._actions_taken),
            "ground_truths": list(self._ground_truths),
            "noise_level": self._config["noise_level"],
        }
        return self._state

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_obs(self) -> WiFiHARObservation:
        idx = min(self._state.step_count, len(self._observations) - 1)
        d = self._observations[idx]
        return WiFiHARObservation(
            text=d["text"],
            step=self._state.step_count,
            task=self._task,
            movement_intensity=d["movement_intensity"],
            doppler_peak=d["doppler_peak"],
            signal_variance=d["signal_variance"],
            noise_estimate=d["noise_estimate"],
            done=self._done,
            reward=0.0,
        )

    def _shaped_reward(self, action: str, ground_truth: str) -> float:
        norm = normalize_action(action)
        if norm == ground_truth:
            return 1.0
        if norm in {"static", "transition"} and ground_truth in {"static", "transition"}:
            return 0.3
        if norm in {"walking", "transition"} and ground_truth in {"walking", "transition"}:
            return 0.3
        if not norm or norm not in {"static", "walking", "transition", "fall"}:
            return -1.0
        return -0.5

    def _final_score(self) -> float:
        if self._task == "single_classify":
            score, _ = grade_single_classify(
                self._actions_taken[0] if self._actions_taken else "",
                self._ground_truths[0],
            )
        elif self._task == "sequence_classify":
            score, _ = grade_sequence_classify(self._actions_taken, self._ground_truths)
        elif self._task == "fall_detection":
            score, _ = grade_fall_detection(
                self._actions_taken,
                self._ground_truths,
                self._fall_timestep,
                self._config["max_steps"],
            )
        else:
            score = 0.001 # Changed from 0.0
            
        # THE FIX: Clamp the score strictly between 0 and 1
        score = max(0.001, min(0.999, float(score)))
        return round(score, 4)
