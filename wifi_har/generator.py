"""
Synthetic CSI Feature Generator for WiFi-HAR Environment.
Generates realistic WiFi Channel State Information feature vectors
and converts them to LLM-readable text observations.
"""

import random
import math
from typing import Tuple

ACTIVITIES = {
    "static": {
        "movement_intensity": (0.05, 0.15),
        "doppler_range": (0.0, 0.3),
        "signal_variance": (0.05, 0.15),
        "pattern_duration": (1.0, 5.0),
    },
    "walking": {
        "movement_intensity": (0.55, 0.80),
        "doppler_range": (0.8, 2.5),
        "signal_variance": (0.45, 0.70),
        "pattern_duration": (1.5, 4.0),
    },
    "transition": {
        "movement_intensity": (0.25, 0.55),
        "doppler_range": (0.3, 1.2),
        "signal_variance": (0.20, 0.50),
        "pattern_duration": (0.5, 2.0),
    },
    "fall": {
        "movement_intensity": (0.85, 1.00),
        "doppler_range": (2.5, 5.0),
        "signal_variance": (0.75, 1.00),
        "pattern_duration": (0.3, 0.8),
    },
}

ACTIVITY_LABELS = list(ACTIVITIES.keys())


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _randf(lo: float, hi: float, noise: float = 0.0) -> float:
    base = lo + random.random() * (hi - lo)
    if noise > 0:
        base += random.gauss(0, noise)
    return base


def generate_observation(
    activity: str,
    noise_level: float = 0.05,
    previous_activity: str = None,
) -> dict:
    profile = ACTIVITIES[activity]

    movement_intensity = _clamp(_randf(*profile["movement_intensity"], noise=noise_level), 0.0, 1.0)
    doppler_peak = _clamp(_randf(*profile["doppler_range"], noise=noise_level * 2), 0.0, 6.0)
    signal_variance = _clamp(_randf(*profile["signal_variance"], noise=noise_level), 0.0, 1.0)
    pattern_duration = _clamp(_randf(*profile["pattern_duration"], noise=noise_level * 0.5), 0.1, 6.0)
    noise_estimate = _clamp(noise_level + random.gauss(0, 0.02), 0.0, 1.0)

    intensity_label = (
        "very low" if movement_intensity < 0.2 else
        "low" if movement_intensity < 0.4 else
        "moderate" if movement_intensity < 0.6 else
        "high" if movement_intensity < 0.8 else
        "very high"
    )

    doppler_label = (
        "negligible (< 0.5 m/s)" if doppler_peak < 0.5 else
        f"low ({doppler_peak:.1f} m/s)" if doppler_peak < 1.0 else
        f"moderate ({doppler_peak:.1f} m/s)" if doppler_peak < 2.0 else
        f"strong ({doppler_peak:.1f} m/s)" if doppler_peak < 3.5 else
        f"very strong ({doppler_peak:.1f} m/s)"
    )

    variance_label = (
        "stable" if signal_variance < 0.2 else
        "slightly variable" if signal_variance < 0.4 else
        "moderately variable" if signal_variance < 0.6 else
        "highly variable" if signal_variance < 0.8 else
        "extremely variable"
    )

    noise_label = (
        "low" if noise_estimate < 0.1 else
        "moderate" if noise_estimate < 0.2 else
        "high"
    )

    prev_str = (
        f"- Previous observation: {previous_activity} detected\n"
        if previous_activity
        else "- Previous observation: none (episode start)\n"
    )

    text = (
        "Current WiFi Signal Observation:\n"
        f"- Movement intensity: {intensity_label} ({movement_intensity:.2f})\n"
        f"- Doppler profile: {doppler_label}\n"
        f"- Signal variance: {variance_label} ({signal_variance:.2f})\n"
        f"- Pattern duration: {pattern_duration:.1f} seconds\n"
        f"- Environmental noise level: {noise_label} ({noise_estimate:.2f})\n"
        f"{prev_str}"
        "\nBased on these WiFi signal features, what human activity is occurring?\n"
        "Respond with exactly one word: static, walking, transition, or fall"
    )

    return {
        "activity": activity,
        "movement_intensity": round(movement_intensity, 4),
        "doppler_peak": round(doppler_peak, 4),
        "signal_variance": round(signal_variance, 4),
        "pattern_duration": round(pattern_duration, 4),
        "noise_estimate": round(noise_estimate, 4),
        "text": text,
    }


def generate_sequence(activities: list, noise_level: float = 0.10) -> list:
    observations = []
    prev = None
    for act in activities:
        obs = generate_observation(act, noise_level=noise_level, previous_activity=prev)
        observations.append(obs)
        prev = act
    return observations


def generate_fall_sequence(
    length: int = 30,
    fall_timestep: int = None,
    noise_level: float = 0.20,
    seed: int = None,
) -> Tuple[list, int]:
    if seed is not None:
        random.seed(seed)
    if fall_timestep is None:
        fall_timestep = random.randint(15, min(25, length - 3))

    activities = []
    for i in range(length):
        if i == fall_timestep:
            activities.append("fall")
        elif i > fall_timestep:
            activities.append("static")
        else:
            activities.append("transition" if random.random() < 0.15 else "walking")

    return generate_sequence(activities, noise_level=noise_level), fall_timestep
