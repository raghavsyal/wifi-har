"""
Deterministic graders for WiFi-HAR tasks.
All return scores in (0, 1) — strictly between 0 and 1.
"""

from typing import List, Tuple


def clamp_score(score: float) -> float:
    """Clamp score to strictly within (0, 1) — never exactly 0.0 or 1.0."""
    return max(0.001, min(0.999, float(score)))

VALID_ACTIONS = {"static", "walking", "transition", "fall"}

PROXIMITY = {
    ("static", "transition"): 0.3,
    ("transition", "static"): 0.3,
    ("walking", "transition"): 0.3,
    ("transition", "walking"): 0.3,
}


def normalize_action(action: str) -> str:
    if not action:
        return ""
    action = action.strip().lower()
    for label in VALID_ACTIONS:
        if label in action:
            return label
    return action


def single_step_score(predicted: str, ground_truth: str) -> float:
    predicted = normalize_action(predicted)
    if predicted == ground_truth:
        return 1.0
    return PROXIMITY.get((predicted, ground_truth), 0.0)


def grade_single_classify(action: str, ground_truth: str) -> Tuple[float, str]:
    score = single_step_score(action, ground_truth)
    norm = normalize_action(action)
    if score == 1.0:
        info = f"Correct: '{norm}' == '{ground_truth}'"
    elif score > 0:
        info = f"Partial: predicted '{norm}', truth '{ground_truth}' (related)"
    else:
        info = f"Wrong: predicted '{norm}', truth '{ground_truth}'"
    return round(clamp_score(score), 4), info


def grade_sequence_classify(actions: List[str], ground_truths: List[str]) -> Tuple[float, str]:
    if len(actions) != len(ground_truths):
        return 0.001, f"Length mismatch: {len(actions)} vs {len(ground_truths)}"

    step_scores = [single_step_score(a, g) for a, g in zip(actions, ground_truths)]
    base = sum(step_scores) / len(step_scores)
    bonus = 0.05 if sum(1 for s in step_scores if s == 1.0) >= len(ground_truths) * 0.8 else 0.0
    final = min(1.0, base + bonus)

    correct = sum(1 for s in step_scores if s == 1.0)
    partial = sum(1 for s in step_scores if 0 < s < 1.0)
    wrong = sum(1 for s in step_scores if s == 0.0)
    info = f"{correct}/{len(ground_truths)} correct, {partial} partial, {wrong} wrong. Score={final:.3f}"
    return round(clamp_score(final), 4), info


def grade_fall_detection(
    actions: List[str],
    ground_truths: List[str],
    fall_timestep: int,
    total_steps: int,
) -> Tuple[float, str]:
    """
    Grade fall detection with partial credit at every level:
      - Pre-fall context accuracy (walking correctly classified): up to 0.20
      - Fall detection (correct label at or after fall_timestep): up to 0.60
        - Penalised by latency and false alarms
      - Post-fall accuracy (static correctly classified): up to 0.20

    This ensures scores vary meaningfully even for agents that miss the fall.
    """
    normalized = [normalize_action(a) for a in actions]

    # ── Pre-fall accuracy (0.0 – 0.20) ──────────────────────────────────────
    pre_actions  = normalized[:fall_timestep]
    pre_truths   = ground_truths[:fall_timestep]
    pre_correct  = sum(1 for a, g in zip(pre_actions, pre_truths) if a == g)
    pre_score    = (pre_correct / fall_timestep * 0.20) if fall_timestep > 0 else 0.0

    # ── Post-fall accuracy (0.0 – 0.20) ─────────────────────────────────────
    post_actions = normalized[fall_timestep + 1:]
    post_truths  = ground_truths[fall_timestep + 1:]
    post_correct = sum(1 for a, g in zip(post_actions, post_truths) if a == g)
    post_score   = (post_correct / len(post_truths) * 0.20) if post_truths else 0.0

    # ── Fall detection (0.0 – 0.60) ─────────────────────────────────────────
    detected_at  = next((i for i, a in enumerate(normalized) if a == "fall"), None)
    false_alarms = sum(1 for i, a in enumerate(normalized) if a == "fall" and i < fall_timestep)
    fa_penalty   = min(0.20, false_alarms * 0.05)

    if detected_at is None or detected_at < fall_timestep:
        detection_score = 0.0
        det_info = "Miss" if detected_at is None else f"False-alarm-only@{detected_at}"
    else:
        latency = detected_at - fall_timestep
        if latency == 0:
            raw = 0.60
        elif latency <= 5:
            raw = 0.60 - latency * 0.08   # 0.52 → 0.20
        elif latency <= 10:
            raw = max(0.20, 0.60 - latency * 0.06)
        else:
            raw = 0.10
        detection_score = max(0.0, raw - fa_penalty)
        det_info = f"Detected@{detected_at}(latency={latency})"

    final = round(clamp_score(pre_score + detection_score + post_score), 4)
    info  = (
        f"{det_info} | pre={pre_score:.3f} det={detection_score:.3f} "
        f"post={post_score:.3f} fa={false_alarms} final={final:.3f}"
    )
    return final, info
