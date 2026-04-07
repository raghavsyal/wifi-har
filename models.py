"""
Data models for the WiFi-HAR Environment.

Uses openenv base types (Action, Observation) so create_app works correctly.
"""

from typing import Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class WiFiHARAction(Action):
    """
    Action for the WiFi-HAR environment.
    Agent outputs one activity label per step.
    """
    activity: str = Field(
        default="static",
        description="Predicted activity: one of 'static', 'walking', 'transition', 'fall'",
    )


class WiFiHARObservation(Observation):
    """
    Observation from the WiFi-HAR environment.
    Contains a human-readable text description of CSI features
    that an LLM agent can interpret directly.
    """
    text: str = Field(
        default="",
        description="Natural language description of WiFi signal features for LLM reasoning",
    )
    step: int = Field(default=0, description="Current timestep in episode")
    task: str = Field(default="single_classify", description="Active task name")
    movement_intensity: float = Field(default=0.0, description="Raw movement intensity [0,1]")
    doppler_peak: float = Field(default=0.0, description="Peak Doppler shift in m/s")
    signal_variance: float = Field(default=0.0, description="Signal variance [0,1]")
    noise_estimate: float = Field(default=0.0, description="Environmental noise [0,1]")
