"""Leader hardware builder for the human teleop side.

Operator-side helper only — the follower lives with `robot.py`. We need
the SO-101 leader (with its built-in arrow-key listener for slider drive)
and the `CAMERAS` tuple so the rerun viewer knows which tracks to render.
"""
from __future__ import annotations

from lerobot_teleoperator_so101_with_slider import (
    SO101WithSliderLeader,
    SO101WithSliderLeaderConfig,
)

from _common import env_bool, env_int, env_str, required_env

CAMERAS: tuple[str, ...] = ("arm_camera", "overhead_camera")
"""Track names. Must match `portal.yaml`."""


def build_leader() -> SO101WithSliderLeader:
    """Construct the SO-101 leader arm + slider keyboard handler. The
    leader manages its own pynput listener for arrow keys (slider drive)
    and only exposes a synchronous ``get_action()`` that returns one
    action dict per call."""
    return SO101WithSliderLeader(SO101WithSliderLeaderConfig(
        id=env_str("SO101_LEADER_ID", "so101_leader"),
        port=required_env("SO101_LEADER_PORT"),
        cruise_velocity=env_int("LESLIDER_CRUISE_VELOCITY", 1500),
        speed_increment=env_int("LESLIDER_SPEED_INCREMENT", 250),
        min_velocity=env_int("LESLIDER_MIN_VELOCITY", 100),
        max_velocity=env_int("LESLIDER_MAX_VELOCITY", 3000),
        invert_direction=env_bool("LESLIDER_INVERT_SLIDER", default=False),
    ))
