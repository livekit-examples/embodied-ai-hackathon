"""Rerun blueprint + helpers for the human operator viewer.

Cameras + state/action time series + Portal metrics. No recording panel
(this operator does not collect data).
"""
from __future__ import annotations

import rerun as rr
import rerun.blueprint as rrb

from .leslider import CAMERAS


def build_blueprint() -> rrb.Blueprint:
    cam_views = [
        rrb.Spatial2DView(origin=f"/observation/{cam}", name=cam) for cam in CAMERAS
    ]
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(*cam_views),
            rrb.Horizontal(
                rrb.TimeSeriesView(origin="/observation", name="Observation (state)"),
                rrb.TimeSeriesView(origin="/action", name="Action (executed)"),
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(origin="/metrics", name="Portal metrics"),
            ),
            row_shares=[2, 1, 1],
        ),
        collapse_panels=True,
    )


def _us_to_ms(us: int | None) -> float | None:
    return us / 1e3 if us else None


def log_metrics(m) -> None:
    if (rtt := _us_to_ms(m.rtt.rtt_us_last)) is not None:
        rr.log("metrics/rtt_last_ms", rr.Scalars(rtt))
    if (sync := _us_to_ms(m.sync.match_delta_us_p50)) is not None:
        rr.log("metrics/sync_delta_p50_ms", rr.Scalars(sync))
    rr.log("metrics/state_jitter_ms", rr.Scalars(_us_to_ms(m.transport.state_jitter_us) or 0.0))
    rr.log("metrics/action_jitter_ms", rr.Scalars(_us_to_ms(m.transport.action_jitter_us) or 0.0))
    rr.log("metrics/states_dropped", rr.Scalars(float(m.sync.states_dropped)))


def log_observation(obs, state: dict, frames: dict) -> None:
    """Mirror a synced observation to rerun. Best-effort; swallows errors
    so logging hiccups never kill the control loop."""
    try:
        if obs is not None:
            rr.set_time("robot_time", timestamp=obs.timestamp_us / 1e6)
        for cam, frame in frames.items():
            rr.log(f"observation/{cam}", rr.Image(frame).compress(), static=True)
        for k, v in state.items():
            rr.log(f"observation/{k}", rr.Scalars(float(v)))
    except Exception:
        pass


def log_action(action, me: str | None) -> None:
    """Mirror the latest executed action plus a 0/1 series flagging
    whether the action came from us or a remote operator."""
    if action is None:
        return
    try:
        rr.log("action/_sender_is_self", rr.Scalars(float(action.sender == me)))
        for k, v in action.values.items():
            rr.log(f"action/{k}", rr.Scalars(float(v)))
    except Exception:
        pass
