import asyncio
import base64
import json
import logging
import textwrap
import time
from contextlib import asynccontextmanager, suppress
from typing import Annotated, Literal

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    inference,
    room_io,
)
from livekit.agents.llm import ChatContext, ChatMessage, ImageContent
from livekit.agents.llm.async_toolset import AsyncRunContext, AsyncToolset
from livekit.agents.utils.images import EncodeOptions, encode
from livekit.plugins import ai_coustics, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.rtc.rpc import RpcError
from pydantic import BaseModel, Field

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Identities of the operator participants to invoke RPCs on. The move-to
# operator also serves reset_pose, since reset_pose is just another arm motion.
OPERATOR_MOVE_TO_IDENTITY = "move-to-operator"
OPERATOR_RUN_POLICY_IDENTITY = "run-policy-operator"
OPERATOR_RUN_MOLMO_IDENTITY = "run-molmo-operator"

# Operator RPCs (move_to, run_policy, reset_pose) block on real arm motion which
# can take tens of seconds; give them plenty of headroom before timing out.
_OPERATOR_RPC_TIMEOUT_S = 180.0

# Identity of the participant publishing the workbench overhead camera,
# and the published track name we subscribe to for visual context.
ROBOT_IDENTITY = "robot"
OVERHEAD_CAMERA_TRACK_NAME = "overhead_camera"

# Maximum age (seconds) for a buffered frame to be considered fresh enough to act on.
_MAX_FRAME_AGE_S = 5.0


class WorkbenchCamera:
    """Subscribes to the robot's `overhead_camera` track and buffers the latest frame.

    Frames are consumed in the background; callers read `latest_frame` to get
    the most recent `rtc.VideoFrame` (or `None` if the robot hasn't connected
    yet, has disconnected, or the stream has stalled). The buffered frame is
    cleared eagerly on `track_unsubscribed` and when the reader loop exits,
    and lazily on read if it has aged past `_MAX_FRAME_AGE_S` — so we never
    feed stale imagery to the planner.
    """

    def __init__(self, room: rtc.Room) -> None:
        self._room = room
        self._latest_frame: rtc.VideoFrame | None = None
        # Monotonic timestamp of the last frame assigned to `_latest_frame`.
        # Only meaningful while `_latest_frame is not None`.
        self._latest_frame_at: float = 0.0
        self._stream: rtc.VideoStream | None = None
        self._reader_task: asyncio.Task | None = None
        # Holds background stream-close tasks so they aren't garbage-collected
        # before they finish (RUF006).
        self._pending_closes: set[asyncio.Task] = set()

        for participant in room.remote_participants.values():
            if participant.identity != ROBOT_IDENTITY:
                continue
            for publication in participant.track_publications.values():
                if (
                    publication.name == OVERHEAD_CAMERA_TRACK_NAME
                    and publication.track is not None
                ):
                    self._start_reader(publication.track)

        room.on("track_subscribed", self._on_track_subscribed)
        room.on("track_unsubscribed", self._on_track_unsubscribed)

    @property
    def latest_frame(self) -> rtc.VideoFrame | None:
        """Most recent buffered frame, or `None` if missing or stale.

        Drops and returns `None` if the buffered frame is older than
        `_MAX_FRAME_AGE_S`. This catches stalls where the track stays
        subscribed but frames stop flowing (publisher hang, silent network
        drop, etc.) and the unsubscribe handler never fires. Suitable for
        non-critical callers (e.g. opportunistically attaching a frame to
        a chat turn) that should silently tolerate absence.
        """
        if self._latest_frame is None:
            return None
        if time.monotonic() - self._latest_frame_at > _MAX_FRAME_AGE_S:
            logger.warning(
                "overhead_camera frame is stale (>%.1fs old); clearing",
                _MAX_FRAME_AGE_S,
            )
            self._latest_frame = None
            return None
        return self._latest_frame

    def require_fresh_frame(self) -> rtc.VideoFrame | None:
        """Return the latest fresh frame, logging a warning if unavailable.

        Convenience for tools that gate execution on having a current view
        of the workbench: same return value as `latest_frame`, but logs a
        consistent warning across call sites when no frame is available.
        Callers should still handle `None` (typically by returning
        `_NO_CAMERA_ERROR` to the LLM).
        """
        frame = self.latest_frame
        if frame is None:
            logger.warning("no overhead_camera frame available; refusing to act")
        return frame

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if (
            participant.identity != ROBOT_IDENTITY
            or publication.name != OVERHEAD_CAMERA_TRACK_NAME
            or track.kind != rtc.TrackKind.KIND_VIDEO
        ):
            return
        self._start_reader(track)

    def _on_track_unsubscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if (
            participant.identity != ROBOT_IDENTITY
            or publication.name != OVERHEAD_CAMERA_TRACK_NAME
        ):
            return
        logger.info("overhead_camera track unsubscribed; clearing latest frame")
        self._latest_frame = None
        self._cancel_reader()

    def _start_reader(self, track: rtc.Track) -> None:
        self._cancel_reader()
        logger.info("Subscribing to overhead_camera track from robot")
        stream = rtc.VideoStream(track)
        self._stream = stream

        async def read_stream() -> None:
            try:
                async for event in stream:
                    self._latest_frame = event.frame
                    self._latest_frame_at = time.monotonic()
            except Exception:
                logger.exception("overhead_camera reader failed")
            finally:
                # Stream ended (publisher unpublished, track ended, reader
                # cancelled, etc.) without necessarily routing through the
                # `track_unsubscribed` handler — clear so we don't keep
                # serving the last good frame indefinitely.
                self._latest_frame = None

        self._reader_task = asyncio.create_task(read_stream())

    def _cancel_reader(self) -> None:
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
        self._reader_task = None
        if self._stream is not None:
            close_task = asyncio.create_task(self._stream.aclose())
            self._pending_closes.add(close_task)
            close_task.add_done_callback(self._pending_closes.discard)
            self._stream = None

    async def aclose(self) -> None:
        self._latest_frame = None
        task = self._reader_task
        self._reader_task = None
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
        if self._stream is not None:
            await self._stream.aclose()
            self._stream = None


# Encode frames at their native resolution (the overhead camera publishes at
# 640x480, which is already small and detailed enough for tool recognition).
_FRAME_ENCODE_OPTIONS = EncodeOptions(format="JPEG")


# Returned by every robot-controlling tool when no overhead camera frame is available.
_NO_CAMERA_ERROR = (
    "Error: no overhead camera feed available; refusing to act without "
    "visual confirmation of the workbench."
)


def _frame_to_image_content(frame: rtc.VideoFrame) -> ImageContent:
    """Encode a video frame as a base64 JPEG data URL and wrap it in ImageContent.

    Following https://docs.livekit.io/agents/multimodality/vision/images/ —
    `ImageContent` accepts a `data:` URL, so we encode the frame ourselves
    rather than handing the raw `VideoFrame` to the LLM (which doesn't survive
    some LLM provider serializers, including OpenAI structured-output mode).
    """
    image_bytes = encode(frame, _FRAME_ENCODE_OPTIONS)
    data_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("utf-8")
    return ImageContent(image=data_url)


# Plan schema: a discriminated union of robot actions. The planner LLM is
# constrained to output JSON that validates against `Plan` via OpenAI's
# strict structured-output mode (see `PlannerToolset.plan_and_execute`).

Arm = Literal["left", "right"]


class MoveTo(BaseModel):
    """Position `arm` over `object_name` so a subsequent policy can act on it."""

    kind: Literal["move_to"]
    object_name: str
    arm: Arm


class RunPolicy(BaseModel):
    """Run a low-level pick-up / put-down policy on `arm`."""

    kind: Literal["run_policy"]
    policy: Literal["pick_up", "put_down"]
    arm: Arm


Action = Annotated[MoveTo | RunPolicy, Field(discriminator="kind")]


class Plan(BaseModel):
    """Ordered list of robot actions that fulfill a task."""

    steps: list[Action]


# Maps low-level policy names to a human-readable verb for the activity badge.
_POLICY_VERBS: dict[str, str] = {"pick_up": "Picking up", "put_down": "Putting down"}

# Stub delay per step, simulating arm motion until real RPCs are wired in.
_STUB_STEP_DURATION_S = 6.0


def _describe_step(step: Action) -> str:
    """Short, human-readable status string for a single plan step."""
    match step:
        case MoveTo(arm=arm, object_name=name):
            return f"Moving {arm} arm to {name}"
        case RunPolicy(arm=arm, policy=policy):
            return f"{_POLICY_VERBS[policy]} with {arm} arm"


async def _execute_run_policy(step: RunPolicy, local_participant) -> None:
    await local_participant.perform_rpc(
        destination_identity=OPERATOR_RUN_POLICY_IDENTITY,
        method="run_policy",
        payload=json.dumps({"policy": step.policy, "side": step.arm}),
        response_timeout=_OPERATOR_RPC_TIMEOUT_S,
    )


async def _execute_move_to(step: MoveTo, local_participant) -> None:
    await local_participant.perform_rpc(
        destination_identity=OPERATOR_MOVE_TO_IDENTITY,
        method="move_to",
        payload=step.object_name,
        response_timeout=_OPERATOR_RPC_TIMEOUT_S,
    )


async def _send_to_reset_pose(ctx: AsyncRunContext) -> None:
    """Move the robot to its reset pose.

    Not a planner action — exposed only via the `reset_pose` function tool.
    Pulls the room from `ctx` itself and no-ops when none is available (e.g.
    `agent.py console`), mirroring the convention used by `execute_plan` /
    `plan_progress`.
    """
    try:
        room = ctx.session.room_io.room
    except RuntimeError:
        room = None
    if room is None:
        logger.warning("No room available; skipping reset_pose RPC")
        return
    try:
        await room.local_participant.perform_rpc(
            destination_identity=OPERATOR_MOVE_TO_IDENTITY,
            method="reset_pose",
            payload="",
            response_timeout=_OPERATOR_RPC_TIMEOUT_S,
        )
    except RpcError as e:
        # Same failure mode as the other operator RPCs: log and continue if
        # the move-to operator isn't connected.
        logger.warning("reset_pose RPC failed: %s", e)


async def _send_run_molmo(ctx: AsyncRunContext, prompt: str) -> str | None:
    """Engage the MolmoAct2 VLA operator with a natural-language prompt.

    Not a planner action: exposed only via the `run_molmo` function tool,
    which is reserved for requests where the user explicitly names the
    `run_molmo` policy. Returns an error string on failure, or None on
    success. Same room-availability convention as the other operator RPCs.
    """
    try:
        room = ctx.session.room_io.room
    except RuntimeError:
        room = None
    if room is None:
        logger.warning("No room available; skipping run_molmo RPC")
        return None
    logger.info("Sending run_molmo RPC with prompt: %r", prompt)
    try:
        await room.local_participant.perform_rpc(
            destination_identity=OPERATOR_RUN_MOLMO_IDENTITY,
            method="run_molmo",
            payload=json.dumps({"prompt": prompt}),
            response_timeout=_OPERATOR_RPC_TIMEOUT_S,
        )
        return None
    except RpcError as e:
        logger.warning("run_molmo RPC failed: %s", e)
        return str(e)


async def _send_release_gripper(ctx: AsyncRunContext) -> None:
    """Manually release the robot's gripper.

    Not a planner action — exposed only via the `release_gripper` function
    tool. Same room-availability convention as `_send_to_reset_pose`.
    """
    try:
        room = ctx.session.room_io.room
    except RuntimeError:
        room = None
    if room is None:
        logger.warning("No room available; skipping release_gripper RPC")
        return
    try:
        await room.local_participant.perform_rpc(
            destination_identity=OPERATOR_MOVE_TO_IDENTITY,
            method="release_gripper",
            payload="",
            response_timeout=_OPERATOR_RPC_TIMEOUT_S,
        )
    except RpcError as e:
        logger.warning("release_gripper RPC failed: %s", e)


async def _execute_step(step: Action, room) -> None:
    """Run a single plan step. Lets `RpcError` propagate to `execute_plan`."""
    logger.info("Executing step: %s", step.model_dump_json())
    if room is None:
        logger.warning("No room available, falling back to stub sleep")
        await asyncio.sleep(_STUB_STEP_DURATION_S)
        return
    local_participant = room.local_participant
    match step:
        case RunPolicy():
            await _execute_run_policy(step, local_participant)
        case MoveTo():
            await _execute_move_to(step, local_participant)


async def execute_plan(
    ctx: AsyncRunContext, plan: Plan, pub: "PlanProgress"
) -> str | None:
    """Run each step of `plan` in sequence, flipping per-step status as it goes.

    On success, the publisher is left in `phase: "completed"` so the UI can
    keep the finished list visible until the next plan replaces it, and `None`
    is returned. On first `RpcError` the failing step is marked `error`, the
    plan phase flips to `error`, remaining steps stay `pending`, and a
    human-readable error string is returned for the caller to relay to the
    LLM. Caught here (not in `plan_and_execute`) so the error never escapes
    `plan_progress(ctx)`'s `BaseException` handler, which would clear the
    snapshot and erase the error state from the UI.
    """
    # `session.room_io` raises RuntimeError when running without a room (e.g.
    # `agent.py console`); the executors handle a None room gracefully.
    try:
        room = ctx.session.room_io.room
    except RuntimeError:
        room = None
    for i, step in enumerate(plan.steps):
        await pub.set_step_status(i, "in_progress")
        try:
            await _execute_step(step, room)
        except RpcError as e:
            label = _describe_step(step)
            logger.warning("RPC failed for step %d (%s): %s", i, step.kind, e)
            await pub.set_step_status(i, "error")
            await pub.set_error()
            return (
                f"Step {i + 1} '{label}' failed: {e}. "
                "Remaining steps were not executed."
            )
        await pub.set_step_status(i, "completed")
    await pub.set_completed()
    return None


# The planner runs in OpenAI strict structured-output mode against the `Plan`
# schema, so the JSON shape is guaranteed; the prompts only need to steer
# semantic choices (which tool, which arm, how many steps).
_PLANNER_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are the planner for an embodied agent with two arms on linear sliders
    above a user's toolbench. Given a task and an image of the toolbench,
    output a sequence of robot actions that fulfills the task.

    Output contract:
    - You will be invoked in JSON-schema-constrained mode. Output ONLY the
      JSON object that matches the Plan schema.
    - Reference tools by short, visually grounded descriptions (color + type,
      e.g. "red screwdriver", "claw hammer") that match what is visible
      in the image.
    - Use the minimum number of steps needed. Do not invent extra steps.
    - Pick `arm` based on which side of the toolbench the object is on.
    """
)


# Hints + Examples block, appended to a user message that also carries the
# toolbench image and the new "# Task: ..." line.
_PLANNER_USER_PROMPT_HEADER = textwrap.dedent(
    """\
    Available actions (as JSON objects):
      {"kind": "move_to", "object_name": <str>, "arm": "left"|"right"}
      {"kind": "run_policy", "policy": "pick_up"|"put_down", "arm": "left"|"right"}

    Examples:

    # Task: hand me the claw hammer.
    {"steps": [
      {"kind": "move_to",    "object_name": "claw hammer", "arm": "right"},
      {"kind": "run_policy", "policy": "pick_up",          "arm": "right"}
    ]}

    # Task: put the blue painter's tape away in the toolbox.
    {"steps": [
      {"kind": "move_to",    "object_name": "blue painter's tape", "arm": "left"},
      {"kind": "run_policy", "policy": "pick_up",                  "arm": "left"},
      {"kind": "move_to",    "object_name": "toolbox",             "arm": "left"},
      {"kind": "run_policy", "policy": "put_down",                 "arm": "left"}
    ]}

    # Task: clear the needle-nose pliers and the utility knife off the bench.
    {"steps": [
      {"kind": "move_to",    "object_name": "needle-nose pliers", "arm": "left"},
      {"kind": "run_policy", "policy": "pick_up",                 "arm": "left"},
      {"kind": "move_to",    "object_name": "utility knife",      "arm": "right"},
      {"kind": "run_policy", "policy": "pick_up",                 "arm": "right"},
      {"kind": "move_to",    "object_name": "toolbox",            "arm": "left"},
      {"kind": "run_policy", "policy": "put_down",                "arm": "left"},
      {"kind": "move_to",    "object_name": "toolbox",            "arm": "right"},
      {"kind": "run_policy", "policy": "put_down",                "arm": "right"}
    ]}
    """
)


# Topic for the plan-progress text stream consumed by the frontend.
_PLAN_TOPIC = "agent.plan"

StepStatus = Literal["pending", "in_progress", "completed", "error"]
PlanPhase = Literal["idle", "planning", "executing", "completed", "error"]


class PlanProgress:
    """Holds the current plan-progress snapshot and publishes it on every change.

    The whole snapshot is sent on each update; the client just renders the
    latest payload it received on `_PLAN_TOPIC`. In console mode (no room),
    snapshots are logged instead.
    """

    def __init__(self, room) -> None:
        self._room = room
        self._phase: PlanPhase = "idle"
        self._task: str = ""
        self._steps: list[dict] = []

    async def _publish(self) -> None:
        payload = json.dumps(
            {"phase": self._phase, "task": self._task, "steps": self._steps}
        )
        if self._room is None:
            logger.info("plan-progress (console): %s", payload)
            return
        await self._room.local_participant.send_text(payload, topic=_PLAN_TOPIC)

    async def set_planning(self, task: str) -> None:
        self._phase = "planning"
        self._task = task
        self._steps = []
        await self._publish()

    async def set_plan(self, descriptions: list[str]) -> None:
        self._phase = "executing"
        self._steps = [{"description": d, "status": "pending"} for d in descriptions]
        await self._publish()

    async def set_step_status(self, index: int, status: StepStatus) -> None:
        self._steps[index]["status"] = status
        await self._publish()

    async def set_completed(self) -> None:
        self._phase = "completed"
        await self._publish()

    async def set_error(self) -> None:
        # Only the status is exposed to the UI; the error message itself is
        # handed back to the LLM via the tool return value and logged.
        self._phase = "error"
        await self._publish()

    async def clear(self) -> None:
        self._phase = "idle"
        self._task = ""
        self._steps = []
        await self._publish()


@asynccontextmanager
async def plan_progress(ctx: RunContext):
    """Yield a `PlanProgress` for the duration of a plan_and_execute call.

    On success, the caller is expected to have moved the publisher to
    `phase: "completed"` (via `execute_plan`), so the finished list stays
    visible to the user until the next plan overwrites it. On error or
    cancellation the publisher is cleared so a stale half-done plan does
    not linger on the UI.

    `session.room_io` raises RuntimeError when the session was started without
    a room (e.g. `agent.py console`); in that case the publisher falls back
    to logging so the agent still works locally.
    """
    try:
        room = ctx.session.room_io.room
    except RuntimeError:
        room = None
    pub = PlanProgress(room)
    try:
        yield pub
    except BaseException:
        await pub.clear()
        raise


class PlannerToolset(AsyncToolset):
    """Async toolset that owns the vision planner LLM and the plan_and_execute tool.

    Implemented as an `AsyncToolset` so the agent can speak progress updates
    via `ctx.update(...)` *while* the work is running, instead of only after
    the tool returns.
    """

    def __init__(self, planner_llm: inference.LLM, camera: WorkbenchCamera) -> None:
        super().__init__(id="planner")
        self._planner_llm = planner_llm
        self._camera = camera

    @function_tool()
    async def plan_and_execute(
        self, ctx: AsyncRunContext, task_description: str
    ) -> str:
        """Plan a toolbench task and execute it on the robot.

        The tool runs in the background; progress is narrated to the user via
        `ctx.update(...)` (planning, moving arms, picking up, etc.). The
        return value is a brief completion string — do NOT recap individual
        steps aloud, since they were already narrated while running.

        Args:
            task_description: A short natural-language description of the task to complete (e.g. "get the red screwdriver",
            "put away the blue screwdriver", "clean up the tools", "organize the tools").
        """
        ctx.disallow_interruptions()
        logger.info("Planning task: %s", task_description)

        frame = self._camera.require_fresh_frame()
        if frame is None:
            return _NO_CAMERA_ERROR

        await ctx.update(f"Planning how to {task_description}")

        planner_ctx = ChatContext()
        planner_ctx.add_message(role="system", content=_PLANNER_SYSTEM_PROMPT)
        planner_ctx.add_message(
            role="user",
            content=[
                _PLANNER_USER_PROMPT_HEADER + f"\n# Task: {task_description}\n",
                _frame_to_image_content(frame),
            ],
        )

        async with plan_progress(ctx) as pub:
            await pub.set_planning(task_description)

            response = await self._planner_llm.chat(
                chat_ctx=planner_ctx,
                response_format=Plan,
            ).collect()
            plan = Plan.model_validate_json(response.text)
            logger.info(
                "Generated plan with %d step(s):\n%s",
                len(plan.steps),
                plan.model_dump_json(indent=2),
            )

            await pub.set_plan([_describe_step(s) for s in plan.steps])
            error = await execute_plan(ctx, plan, pub)

        if error is not None:
            return f"Error: {error}"
        return f"Completed: {task_description}"

    @function_tool()
    async def reset_pose(self, ctx: AsyncRunContext) -> str:
        """Move the robot's arms to their reset / home position.

        Use when the user explicitly asks the robot to reset, reset, park, go
        home, get out of the way, return to its starting position, or similar.
        Do NOT call this in response to tool-handling requests like
        "pick up X" or "put away Y" — `plan_and_execute` already homes the
        arms before running a plan.
        """
        ctx.disallow_interruptions()
        logger.info("User-invoked reset_pose")
        await ctx.update("Returning to reset pose")
        await _send_to_reset_pose(ctx)
        return "Robot is at reset."

    @function_tool()
    async def run_molmo(self, ctx: AsyncRunContext, prompt: str) -> str:
        """Hand control to the MolmoAct2 VLA policy with a natural-language prompt.

        Reserved for requests where the user EXPLICITLY names `run_molmo` (e.g.
        "pick this up using molmo", "use molmo to grab the screwdriver",
        "run molmo to put away the hammer"). Do NOT call this for ordinary
        pick-up / put-away / cleanup requests — those are handled by
        `plan_and_execute`. If the user just says "pick this up" without
        naming molmo, use `plan_and_execute` instead.

        Args:
            prompt: A short natural-language instruction passed directly to the
            MolmoAct2 policy (e.g. "pick up the blue cube", "grasp the red
            screwdriver"). Strip any "using run_molmo" framing — the policy
            only needs the underlying task.
        """
        ctx.disallow_interruptions()
        logger.info("User-invoked run_molmo: %s", prompt)
        if self._camera.require_fresh_frame() is None:
            return _NO_CAMERA_ERROR
        await ctx.update(f"Running molmo policy: {prompt}")
        error = await _send_run_molmo(ctx, prompt)
        if error is not None:
            return f"Error: run_molmo failed: {error}"
        return f"run_molmo completed: {prompt}"

    @function_tool()
    async def release_gripper(self, ctx: AsyncRunContext) -> str:
        """Manually open / release the robot's gripper.

        Use when the user explicitly asks the robot to release, open the
        gripper, drop / let go of what it is holding, or similar. Do NOT
        call this as part of a pick-up or put-away request — those are
        handled by `plan_and_execute`.
        """
        ctx.disallow_interruptions()
        logger.info("User-invoked release_gripper")
        await ctx.update("Releasing the gripper")
        await _send_release_gripper(ctx)
        return "Gripper released."


class Orchestrator(Agent):
    def __init__(self, camera: WorkbenchCamera) -> None:
        # Dedicated planner LLM used inside the planner toolset for
        # vision-based reasoning, kept separate from the voice pipeline LLM
        # to avoid mixing concerns.
        planner_llm = inference.LLM(model="openai/gpt-5.3-chat-latest")
        self._camera = camera

        super().__init__(
            llm=inference.LLM(model="openai/gpt-5.3-chat-latest"),
            instructions=textwrap.dedent(
                """\
                You are the toolbench helper, an embodied AI agent equipped with two robotic arms on linear sliders.
                You can pick up and move the user's tools at their request, e.g.:
                - "Can you get me the wrench?"
                - "Can you put away the blue screwdriver?"
                - "Can you clean up the tools"

                Use the `plan_and_execute` tool for any such request. The tool runs in the background and
                narrates progress aloud as it works (planning, moving, picking up, etc.).

                If `plan_and_execute` returns a string that begins with "Error:", briefly apologize and
                tell the user what went wrong in one short sentence. The error message contains a step
                number and a raw technical reason (e.g. RPC timeouts, internal exception text, server
                identifiers) which may not be useful or intelligible to the user — translate it into
                plain language and only mention details the user can act on (e.g. which step failed, or
                that the robot couldn't reach / grasp something). Do not read out the raw error text
                verbatim, and do not retry automatically.

                `plan_and_execute` and `run_molmo` may also refuse with an error mentioning that no
                overhead camera feed is available. The workbench camera is required for safe motion
                planning, so if you see that error, briefly tell the user the camera is offline and
                ask them to check it before trying again. Do not retry the tool until the user
                indicates the camera is back. `reset_pose` and `release_gripper` remain available
                even when the camera is offline, since they are fixed motions that don't depend on
                seeing the workbench.

                If the user explicitly asks the robot to reset, park, go home, return to its starting
                position, or get out of the way, call the `reset_pose` tool.

                If the user explicitly asks the robot to release / open the gripper or drop / let go of
                what it is holding, call the `release_gripper` tool.

                Only call the `run_molmo` tool when the user EXPLICITLY names it in the request,
                for example "pick this up using run_molmo", "use run_molmo to grab the wrench",
                or "run molmo to put away the hammer". For ordinary requests like "pick this up"
                or "put the screwdriver away" that do not mention run_molmo, use `plan_and_execute`
                instead. Never substitute `run_molmo` for `plan_and_execute` on your own.
                """
            ),
            tools=[PlannerToolset(planner_llm, camera)],
        )

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        frame = self._camera.latest_frame
        if frame is not None:
            new_message.content.append(_frame_to_image_content(frame))


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="orchestrator")
async def orchestrator(ctx: JobContext):

    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="en-US"),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
            language="en",
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await ctx.connect()

    camera = WorkbenchCamera(ctx.room)
    ctx.add_shutdown_callback(camera.aclose)

    await session.start(
        agent=Orchestrator(camera),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            participant_identity="user",
            close_on_disconnect=False,
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S
                ),
            ),
        ),
    )

    # Dev shortcut: simulate a user utterance at launch so we can iterate on
    # plan_and_execute without speaking into a mic each time. Set the env var
    # to any prompt, e.g. DEV_SIMULATE_PROMPT="pick up the red screwdriver".
    # dev_prompt = os.getenv("DEV_SIMULATE_PROMPT")
    # if dev_prompt:
    #     logger.info("DEV_SIMULATE_PROMPT set, simulating user input: %r", dev_prompt)
    #     session.generate_reply(user_input=dev_prompt)


if __name__ == "__main__":
    cli.run_app(server)
