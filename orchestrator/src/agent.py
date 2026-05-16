import asyncio
import base64
import json
import logging
import os
import textwrap
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
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
from livekit.agents.llm import ChatContext, ImageContent
from livekit.agents.llm.async_toolset import AsyncRunContext, AsyncToolset
from livekit.plugins import ai_coustics, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from pydantic import BaseModel, Field

# Repo-relative path to the toolbench image used as visual context for planning.
TOOLS_IMAGE_PATH = Path(__file__).resolve().parents[1] / "images" / "real_setup.jpg"
# Pre-encode the image once at import time; it's static for the demo.
TOOLS_IMAGE_DATA_URL = "data:image/jpeg;base64," + base64.b64encode(
    TOOLS_IMAGE_PATH.read_bytes()
).decode("utf-8")

logger = logging.getLogger("agent")

load_dotenv(".env.local")


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



# TODO: replace the sleep with an RPC to the robot worker, e.g.
#     await room.local_participant.perform_rpc(
#         destination_identity="robot",
#         method="execute_step",
#         payload=step.model_dump_json(),
#     )

async def _execute_run_policy(step: RunPolicy, local_participant) -> None:
    logger.warning("Not wired to RPC yet")
    await asyncio.sleep(_STUB_STEP_DURATION_S)

async def _execute_move_to(step: MoveTo, local_participant) -> None:
    logger.warning("Not wired to RPC yet")
    await asyncio.sleep(_STUB_STEP_DURATION_S)

async def _execute_step(step: Action, room) -> None:
    """Run a single plan step."""
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
) -> None:
    """Run each step of `plan` in sequence, flipping per-step status and narrating aloud.

    On success, the publisher is left in `phase: "completed"` so the UI can
    keep the finished list visible until the next plan replaces it.
    """
    # `session.room_io` raises RuntimeError when running without a room (e.g.
    # `agent.py console`); the executors handle a None room gracefully.
    try:
        room = ctx.session.room_io.room
    except RuntimeError:
        room = None
    for i, step in enumerate(plan.steps):
        label = _describe_step(step)
        await pub.set_step_status(i, "in_progress")
        await ctx.update(label)
        await _execute_step(step, room)
        await pub.set_step_status(i, "completed")
    await pub.set_completed()


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
    - Pick `arm` based on which side of the toolbench the object is on,
      preferring "right" when ambiguous.
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

StepStatus = Literal["pending", "in_progress", "completed"]
PlanPhase = Literal["idle", "planning", "executing", "completed"]


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

    def __init__(self, planner_llm: inference.LLM) -> None:
        super().__init__(id="planner")
        self._planner_llm = planner_llm

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

        await ctx.update(f"Planning how to {task_description}")

        planner_ctx = ChatContext()
        planner_ctx.add_message(role="system", content=_PLANNER_SYSTEM_PROMPT)
        planner_ctx.add_message(
            role="user",
            content=[
                _PLANNER_USER_PROMPT_HEADER + f"\n# Task: {task_description}\n",
                ImageContent(image=TOOLS_IMAGE_DATA_URL),
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
            await execute_plan(ctx, plan, pub)

        return f"Completed: {task_description}"


class Orchestrator(Agent):
    def __init__(self) -> None:
        # Dedicated planner LLM used inside the planner toolset for
        # vision-based reasoning, kept separate from the voice pipeline LLM
        # to avoid mixing concerns.
        planner_llm = inference.LLM(model="openai/gpt-5.2-chat-latest")

        super().__init__(
            llm=inference.LLM(model="openai/gpt-5.2-chat-latest"),
            instructions=textwrap.dedent(
                """\
                You are the toolbench helper, an embodied AI agent equipped with two robotic arms on linear sliders.
                You can pick up and move the user's tools at their request, e.g.:
                - "Can you get me the wrench?"
                - "Can you put away the blue screwdriver?"
                - "Can you clean up the tools"

                Use the `plan_and_execute` tool for any such request. The tool runs in the background and
                narrates progress aloud as it works (planning, moving, picking up, etc.). After it returns,
                briefly confirm completion in one short sentence (e.g. "Done!" or "All set, here you go.").
                Do not restate the individual steps; they have already been narrated.
                """
            ),
            tools=[PlannerToolset(planner_llm)],
        )


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
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await session.start(
        agent=Orchestrator(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S
                ),
            ),
        ),
    )

    await ctx.connect()

    # Dev shortcut: simulate a user utterance at launch so we can iterate on
    # plan_and_execute without speaking into a mic each time. Set the env var
    # to any prompt, e.g. DEV_SIMULATE_PROMPT="pick up the red screwdriver".
    # dev_prompt = os.getenv("DEV_SIMULATE_PROMPT")
    # if dev_prompt:
    #     logger.info("DEV_SIMULATE_PROMPT set, simulating user input: %r", dev_prompt)
    #     session.generate_reply(user_input=dev_prompt)


if __name__ == "__main__":
    cli.run_app(server)
