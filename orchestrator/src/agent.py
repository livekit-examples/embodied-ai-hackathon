import base64
import logging
import os
import textwrap
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, Union

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
from livekit.plugins import ai_coustics, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from pydantic import BaseModel, Field

# Repo-relative path to the toolbench image used as visual context for planning.
TOOLS_IMAGE_PATH = Path(__file__).resolve().parents[1] / "images" / "tools.jpg"
# Pre-encode the image once at import time; it's static for the demo.
TOOLS_IMAGE_DATA_URL = "data:image/jpeg;base64," + base64.b64encode(
    TOOLS_IMAGE_PATH.read_bytes()
).decode("utf-8")

logger = logging.getLogger("agent")

load_dotenv(".env.local")


# Plan schema: a discriminated union of robot actions. The planner LLM is
# constrained to output JSON that validates against `Plan` via OpenAI's
# strict structured-output mode (see `Orchestrator.plan_and_execute`).

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


Action = Annotated[Union[MoveTo, RunPolicy], Field(discriminator="kind")]


class Plan(BaseModel):
    """Ordered list of robot actions that fulfill a task."""

    steps: list[Action]


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


@asynccontextmanager
async def activity_status(ctx: RunContext, activity: str, detail: str = ""):
    """Publish the agent's current activity to the frontend via participant attributes.

    The frontend reads `agent.activity` (machine-readable phase) and
    `agent.activity.detail` (human-readable detail) off the agent participant
    to render a status badge while a tool is running. Empty strings clear
    the keys (LiveKit treats "" as deletion). The keys are always cleared
    on exit, including on exceptions or interruptions.

    In console mode (no RoomIO is wired up), falls back to logging so the
    agent still works locally without a room.
    """
    # `session.room_io` raises RuntimeError when the session was started without
    # a room (e.g. `agent.py console`); fall back to logging in that case.
    try:
        room = ctx.session.room_io.room
    except RuntimeError:
        room = None

    if room is None:
        logger.info("activity[%s]: %s", activity, detail or "<no detail>")
        try:
            yield
        finally:
            logger.info("activity[%s]: done", activity)
        return

    await room.local_participant.set_attributes(
        {
            "agent.activity": activity,
            "agent.activity.detail": detail,
        }
    )
    try:
        yield
    finally:
        await room.local_participant.set_attributes(
            {
                "agent.activity": "",
                "agent.activity.detail": "",
            }
        )


class Orchestrator(Agent):
    def __init__(self) -> None:
        # Dedicated planner LLM used inside tools for vision-based reasoning,
        # kept separate from the voice pipeline LLM to avoid mixing concerns.
        self._planner_llm = inference.LLM(model="openai/gpt-5.2-chat-latest")

        super().__init__(
            llm=inference.LLM(model="openai/gpt-5.2-chat-latest"),
            instructions=textwrap.dedent(
                """\
                You are the toolbench helper, an embodied AI agent equipped with two robotic arms on linear sliders. With these arms,
                you can reach any part of the user's toolbench to pick up and move around their tools at their request. Some examples of
                requests you can fulfill are:
                - "Can you get me the wrench?"
                - "Can you put away the blue screwdriver?"
                - "Can you clean up the tools"

                Based on the user's request, invoke the `plan_and_execute` function to plan and execute the task.
                Keep spoken responses short and confirm what you are doing in plain language.
                The tool returns a structured JSON plan describing the steps the robot will take; never read it aloud or recite kinds, arms, policies, or quoted strings from it. Just describe the action in one short sentence (e.g. "Picking up the red screwdriver now").
                """
            ),
        )

    @function_tool()
    async def plan_and_execute(self, context: RunContext, task_description: str) -> str:
        """Plan a task on the toolbench and return a JSON-encoded `Plan`.

        The returned value is a JSON object built from the robot's `move_to` /
        `run_policy` primitives that fulfills the task, e.g.
        `{"steps": [{"kind": "move_to", ...}, {"kind": "run_policy", ...}]}`.
        Briefly confirm the action to the user in plain language; do NOT read
        the JSON, kinds, policies, or arm IDs aloud.

        Args:
            task_description: A short natural-language description of the task to complete (e.g. "get the red screwdriver",
            "put away the blue screwdriver", "clean up the tools", "organize the tools").
        """
        context.disallow_interruptions()

        logger.info("Planning task: %s", task_description)

        planner_ctx = ChatContext()
        planner_ctx.add_message(role="system", content=_PLANNER_SYSTEM_PROMPT)
        planner_ctx.add_message(
            role="user",
            content=[
                _PLANNER_USER_PROMPT_HEADER + f"\n# Task: {task_description}\n",
                ImageContent(image=TOOLS_IMAGE_DATA_URL),
            ],
        )

        async with activity_status(context, "planning", task_description):
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

        # TODO: dispatch each step deterministically against the robot arms,
        # e.g. via `match step:` over MoveTo / RunPolicy.
        return plan.model_dump_json()


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
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
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
