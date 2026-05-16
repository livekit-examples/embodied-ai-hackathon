import base64
import logging
import os
import textwrap
from contextlib import asynccontextmanager
from pathlib import Path

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

# Repo-relative path to the toolbench image used as visual context for planning.
TOOLS_IMAGE_PATH = Path(__file__).resolve().parents[1] / "images" / "tools.jpg"
# Pre-encode the image once at import time; it's static for the demo.
TOOLS_IMAGE_DATA_URL = "data:image/jpeg;base64," + base64.b64encode(
    TOOLS_IMAGE_PATH.read_bytes()
).decode("utf-8")

logger = logging.getLogger("agent")

load_dotenv(".env.local")


# Code as Policies prompt: the planner must emit Python code that drives the
# robot via a tiny API, not English prose. See arXiv:2209.07753.
_PLANNER_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You write robot policy code for an embodied agent with two arms on linear sliders
    above a user's toolbench, in the style of "Code as Policies" (Liang et al., 2022).

    Output contract:
    - Output ONLY valid Python source code. No prose, no explanations, no markdown
      code fences, no leading/trailing commentary.
    - Use ONLY the two API functions provided in the Hints (move_to, run_policy).
      Do not invent new functions, do not import anything, do not call any other names.
    - Reference tools by short, visually grounded descriptions (color + type, e.g.
      "red screwdriver", "claw hammer") that match what is visible in the image.
    - Keep the program straight-line and minimal: one statement per line, no helper
      functions, no loops or conditionals unless the task genuinely requires them.
    """
)


# Hints + Examples block, appended to a user message that also carries the
# toolbench image and the new "# Task: ..." comment.
_PLANNER_USER_PROMPT_HEADER = textwrap.dedent(
    '''\
    # Available robot API:
    from robot import move_to, run_policy
    # move_to(object_name: str, arm: str) -> None
    #   Positions `arm` ("left" | "right") over `object_name` so a subsequent
    #   policy can act on it. `object_name` is a short visual description
    #   the downstream policy can recognize (e.g. "red screwdriver", "claw hammer").
    # run_policy(policy_name: str, arm: str) -> None
    #   Runs a low-level policy on `arm`. `policy_name` must be one of:
    #   "pick-up", "put-down".

    # Task: hand me the claw hammer.
    move_to("claw hammer", arm="right")
    run_policy("pick-up", arm="right")

    # Task: put the blue painter's tape away in the toolbox.
    move_to("blue painter's tape", arm="left")
    run_policy("pick-up", arm="left")
    move_to("toolbox", arm="left")
    run_policy("put-down", arm="left")

    # Task: clear the needle-nose pliers and the utility knife off the bench.
    move_to("needle-nose pliers", arm="left")
    run_policy("pick-up", arm="left")
    move_to("utility knife", arm="right")
    run_policy("pick-up", arm="right")
    move_to("toolbox", arm="left")
    run_policy("put-down", arm="left")
    move_to("toolbox", arm="right")
    run_policy("put-down", arm="right")
    '''
)


def _extract_python(text: str) -> str:
    """Strip an optional surrounding ```python ... ``` (or plain ```) markdown fence.

    LLMs frequently wrap code in fences even when told not to. We only handle the
    common single-fence case; anything else is returned as-is for the caller to
    deal with.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    # Drop the opening fence line (``` or ```python or ```py).
    first_newline = stripped.find("\n")
    if first_newline == -1:
        return stripped
    body = stripped[first_newline + 1 :]

    if body.endswith("```"):
        body = body[: -len("```")]

    return body.strip()


@asynccontextmanager
async def activity_status(ctx: RunContext, activity: str, detail: str = ""):
    """Publish the agent's current activity to the frontend via participant attributes.

    The frontend reads `agent.activity` (machine-readable phase) and
    `agent.activity.detail` (human-readable detail) off the agent participant
    to render a status badge while a tool is running. Empty strings clear
    the keys (LiveKit treats "" as deletion). The keys are always cleared
    on exit, including on exceptions or interruptions.
    """
    room = ctx.session.room_io.room
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
                The tool returns Python robot code; never read it aloud or recite function names, arm IDs, or quoted strings from it. Just describe the action in one short sentence (e.g. "Picking up the red screwdriver now").
                """
            ),
        )

    @function_tool()
    async def plan_and_execute(self, context: RunContext, task_description: str) -> str:
        """Plan a task on the toolbench and return executable robot policy code.

        The returned value is a short Python program built from the robot's
        `move_to` / `run_policy` primitives that fulfills the task. Briefly
        confirm the action to the user in plain language; do NOT read the code,
        function names, or arm IDs aloud.

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
            response = await self._planner_llm.chat(chat_ctx=planner_ctx).collect()
            code = _extract_python(response.text)
            logger.info("Generated policy code:\n%s", code)

        # TODO: actually dispatch the generated code to the robot arms.
        return code


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
