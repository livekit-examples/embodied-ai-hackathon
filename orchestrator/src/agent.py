import base64
import logging
import textwrap
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
TOOLS_IMAGE_PATH = Path(__file__).resolve().parents[2] / "images" / "tools.jpg"
# Pre-encode the image once at import time; it's static for the demo.
TOOLS_IMAGE_DATA_URL = "data:image/jpeg;base64," + base64.b64encode(
    TOOLS_IMAGE_PATH.read_bytes()
).decode("utf-8")

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
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
                """
            ),
        )

    @function_tool()
    async def plan_and_execute(self, context: RunContext, task_description: str) -> str:
        """Plan and execute a task on the toolbench.

        Args:
            task_description: A short natural-language description of the task to complete (e.g. "get the red screwdriver",
            "put away the blue screwdriver", "clean up the tools", "organize the tools").
        """
        context.disallow_interruptions()

        logger.info("Planning task: %s", task_description)

        planner_ctx = ChatContext()
        planner_ctx.add_message(
            role="system",
            content=(
                "You are a planner for an embodied robotic agent with two arms on linear sliders "
                "above a user's toolbench. Given an image of the current toolbench state and a task, "
                "produce a concise, numbered list of concrete physical steps the robot should perform. "
                "Reference tools by their visible appearance (color, type, location on the bench). "
                "Keep the plan short (3-7 steps) and unambiguous."
            ),
        )
        planner_ctx.add_message(
            role="user",
            content=[
                f"Task: {task_description}",
                "Current toolbench state:",
                ImageContent(image=TOOLS_IMAGE_DATA_URL),
            ],
        )

        plan = await self._planner_llm.chat(chat_ctx=planner_ctx).collect()
        logger.info("Planner produced steps:\n%s", plan.text)

        # TODO: actually dispatch each step to the robot arms.
        return plan.text


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):

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
        agent=Assistant(),
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


if __name__ == "__main__":
    cli.run_app(server)
