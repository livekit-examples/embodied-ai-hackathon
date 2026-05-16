import base64
import logging
import textwrap
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ChatContext,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
)
from livekit.agents.llm import ImageContent
from livekit.plugins import ai_coustics, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents import function_tool, RunContext

# Repo-relative path to the demo image fed into the agent's initial context.
TOOLS_IMAGE_PATH = Path(__file__).resolve().parents[2] / "images" / "tools.jpg"

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            chat_ctx=chat_ctx,
            # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
            # See all available models at https://docs.livekit.io/agents/models/llm/
            llm=inference.LLM(model="openai/gpt-5.2-chat-latest"),
            # To use a realtime model instead of a voice pipeline, replace the LLM
            # with a RealtimeModel and remove the STT/TTS from the AgentSession
            # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/)
            # 1. Install livekit-agents[openai]
            # 2. Set OPENAI_API_KEY in .env.local
            # 3. Add `from livekit.plugins import openai` to the top of this file
            # 4. Replace the llm argument with:
            #     llm=openai.realtime.RealtimeModel(voice="marin")
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
    async def plan_and_execute(self, context: RunContext, task_description: str):
        """Plan and execute a task on the toolbench.

        Args:
            task_description: A short natural-language description of the task to complete (e.g. "get the red screwdriver",
            "put away the blue screwdriver", "clean up the tools", "organize the tools").
        """

        # TODO: query the LLM "descirbe the steps to complete the task: {task_description}. Provide the tool image"

        context.disallow_interruptions()

server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # Seed the agent's chat context with a local image so it can answer
    # questions like "what do you see?" without needing a frontend upload.
    image_bytes = TOOLS_IMAGE_PATH.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    initial_ctx = ChatContext()
    initial_ctx.add_message(
        role="user",
        content=[
            "Current state of the user's toolbench",
            ImageContent(image=f"data:image/jpeg;base64,{image_b64}"),
        ],
    )

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(chat_ctx=initial_ctx),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S
                ),
            ),
        ),
    )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = anam.AvatarSession(
    #     persona_config=anam.PersonaConfig(
    #         name="...",
    #         avatarId="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/anam
    #     ),
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
