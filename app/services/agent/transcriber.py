import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    MetricsCollectedEvent,
    RoomOutputOptions,
    StopResponse,
    WorkerOptions,
    cli,
    llm,
)
from livekit.plugins import assemblyai

load_dotenv()

logger = logging.getLogger("transcriber")

class Transcriber(Agent):
    def __init__(self):
        super().__init__(
            instructions="not-needed",
            stt=assemblyai.STT(end_of_turn_confidence_threshold=0.2,
                               min_end_of_turn_silence_when_confident=30,
                               max_turn_silence=50,
                               ),
        )
    
    async def on_user_turn_completed(self, chat_ctx: llm.ChatContext, new_message: llm.ChatMessage):
        logger.info(f" -> {new_message.text_content}")

        raise StopResponse()
    
async def inbound_entrypoint(ctx: JobContext):
    logger.info(f"starting transcriber (speech to text) example, room: {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    session = AgentSession()

    await session.start(
        agent=Transcriber(),
        room=ctx.room,
        room_output_options=RoomOutputOptions(
            transcription_enabled=True,
            audio_enabled=False,
        ),
    )