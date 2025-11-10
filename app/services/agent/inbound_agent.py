import asyncio
import os
import logging
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
    RoomInputOptions,
    RoomOutputOptions,
    function_tool
)
from livekit.plugins import deepgram, openai, cartesia, silero, noise_cancellation
from llama_index.core.schema import MetadataMode
from livekit.agents.voice.agent import ModelSettings
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("inbound-agent")


# Function tools to enhance your agent's capabilities
@function_tool
async def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return f"The current time is {datetime.now().strftime('%I:%M %p')}" 

###### Pinecone Vector DB Loader ######
from pathlib import Path
from dotenv import load_dotenv
import os
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

###### PINECONE SETUP ######
# Initialize Pinecone client
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("ai-tutor")

# Wrap in a LlamaIndex vector store
vector_store = PineconeVectorStore(pinecone_index=index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

@function_tool
async def ask_knowledge_base(question: str) -> str:
    """Query the Pinecone index (knowledge base) for relevant information."""
    query_engine = index.as_query_engine(similarity_top_k=2)
    response = query_engine.query(question)
    return response.response
###### PINECONE SETUP ######

###### Inbound RAG Agent ######
class InboundAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=("""You are a friendly and helpful AI assistant answering phone calls.
            
            Your personality:
            - Professional yet warm and approachable
            - Speak clearly and at a moderate pace for phone calls
            - Keep responses concise but complete
            - Ask clarifying questions when needed
            
            Your capabilities:
            - Answer questions on a wide range of topics
            - Provide weather information when asked
            - Tell the current time
            - Have natural conversations
            
            Always identify yourself as an AI assistant when asked.
            Keep responses conversational and under 30 seconds for phone clarity."""),
            tools=[get_current_time, ask_knowledge_base],
        )
    #     self.index = index
    # async def llm_node(
    #     self,
    #     chat_ctx: llm.ChatContext,
    #     tools: list[llm.FunctionTool],
    #     model_settings: ModelSettings,
    # ):
    #     user_msg = chat_ctx.items[-1]
    #     assert isinstance(user_msg, llm.ChatMessage) and user_msg.role == "user"
    #     user_query = user_msg.text_content
    #     assert user_query is not None

    #     retriever = self.index.as_retriever()
    #     nodes = await retriever.aretrieve(user_query)

    #     instructions = "Context that might help answer the user's question:"
    #     for node in nodes:
    #         node_content = node.get_content(metadata_mode=MetadataMode.LLM)
    #         instructions += f"\n\n{node_content}"

    #     # update the instructions for this turn, you may use some different methods
    #     # to inject the context into the chat_ctx that fits the LLM you are using
    #     system_msg = chat_ctx.items[0]
    #     if isinstance(system_msg, llm.ChatMessage) and system_msg.role == "system":
    #         # TODO(long): provide an api to update the instructions of chat_ctx
    #         system_msg.content.append(instructions)
    #     else:
    #         chat_ctx.items.insert(0, llm.ChatMessage(role="system", content=[instructions]))
    #     print(f"update instructions: {instructions[:100].replace('\n', '\\n')}...")

    #     # update the instructions for agent
    #     # await self.update_instructions(instructions)

    #     return Agent.default.llm_node(self, chat_ctx, tools, model_settings)

###### Inbound RAG Agent ######

# async def entrypoint(ctx: JobContext):
#     """Main entry point for the telephony voice agent."""
#     await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
#     # Wait for participant (caller) to join
#     participant = await ctx.wait_for_participant()
#     logger.info(f"Phone call connected from participant: {participant.identity}")
    
    
#     # Configure the voice processing pipeline optimized for telephony
#     agent = InboundAgent(index)
#     session = AgentSession(),
#     await session.start(agent=agent, room=ctx.room)

#     await session.say("Hey, how can I help you today?", allow_interruptions=True)

async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    agent = InboundAgent()
    session = AgentSession(
        stt=deepgram.STT(language="multi"),
        llm=openai.LLM(),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=InboundAgent(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
            close_on_disconnect=False,
        ),
        room_output_options=RoomOutputOptions(
            audio_enabled=True,  # Avatar handles audio
        ),
    )

    await session.generate_reply(
        instructions="Greet the user as an iPear customer service representative. Speak english."
    )

if __name__ == "__main__":
    # Configure logging for better debugging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the agent with the name that matches your dispatch rule
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="inbound-agent"  # This must match your dispatch rule
    ))
