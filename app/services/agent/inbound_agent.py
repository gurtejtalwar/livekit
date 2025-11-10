import asyncio
import logging
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    llm,
    function_tool
)
from livekit.plugins import deepgram, openai, cartesia, silero
from llama_index.core.schema import MetadataMode
from livekit.agents.voice.agent import ModelSettings

load_dotenv()
logger = logging.getLogger("inbound-agent")

# Function tools to enhance your agent's capabilities
@function_tool
async def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return f"The current time is {datetime.now().strftime('%I:%M %p')}" 

###### Pinecone Vector DB Loader ######
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone
# (Make sure you have PINECONE_API_KEY and PINECONE_ENV set in .env)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# name of the existing Pinecone index
PINECONE_INDEX_NAME = "your-existing-index-name"

# connect to existing Pinecone index
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

# create LlamaIndex vector store wrapper
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# create a storage context using the Pinecone vector store
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# load index from the Pinecone vector store
# (if the index was previously built and stored in Pinecone)
index = load_index_from_storage(storage_context)

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
            tools=[get_current_time],
            vad=silero.VAD.load(),
            stt=deepgram.STT(),
            llm=openai.LLM(),
            tts=openai.TTS(),
        )
    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool],
        model_settings: ModelSettings,
    ):
        user_msg = chat_ctx.items[-1]
        assert isinstance(user_msg, llm.ChatMessage) and user_msg.role == "user"
        user_query = user_msg.text_content
        assert user_query is not None

        retriever = self.index.as_retriever()
        nodes = await retriever.aretrieve(user_query)

        instructions = "Context that might help answer the user's question:"
        for node in nodes:
            node_content = node.get_content(metadata_mode=MetadataMode.LLM)
            instructions += f"\n\n{node_content}"

        # update the instructions for this turn, you may use some different methods
        # to inject the context into the chat_ctx that fits the LLM you are using
        system_msg = chat_ctx.items[0]
        if isinstance(system_msg, llm.ChatMessage) and system_msg.role == "system":
            # TODO(long): provide an api to update the instructions of chat_ctx
            system_msg.content.append(instructions)
        else:
            chat_ctx.items.insert(0, llm.ChatMessage(role="system", content=[instructions]))
        print(f"update instructions: {instructions[:100].replace('\n', '\\n')}...")

        # update the instructions for agent
        # await self.update_instructions(instructions)

        return Agent.default.llm_node(self, chat_ctx, tools, model_settings)

###### Inbound RAG Agent ######

async def entrypoint(ctx: JobContext):
    """Main entry point for the telephony voice agent."""
    await ctx.connect()
    
    # Wait for participant (caller) to join
    participant = await ctx.wait_for_participant()
    logger.info(f"Phone call connected from participant: {participant.identity}")
    
    
    # Configure the voice processing pipeline optimized for telephony
    session = AgentSession(
        # Voice Activity Detection
        vad=silero.VAD.load(),
        
        # Speech-to-Text - Deepgram Nova-3
        stt=deepgram.STT(
            model="nova-3",  # Latest model
            language="en-US",
            interim_results=True,
            punctuate=True,
            smart_format=True,
            filler_words=True,
            endpointing_ms=25,
            sample_rate=16000
        ),
        
        # Large Language Model - GPT-4o-mini
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.7
        ),
        
        # Text-to-Speech - Cartesia Sonic-2
        tts=cartesia.TTS(
            model="sonic-2",
            voice="a0e99841-438c-4a64-b679-ae501e7d6091",  # Professional female voice
            language="en",
            speed=1.0,
            sample_rate=24000
        )
    )
    
    # Start the agent session
    await session.start(agent=agent, room=ctx.room)
    
    # Generate personalized greeting based on time of day
    import datetime
    hour = datetime.datetime.now().hour
    if hour < 12:
        time_greeting = "Good morning"
    elif hour < 18:
        time_greeting = "Good afternoon"
    else:
        time_greeting = "Good evening"
    
    await session.generate_reply(
        instructions=f"""Say '{time_greeting}! Thank you for calling. Whats happening?'
        Speak warmly and professionally at a moderate pace."""
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
