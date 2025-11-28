import asyncio
import time
import os
import logging
from typing import Tuple, AsyncIterable
from contextlib import contextmanager

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    llm,
    stt,
    RoomInputOptions,
    RoomOutputOptions,
    metrics, 
    MetricsCollectedEvent,
    RunContext,
    ChatContext, 
    ChatMessage,

)
from livekit import rtc
from livekit.agents.llm import function_tool
from livekit.plugins import deepgram, openai, cartesia, silero, noise_cancellation, elevenlabs, assemblyai
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.openai import OpenAIEmbedding

from livekit.agents.voice.agent import ModelSettings
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins.turn_detector.english import EnglishModel

from app.services.agent.cache import semantic_context_cache

logger = logging.getLogger("inbound-agent")
for noisy_logger in ["pymongo", "pymongo.topology", "pymongo.connection"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

from collections import OrderedDict
from typing import Optional

class LRUCache:
    """Simple thread-safe LRU for async apps (no awaits needed)."""
    def __init__(self, max_size: int = 512):
        self.max_size = max_size
        self.cache = OrderedDict()

    def get(self, key: str) -> Optional[str]:
        if key not in self.cache:
            return None
        # mark as recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, key: str, value: str):
        self.cache[key] = value
        self.cache.move_to_end(key)

        # evict least recently used
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

# create global cache instance
mongo_lru_cache = LRUCache(max_size=512)

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
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
from functools import lru_cache

load_dotenv(override=True)

# ---------------------- TIMER UTILITY ----------------------
class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dur = time.perf_counter() - self.start
        print(f"\nTIMER: {self.name} took {dur:.4f} seconds")

# ---------------------- GLOBAL SETUP ----------------------
with Timer("Pinecone + Index Initialization"):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    pc_index = pc.Index("ai-tutor")
    # pc_index = pc.Index("prod-itsbot")
    vector_store = PineconeVectorStore(
        pinecone_index=pc_index,
        # namespace="f280790d-1517-456a-8954-2c296b38f8e1"
    )
    embed_model = OpenAIEmbedding(
        api_key=os.environ["OPENAI_API_KEY"],
        model="text-embedding-3-small"
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )
    retriever = index.as_retriever(similarity_top_k=1, use_async=True)

with Timer("Mongo Initialization"):
    mongo_client = AsyncIOMotorClient(os.environ["MONGO_URI"])
    db = mongo_client["itsbot-db"]
    parents_collection = db["parentdocs"]

# ---------------------- CACHED EMBEDDINGS ----------------------
@lru_cache(maxsize=1000)
def get_cached_embedding(text: str):
    """Cache embeddings for repeated/similar queries."""
    return embed_model.get_text_embedding(text)

# Run cache check and retriever fetch IN PARALLEL
async def check_cache(question):
    # with Timer("Cache Fetch"):
    return semantic_context_cache.get(question)

async def fetch_from_retriever(question):
    # with Timer("Retriever Fetch"):
    return await retriever.aretrieve(question)
# ---------------------- MAIN PIPELINE ----------------------
@llm.function_tool
async def ask_knowledge_base(context: RunContext, question: str):
    """Ultra-fast retrieval with streaming context"""
    # Send a verbal status update to the user after a short delay
    # async def _speak_status_update(delay: float = 0.5):
    #     # await asyncio.sleep(delay)
    #     await context.session.generate_reply(instructions=f"""
    #         You are searching the knowledge base for \"{question}\" but it is taking a little while.
    #         Update the user on your progress, but be very brief.
    #     """)
    
    # status_update_task = asyncio.create_task(_speak_status_update(0.5))

    # # Check if we have preemptive result (semantic match)
    # if preemptive_cache:
    #     preemptive_result = await preemptive_cache.get_result(question, timeout=0.5)
    #     if preemptive_result:
    #         print(f"⚡ Preemptive result found")
    #         return preemptive_result
        
    # print(f"⚡ No preemptive result, fetching now: '{question[:50]}...'")

    # Run cache and retriever in parallel
    cache_task = asyncio.create_task(check_cache(question))
    retriever_task = asyncio.create_task(fetch_from_retriever(question))
    
    # Wait for whichever completes first
    done, pending = await asyncio.wait(
        {cache_task, retriever_task},
        return_when=asyncio.FIRST_COMPLETED,
        timeout=0.5  # Max 500ms wait
    )
    print(f"Done: {done}\nPending: {pending}")
    # Check cache first
    if cache_task in done:
        cache_result = await cache_task
        if cache_result:
            matched_question, cached_context, similarity = cache_result
            print(f"✓ Cache Hit! Similarity: {similarity:.3f}")
            
            # Cancel pending retriever
            for task in pending:
                task.cancel()
            # status_update_task.cancel()
            return cached_context
    
    # Use retriever results
    if retriever_task in done:
        results = await retriever_task
    else:
        results = await retriever_task  # Wait if not done yet
    
    # Build context with LIMIT
    context_parts = [node.text for node in results[:3]]  # Limit to top 3 for speed
    context = "\n".join(context_parts)
    
    # Async cache update (fire and forget)
    asyncio.create_task(semantic_context_cache.set_async(question, context))
    
    # Cancel status update if search completed before timeout
    # status_update_task.cancel()

    return context
# ---------------------- PRE-WARM CONNECTIONS ----------------------
async def prewarm():
    """Pre-initialize HTTPS sessions and caches to avoid cold-start delay."""
    print("Prewarming connections...")
    try:
        _ = await retriever.aretrieve("ping")  # Warm Pinecone/OpenAI
    except Exception as e:
        print("Prewarm failed (harmless):", e)
    _ = get_cached_embedding("ping")  # Warm cache
    print("Prewarm complete.")

# ###### Inbound RAG Agent ######
# class InboundAgent(Agent):
#     def __init__(self):
#         super().__init__(
#             instructions=(
#                 "You are a Eminence Technology customer service AI assistant. "
#                 # "For ANY Eminence Technology-related or factual question, you MUST use the 'ask_knowledge_base' tool FIRST. "
#                 # "Do not rely on your internal memory. "
#                 # "After receiving the tool's output, use it to construct a conversational, human-like answer. "
#                 # "If the tool returns no relevant data, politely say you don't have enough information. "
#                 # "Keep responses concise and optimized for spoken delivery. PLEASE MAKE SURE THAT THE RESPONSES ARE SHORT SO THAT IT MIMICKS A PHONE CONVERSATION BETWEEN HUMANS. "
#                 # "Do not respond with asterick, bullet points,etc  please respond how you would in a normal conversation with a human. "
#                 # "PLEASE keep your tone friendly and enthusiastic. Always Respond politely to the customer. You are allowed to do small talks with the customer BUT DO NOT STRAY AWAY FROM THE BUSINESS AND OBJECTIVE OF THE CONVERSATION"
#                 # "Format numbers naturally (e.g., 'five hundred and twelve gigabytes')." \
#                 # "Please return the text with formatted emotion type before sentence to indicate the TTS model on which emotion to synthesie the speed with, for eg, [enthusiastically] Hello, how are you."
#             ),
#             stt=assemblyai.STT(
#                 end_of_turn_confidence_threshold=0.2,
#                 min_end_of_turn_silence_when_confident=0.3,
#                 max_end_of_turn_silence=0.5
#             ),
#             # stt=assemblyai.STT(model="universal-streaming-multilingual"),
#             llm=openai.LLM(tool_choice="auto", max_completion_tokens=50),
#             # tts=elevenlabs.TTS(),#model="eleven_v3",voice_id="EkK5I93UQWFDigLMpZcX"),
#             tts=cartesia.TTS
#             (
#                 model="sonic-3",
#                 voice="6ccbfb76-1fc6-48f7-b71d-91ac6298247b",
#                 emotion="Happy",
#                 speed=1.0,
#                 volume=2
#             ),
#             # vad=silero.VAD.load(min_speech_duration=0.2,
#             #                     min_silence_duration=0.3),
#             # turn_detection=EnglishModel(),
#             # preemptive_generation=True,
#             # tools=[get_current_time],# ask_knowledge_base],
#             min_endpointing_delay=0.1,  # Minimum wait after silence
#             max_endpointing_delay=0.5,  # Maximum wait before forcing turn end
#             allow_interruptions=True,
#             use_tts_aligned_transcript=False
#         )
#     # async def llm_node(
#     #     self, chat_ctx, tools, model_settings=None
#     # ):
#     #     """Optimized LLM node with minimal overhead"""
#     #     # with Timer("LLM Node:"):
#     #     async for chunk in super().llm_node(chat_ctx, tools, model_settings):
#     #         yield chunk 

#     # async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage):
#     #     """
#     #     Called when user finishes speaking - perfect time to start preemptive retrieval!
#     #     This happens BEFORE the LLM processes the message.
#     #     """
#     #     # Send a verbal status update to the user after a short delay
#     #     fast_llm_ctx = turn_ctx.copy(

#     #         exclude_instructions=True, exclude_function_call=True
#     #     ).truncate(max_items=3)
#     #     fast_llm_ctx.items.insert(0, self._fast_llm_prompt)
#     #     fast_llm_ctx.items.append(new_message)

#     #     filler_response_fut = asyncio.Future[str]()

#     #     async def _speak_status_update(delay: float = 0.5):
#     #         await asyncio.sleep(delay)
#     #         async for chunk in self.llm.chat(chat_ctx=fast_llm_ctx).to_str_iterable():
#     #             filler_response += chunk
#     #         await turn_ctx.session.generate_reply(instructions=f"""
#     #             You are searching the knowledge base for \"{new_message.text_content}\" but it is taking a little while.
#     #             Update the user on your progress, but be very brief.
#     #         """)
#     #         filler_response_fut.set_result(filler_response)

#     #     status_update_task = asyncio.create_task(_speak_status_update(0.2))

#     #     rag_content = await ask_knowledge_base(new_message.text_content)

#     #     status_update_task.cancel()
        
#     #     turn_ctx.add_message(role="assistant", content=rag_content)
#     #     await self.update_chat_ctx(turn_ctx)

#         # print(f"\n{'='*60}")
#         # print(f"🎤 User turn completed: '{new_message.content[:100]}...'")
#         # print(f"{'='*60}\n")
        
#         # user_question = new_message.text_content.strip()
        
#         # # Decide if we should preemptively retrieve
#         # should_retrieve = self._should_preemptively_retrieve(user_question)
        
#         # if should_retrieve and preemptive_cache:
#         #     print(f"⚡ Starting preemptive semantic retrieval...")
#         #     # Start retrieval in background (non-blocking)
#         #     asyncio.create_task(preemptive_cache.start_retrieval(user_question))
#         # else:
#         #     print(f"⏭ Skipping preemptive retrieval (doesn't look like a KB question)")
        
#         # await super().on_user_turn_completed(turn_ctx, new_message)
    
    
#     # def _should_preemptively_retrieve(self, question: str) -> bool:
#     #     """
#     #     Heuristic to decide if we should preemptively retrieve.
#     #     Returns True if question likely needs knowledge base.
#     #     """
#     #     question_lower = question.lower()
        
#     #     # Skip if too short or greeting
#     #     if len(question.split()) < 3:
#     #         return False
        
#     #     # Skip common greetings/small talk
#     #     greeting_patterns = [
#     #         "hello", "hi ", "hey", "good morning", "good afternoon",
#     #         "how are you", "thanks", "thank you", "bye", "goodbye"
#     #     ]
#     #     if any(pattern in question_lower for pattern in greeting_patterns):
#     #         return False
        
#     #     # Retrieve if contains question words or product-related terms
#     #     retrieve_indicators = [
#     #         "what", "how", "when", "where", "why", "can you",
#     #         "tell me", "explain", "information", "about",
#     #         "product", "price", "feature", "service", "support",
#     #         "eminence", "technology", "help with"
#     #     ]
        
#     #     return any(indicator in question_lower for indicator in retrieve_indicators)


#     # async def tts_node(self, text, model_settings):
#     #     return super().tts_node(text, model_settings)

class KnowledgeBaseAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a Eminence Technology customer service AI assistant. "
                # "For ANY Eminence Technology-related or factual question, you MUST use the 'ask_knowledge_base' tool FIRST. "
                # "Do not rely on your internal memory. "
                # "After receiving the tool's output, use it to construct a conversational, human-like answer. "
                # "If the tool returns no relevant data, politely say you don't have enough information. "
                # "Keep responses concise and optimized for spoken delivery. PLEASE MAKE SURE THAT THE RESPONSES ARE SHORT SO THAT IT MIMICKS A PHONE CONVERSATION BETWEEN HUMANS. "
                # "Do not respond with asterick, bullet points,etc  please respond how you would in a normal conversation with a human. "
                # "PLEASE keep your tone friendly and enthusiastic. Always Respond politely to the customer. You are allowed to do small talks with the customer BUT DO NOT STRAY AWAY FROM THE BUSINESS AND OBJECTIVE OF THE CONVERSATION"
                # "Format numbers naturally (e.g., 'five hundred and twelve gigabytes')." \
                # "Please return the text with formatted emotion type before sentence to indicate the TTS model on which emotion to synthesie the speed with, for eg, [enthusiastically] Hello, how are you."
            ),
            # stt=deepgram.STT(
            #     interim_results=True,
            #     endpointing_ms=0.1,
            #     mip_opt_out=True
            # ),
            stt=assemblyai.STT(
                end_of_turn_confidence_threshold=0.2,
                min_end_of_turn_silence_when_confident=0.3,
                max_turn_silence=0.5
            ),
            # stt=assemblyai.STT(model="universal-streaming-multilingual"),
            llm=openai.LLM(tool_choice="none", max_completion_tokens=50),
            # tts=elevenlabs.TTS(),#model="eleven_v3",voice_id="EkK5I93UQWFDigLMpZcX"),
            tts=cartesia.TTS
            (
                model="sonic-3",
                voice="6ccbfb76-1fc6-48f7-b71d-91ac6298247b",
                emotion="Happy",
                speed=1.0,
                volume=2
            ),
            # vad=silero.VAD.load(min_speech_duration=0.2,
            #                     min_silence_duration=0.3),
            # turn_detection=EnglishModel(),
            # preemptive_generation=True,
            # tools=[get_current_time],# ask_knowledge_base],
            min_endpointing_delay=0.1,  # Minimum wait after silence
            max_endpointing_delay=0.5,  # Maximum wait before forcing turn end
            allow_interruptions=True,
            use_tts_aligned_transcript=False
        )
        self.text_buffer = []
        self.buffer_timeout = 1.5  # seconds to wait before triggering KB search
        self.min_words = 5  # minimum words before triggering search
        
    async def stt_node(
        self, 
        audio: AsyncIterable[rtc.AudioFrame], 
        model_settings: ModelSettings
    ) -> Optional[AsyncIterable[stt.SpeechEvent]]:
        """
        Override STT node to buffer transcribed text and trigger knowledge base searches
        """
        # Get the default STT events
        events = Agent.default.stt_node(self, audio, model_settings)
        
        if events is None:
            return None
            
        # Wrap the events to add buffering logic
        async def buffered_events():
            buffer_task = None
            current_utterance = []
            
            async for event in events:
                # Pass through all events
                yield event
                
                # Buffer final transcriptions
                if isinstance(event, stt.SpeechEvent):
                    if event.type == stt.SpeechEventType.PREFLIGHT_TRANSCRIPT:
                        text = event.alternatives[0].text.strip()
                        if text:
                            current_utterance.append(text)
                            
                            # Cancel existing buffer task if new text arrives
                            if buffer_task and not buffer_task.done():
                                buffer_task.cancel()
                            
                            # Start new buffer timeout
                            buffer_task = asyncio.create_task(
                                self._process_buffered_text(current_utterance[:])
                            )
                    
                    elif event.type == stt.SpeechEventType.END_OF_SPEECH:
                        # Speech ended, process immediately if we have content
                        if current_utterance:
                            if buffer_task and not buffer_task.done():
                                buffer_task.cancel()
                            await self._process_buffered_text(current_utterance)
                            current_utterance = []
        
        return buffered_events()

    async def _process_buffered_text(self, utterance_parts: list[str]):
        """
        Wait for buffer timeout, then send accumulated text to knowledge base
        """
        await asyncio.sleep(self.buffer_timeout)
        
        full_text = " ".join(utterance_parts).strip()
        word_count = len(full_text.split())
        
        # Only trigger KB search if we have enough content
        if word_count >= self.min_words:
            print(f"🔍 Triggering KB search: '{full_text[:100]}...'")
            # Create a minimal context object for the KB function
            context = type('obj', (object,), {
                'session': self.session
            })()

            try:
                # Fetch knowledge base results
                updated_ctx = self.chat_ctx.copy()

                kb_context = await ask_knowledge_base(context, full_text)
                
                # Inject the KB context into the chat for the LLM
                if kb_context:
                    updated_ctx.add_message(
                        role="assistant",
                        content=f"[Knowledge Base Context]: {kb_context}"
                    )
                    await self.update_chat_ctx(updated_ctx)
                    print(f"✓ KB context injected ({len(kb_context)} chars)")
                            # Generate reply with the KB context
                    await self.session.generate_reply(
                        instructions="Use the Knowledge Base Context provided above to answer the user's question accurately and concisely."
                    )
                # async for chunk in super().llm_node(chat_ctx, tools, model_settings):
                #     yield chunk 
            except Exception as e:
                print(f"❌ KB search failed: {e}")



async def inbound_entrypoint(ctx: JobContext):
    # Prewarm in parallel with connection
    prewarm_task = asyncio.create_task(prewarm())
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    await prewarm_task  # Ensure prewarm completes
    
    agent = KnowledgeBaseAgent()
    session = AgentSession()

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
            close_on_disconnect=True,
        ),
    )
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)
    await session.say("Thanks for calling Eminence Technology customer support. My name is Lala, let me know how I can assist you")

import numpy as np
# Global cache for preemptive retrieval results
class PreemptiveSemanticCache:
    """
    Semantic cache for preemptive retrieval using cosine similarity.
    Stores both completed results and running tasks.
    """
    def __init__(self, embed_model, capacity=50, similarity_threshold=0.92):
        """
        Args:
            embed_model: OpenAI embedding model instance
            capacity: Maximum number of cached items
            similarity_threshold: Minimum cosine similarity (0-1) to consider a match
        """
        self.embed_model = embed_model
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        
        # Cache structure: question_text -> (embedding, result, timestamp)
        self._results_cache = OrderedDict()
        
        # Running tasks: question_text -> (embedding, task, timestamp)
        self._tasks_cache = OrderedDict()
        
        self._lock = asyncio.Lock()
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        emb1 = np.array(emb1, dtype=np.float32)
        emb2 = np.array(emb2, dtype=np.float32)
        
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def _find_similar(self, query_embedding: np.ndarray, cache_dict: OrderedDict) -> Optional[Tuple[str, float]]:
        """
        Find most similar question in cache.
        
        Returns:
            Tuple of (matched_question, similarity_score) if found, None otherwise
        """
        if not cache_dict:
            return None
        
        query_emb = np.array(query_embedding, dtype=np.float32)
        best_similarity = -1.0
        best_key = None
        
        for cached_question, (cached_emb, *_) in cache_dict.items():
            similarity = self._cosine_similarity(query_emb, cached_emb)
            if similarity > best_similarity:
                best_similarity = similarity
                best_key = cached_question
        
        if best_similarity >= self.similarity_threshold:
            return (best_key, best_similarity)
        
        return None
    
    async def start_retrieval(self, question: str) -> asyncio.Task:
        """Start retrieval in background and return the task."""
        # Get embedding for the question
        query_embedding = self.embed_model.get_query_embedding(question)
        
        async with self._lock:
            # Check if similar task already running
            similar = self._find_similar(query_embedding, self._tasks_cache)
            if similar:
                matched_question, similarity = similar
                print(f"✓ Similar task already running (sim={similarity:.3f}): '{matched_question[:50]}...'")
                return self._tasks_cache[matched_question][1]
            
            # Check if similar result already cached
            similar = self._find_similar(query_embedding, self._results_cache)
            if similar:
                matched_question, similarity = similar
                print(f"✓ Similar result already cached (sim={similarity:.3f}): '{matched_question[:50]}...'")
                # Return completed task with cached result
                result = self._results_cache[matched_question][1]
                return asyncio.create_task(self._return_result(result))
            
            # Start new retrieval task
            task = asyncio.create_task(self._retrieve(question, query_embedding))
            self._tasks_cache[question] = (query_embedding, task, asyncio.get_event_loop().time())
            
            # Evict oldest task if over capacity
            if len(self._tasks_cache) > self.capacity:
                old_key = next(iter(self._tasks_cache))
                old_task = self._tasks_cache[old_key][1]
                if not old_task.done():
                    old_task.cancel()
                del self._tasks_cache[old_key]
            
            return task
    
    async def _return_result(self, result):
        """Helper to return a result immediately."""
        return result
    
    async def _retrieve(self, question: str, query_embedding: np.ndarray):
        """Internal retrieval logic."""
        try:
            # Try semantic cache first
            cache_result = await check_cache(question)
            if cache_result:
                matched_question, cached_context, similarity = cache_result
                print(f"✓ Preemptive semantic cache hit! Similarity: {similarity:.3f}")
                result = cached_context
            else:
                # Fetch from retriever
                print(f"⚡ Preemptive retrieval started for: '{question[:50]}...'")
                results = await asyncio.wait_for(
                    fetch_from_retriever(question), 
                    timeout=2.0
                )
                result = "\n".join(node.text for node in results[:3])
                
                # Store in semantic cache (fire-and-forget)
                asyncio.create_task(semantic_context_cache.set_async(question, result))
            
            # Store result in cache
            async with self._lock:
                self._results_cache[question] = (query_embedding, result, asyncio.get_event_loop().time())
                
                # Move to end (most recently used)
                self._results_cache.move_to_end(question)
                
                # Evict oldest result if over capacity
                if len(self._results_cache) > self.capacity:
                    self._results_cache.popitem(last=False)
                
                # Remove from tasks cache
                if question in self._tasks_cache:
                    del self._tasks_cache[question]
            
            print(f"✓ Preemptive retrieval completed for: '{question[:50]}...'")
            return result
            
        except asyncio.TimeoutError:
            print(f"⚠ Preemptive retrieval timeout for: '{question[:50]}...'")
            return None
        except Exception as e:
            print(f"✗ Preemptive retrieval error: {e}")
            return None
    
    async def get_result(self, question: str, timeout: float = 0.5) -> Optional[str]:
        """Get result using semantic similarity search."""
        # Get embedding for the question
        query_embedding = self.embed_model.get_query_embedding(question)
        
        async with self._lock:
            # Check results cache first (semantic search)
            similar = self._find_similar(query_embedding, self._results_cache)
            if similar:
                matched_question, similarity = similar
                # Move to end (LRU)
                self._results_cache.move_to_end(matched_question)
                result = self._results_cache[matched_question][1]
                print(f"✓ Using preemptive result (sim={similarity:.3f}): '{matched_question[:50]}...'")
                return result
            
            # Check if similar task is running
            similar = self._find_similar(query_embedding, self._tasks_cache)
            if not similar:
                return None
            
            matched_question, similarity = similar
            task = self._tasks_cache[matched_question][1]
        
        # Wait for task if still running
        if not task.done():
            try:
                print(f"⏳ Waiting for preemptive task (sim={similarity:.3f})...")
                result = await asyncio.wait_for(task, timeout=timeout)
                if result:
                    print(f"✓ Preemptive task completed just in time!")
                return result
            except asyncio.TimeoutError:
                print(f"⚠ Preemptive task timeout, fetching fresh")
                return None
        else:
            # Task already done
            try:
                result = task.result()
                print(f"✓ Retrieved completed preemptive task (sim={similarity:.3f})")
                return result
            except Exception:
                return None
    
    def clear(self):
        """Clear all cached results and cancel pending tasks."""
        for _, (_, task, _) in self._tasks_cache.items():
            if not task.done():
                task.cancel()
        self._results_cache.clear()
        self._tasks_cache.clear()


preemptive_cache = PreemptiveSemanticCache(
    embed_model=embed_model,
    capacity=50,              # Adjust based on memory
    similarity_threshold=0.92  # Higher = stricter matching
)