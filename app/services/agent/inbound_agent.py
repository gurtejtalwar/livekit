

# ---------------------- MAIN PIPELINE ----------------------

###### Inbound RAG Agent ######
class InboundAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a Eminence Technology customer service AI assistant. "
                "For ANY Eminence Technology-related or factual question, you MUST use the 'ask_knowledge_base' tool FIRST. "
                "Do not rely on your internal memory. "
                "After receiving the tool's output, use it to construct a conversational, human-like answer. "
                "If the tool returns no relevant data, politely say you don't have enough information. "
                "Keep responses concise and optimized for spoken delivery. PLEASE MAKE SURE THAT THE RESPONSES ARE SHORT SO THAT IT MIMICKS A PHONE CONVERSATION BETWEEN HUMANS. "
                "Do not respond with asterick, bullet points,etc  please respond how you would in a normal conversation with a human. "
                "PLEASE keep your tone friendly and enthusiastic. Always Respond politely to the customer. You are allowed to do small talks with the customer BUT DO NOT STRAY AWAY FROM THE BUSINESS AND OBJECTIVE OF THE CONVERSATION"
                "Format numbers naturally (e.g., 'five hundred and twelve gigabytes')." \
                # "Please return the text with formatted emotion type before sentence to indicate the TTS model on which emotion to synthesie the speed with, for eg, [enthusiastically] Hello, how are you."
            ),
            stt=deepgram.STT(),
            # stt=assemblyai.STT(),
            # stt=assemblyai.STT(model="universal-streaming-multilingual"),
            # llm=openai.LLM(model="gpt-4o-mini", tool_choice="auto", max_completion_tokens=50),
            llm=groq.LLM(model="qwen/qwen3-32b", tool_choice="auto", max_completion_tokens=100),
            # tts=elevenlabs.TTS(),#model="eleven_v3",voice_id="EkK5I93UQWFDigLMpZcX"),
            tts=cartesia.TTS
            (
                model="sonic-turbo",
                voice="6ccbfb76-1fc6-48f7-b71d-91ac6298247b",
                emotion="Happy",
                speed=1.0,
                volume=2
            ),
            turn_detection=EnglishModel(),
            tools=[get_current_time, ask_knowledge_base],
            min_endpointing_delay=0.05,  # Minimum wait after silence
            max_endpointing_delay=0.3,  # Maximum wait before forcing turn end
            allow_interruptions=True,
            use_tts_aligned_transcript=False
        )

def content_to_string(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )

    return ""