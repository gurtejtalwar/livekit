from bson import ObjectId

from app.shared import models
from app.agent import AgentConfig

def load_agent_runtime_config(agent_id: str, user_data):
    agent: models.VoiceAgent = models.VoiceAgent.objects(
        id=ObjectId(agent_id)).first()
    if not agent:
        raise ValueError("Agent not found")

    config_doc: models.VoiceAgentConfig = models.VoiceAgentConfig.objects(
        agentId=agent.id).first()
    voice_doc: models.VoiceAgentVoiceConfig = models.VoiceAgentVoiceConfig.objects(
        agentId=agent.id).first()
    identity_doc = models.VoiceAgentIdentity.objects(agentId=agent.id).first()
    advanced_doc = models.VoiceAgentAdvancedSettings.objects(agentId=agent.id).first()

    # ---------- SYSTEM PROMPT ----------
    system_prompt = (
        config_doc.systemPrompt
        if config_doc and config_doc.systemPrompt
        else agent.agentConfig.get("systemPrompt")
        if agent.agentConfig
        else ""
    )

    system_prompt += (
        f"\nUser Data: Name: {user_data.name}, "
        f"Email: {user_data.email}, "
        f"Phone: {user_data.phone}\n"
    )

    # ---------- LLM ----------
    llm = config_doc.llm if config_doc and config_doc.llm else {}
    llm_provider = llm.get("provider", "groq")
    llm_model = llm.get("model", "qwen/qwen3-32b")
    max_tokens = llm.get("max_tokens", 1000)

    # ---------- TTS ----------
    tts = voice_doc.tts if voice_doc and voice_doc.tts else {}
    tts_provider = tts.get("provider", "elevenlabs")
    voice_id = tts.get("voice_id", voice_doc.voiceType if voice_doc else None)
    speed = tts.get("speed", 0.75)
    volume = tts.get("volume", 2.0)

    # ---------- STT ----------
    stt = voice_doc.stt if voice_doc and voice_doc.stt else {}
    stt_provider = stt.get("provider", "deepgram")

    # ---------- TOOLS ----------
    tools = config_doc.tools if config_doc and config_doc.tools else []

    # ---------- GREETING ----------
    greeting = (
        config_doc.welcomeMessage
        if config_doc and config_doc.welcomeMessage
        else "Hello! How can I assist you today?"
    )

    return AgentConfig(
        user_id=str(user_data.id),
        agent_id=str(agent.id),
        agent_name=agent.agentName,
        knowledge_base_id=agent.knowledgeBaseId,
        system_prompt=system_prompt,
        llm_provider=llm_provider,
        llm_model=llm_model,
        max_tokens=max_tokens,
        tts_provider=tts_provider,
        voice_id=voice_id,
        speed=speed,
        volume=volume,
        stt_provider=stt_provider,
        tools=tools,
        greeting=greeting,
    )

