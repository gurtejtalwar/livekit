from bson import ObjectId

from app.shared import models
from app.agents import AgentConfig
from app.agents.prompt import inbound
from app.agents import UserData

async def load_agent_runtime_config(agent_id: str, user_data: UserData):
    agent: models.VoiceAgent = models.VoiceAgent.objects(
        id=ObjectId(agent_id)).first()
    if not agent:
        raise ValueError("Agent not found")

    config_doc: models.VoiceAgentConfigLivekit = models.VoiceAgentConfigLivekit.objects(
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
        f"\nUser Data: \n"
        f"Caller Name: {user_data.name}\n "
        f"Caller Email: {user_data.email}\n "
        f"Caller Phone: {user_data.phone}\n"
        f"Agent ID: {user_data.agent_id}\n"
        f"Caller ID: {user_data.user_id}\n"
        f"Caller Current Time: {user_data.user_current_time}\n"
        f"Caller Timezone: {user_data.user_timezone}\n"
    )
    lk_prompt = inbound.lk_prompt.format(
        agent_name=agent.agentName,
        admin_goal=system_prompt,
        language=config_doc.language if config_doc and config_doc.language else "English",
        additional_languages=", ".join(config_doc.additionalLanguages) if config_doc and config_doc.additionalLanguages else [],
        time=get_time_in_timezone(config_doc.timezone),
        timezone=config_doc.timezone
    )
    # ---------- LLM ----------
    llm = config_doc.llm if config_doc and config_doc.llm else {}
    llm_provider = llm.get("provider", "groq")
    llm_model = llm.get("model", "qwen/qwen3-32b")
    max_tokens = llm.get("max_tokens", 1000)

    # ---------- TTS ----------
    tts = config_doc.tts if config_doc and config_doc.tts else {}
    tts_provider = tts.get("provider", "elevenlabs")
    voice_id = tts.get("voice_id", config_doc.voiceType if config_doc else None)
    speed = tts.get("speed", 0.5)
    volume = tts.get("volume", 2.0)

    # ---------- STT ----------
    stt = config_doc.stt if config_doc and config_doc.stt else {}
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
        user_id=str(user_data.user_id),
        agent_id=str(agent.id),
        agent_name=agent.agentName,
        knowledge_base_id=agent.knowledgeBaseId,
        system_prompt=lk_prompt,
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


from zoneinfo import ZoneInfo
from datetime import datetime

def get_time_in_timezone(tz_name: str):
    return datetime.now(ZoneInfo(tz_name))
