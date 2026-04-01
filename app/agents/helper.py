from bson import ObjectId
import logging
from app.shared import models
from app import agents
from app.agents.prompt import inbound
from app.agents import UserData
from app.models.call_models import get_lead_by_phone

logger = logging.getLogger(__name__)

async def get_user_data(user_id: str, phone_number: str): #TODO HAZARD
    lead = await get_lead_by_phone(phone_number, user_id)

    if lead is not None:
        return UserData(
            user_id=str(lead.user_id),
            name=lead.first_name,
            email=lead.email,
            phone=lead.phone
        )
    else:
        return UserData(
            user_id=None,
            name=None,
            email=None,
            phone=phone_number
        )

async def load_agent_runtime_config(agent_id: str, user_data: UserData):
    workflow_doc = None
    agent: models.VoiceAgent = models.VoiceAgent.objects(
        id=ObjectId(agent_id)).first()
    if not agent:
        raise ValueError("Agent not found")
    if agent.status!="active":
        logger.info("Agent is Inactive")
        return ValueError("Agent is Inactive") #TODO need to find error interceptor for LK instead of returning
    
    config_doc: models.VoiceAgentConfig = models.VoiceAgentConfig.objects(
        agentId=agent.id).first()
    voice_config_doc: models.VoiceAgentVoiceConfig = models.VoiceAgentVoiceConfig.objects(
        agentId=agent.id).first()
    identity_doc: models.VoiceAgentIdentity = models.VoiceAgentIdentity.objects(agentId=agent.id).first()
    advanced_doc: models.VoiceAgentAdvancedSettings = models.VoiceAgentAdvancedSettings.objects(agentId=agent.id).first()
    escalation_doc: models.VoiceAgentEscalation = models.VoiceAgentEscalation.objects(agentId=agent.id).first()
    # voice_doc: models.VoiceAgentVoiceConfig = models.VoiceAgentVoiceConfig.objects(
    #     agentId=agent.id).first()

    if config_doc.isWorkflowEnabled:
        workflow_doc: models.VoiceAgentWorkflow = models.VoiceAgentWorkflow.objects(agentId=agent.id).first()
    
    human_phone_number = None
    if escalation_doc.humanEscalationEnabled is True and escalation_doc.teamMembers:
        team_member_id = escalation_doc.teamMembers[0]["_id"]
        voicebot_settings_doc: models.VoiceBotSettings = models.VoiceBotSettings.objects(userId=ObjectId(team_member_id)).first()
        human_phone_number = voicebot_settings_doc.phone_number
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
        language=voice_config_doc.language if voice_config_doc and voice_config_doc.language else "English",
        additional_languages=", ".join(config_doc.additionalLanguages) if config_doc and config_doc.additionalLanguages else [],
        time=get_time_in_timezone(config_doc.timezone),
        timezone=config_doc.timezone
    )
    # ---------- LLM ----------
    llm = config_doc.llm if config_doc and config_doc.llm else {}
    llm_provider = llm.get("provider", "groq")
    llm_model = llm.get("model", "qwen/qwen3-32b")
    llm_max_tokens = llm.get("max_tokens", 1000)

    # ---------- TTS ----------
    tts = voice_config_doc.tts if voice_config_doc and voice_config_doc.tts else {}
    tts_provider = tts.get("provider")
    tts_model = tts.get("model")
    tts_voice_id = tts.get("voice_id")
    tts_speed = tts.get("speed", 0.5)
    tts_volume = tts.get("volume", 2.0)
    tts_emotion = tts.get("emotion", "Happy")
    tts_language = tts.get("language", "en")

    # ---------- STT ----------
    stt = voice_config_doc.stt if voice_config_doc and voice_config_doc.stt else {}
    stt_provider = stt.get("provider", "deepgram")
    stt_model = stt.get("model", "flux-general-en")
    stt_language = stt.get("language", "en")

    # ---------- TOOLS ----------
    tools = config_doc.tools if config_doc and config_doc.tools else []

    # ---------- GREETING ----------
    greeting = (
        config_doc.welcomeMessage
        if config_doc and config_doc.welcomeMessage
        else "Hello! How can I assist you today?"
    )

    return agents.AgentConfig(
        user_id=str(agent.userId),
        agent_id=str(agent.id),
        admin_id=str(agent.adminId),
        agent_name=agent.agentName,
        knowledge_base_id=identity_doc.resourceCentreId,
        system_prompt=lk_prompt,
        workflow_graph_json=workflow_doc.workflow_config if workflow_doc else None,
        models=agents.ModelConfig(
            stt=agents.STTConfig(
                provider=stt_provider,
                model=stt_model,
                stt=stt_language
            ),
            llm=agents.LLMConfig(
                model=llm_model,
                provider=llm_provider,
                max_tokens=llm_max_tokens
            ),
            tts=agents.TTSConfig(
                model=tts_model,
                provider=tts_provider,
                voice_id=tts_voice_id,
                speed=tts_speed,
                volume=tts_volume,
                emotion=tts_emotion,
                language=tts_language,
            ),
        ),
        tools=tools,
        greeting=greeting,
        allow_recording=advanced_doc.privacy.get("audioRecording", True),
        max_duration=advanced_doc.inboundTimeout.get("maxDuration", -1) if advanced_doc else None,
        human_phone_number=human_phone_number,
        outbound_trunk_id=agent.outboundTrunkId if agent.outboundTrunkId else ""
    )


from zoneinfo import ZoneInfo
from datetime import datetime

def get_time_in_timezone(tz_name: str):
    return datetime.now(ZoneInfo(tz_name))
