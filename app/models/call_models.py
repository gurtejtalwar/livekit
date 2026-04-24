import time
import json
import logging
from bson import ObjectId
from datetime import datetime

from mongoengine import (
    DoesNotExist,
    Document,
    StringField,
    IntField,
    FloatField,
    BooleanField,
    ObjectIdField,
    DateTimeField,
    EmbeddedDocument,
    EmbeddedDocumentField,  
    EmbeddedDocumentListField,
    ObjectIdField,
    ListField,
    DictField,
    EmailField
)
from livekit.agents import llm, AgentSession
from livekit.agents.metrics import UsageSummary

from app.shared import schemas
from app.agents import AgentConfig, CallDetails, LeadSummary
from app.models import db


logger = logging.getLogger(__name__)

class UsageSummaryEmbedded(EmbeddedDocument):
    llm_prompt_tokens = IntField(default=0)
    llm_prompt_cached_tokens = IntField(default=0)

    llm_input_audio_tokens = IntField(default=0)
    llm_input_cached_audio_tokens = IntField(default=0)

    llm_input_text_tokens = IntField(default=0)
    llm_input_cached_text_tokens = IntField(default=0)

    llm_input_image_tokens = IntField(default=0)
    llm_input_cached_image_tokens = IntField(default=0)

    llm_completion_tokens = IntField(default=0)

    llm_output_audio_tokens = IntField(default=0)
    llm_output_image_tokens = IntField(default=0)
    llm_output_text_tokens = IntField(default=0)

    tts_characters_count = IntField(default=0)
    tts_audio_duration = FloatField(default=0.0)
    stt_audio_duration = FloatField(default=0.0)


class VoiceCalls(Document):
    meta = {
        "collection": "voice-calls",
        "indexes": [
            "user_id",
            ("user_id", "-start_time_unix_secs"),
            ("agent_id", "-start_time_unix_secs"),
        ],
    }

    # ownership
    user_id = ObjectIdField(required=True)

    call_type = StringField()

    # identifiers
    agent_id = ObjectIdField()
    agent_name = StringField()

    branch_id = StringField(null=True) #FLAG
    version_id = StringField(null=True) #FLAG

    # timing
    start_time_unix_secs = IntField()
    call_duration_secs = IntField()
    message_count = IntField()
    tools_count = IntField()

    # status
    status = StringField(choices=("completed", "in_progress", "failed"))
    call_successful = StringField(choices=("success", "failure", "unknown"))

    transcript_summary = StringField()
    call_summary_title = StringField()

    direction = StringField() #TODO ASK
    rating = IntField()

    agent_phone = StringField()
    customer_phone = StringField()

    escalation_transfer = StringField(choices=("yes", "no"), default="no")
    needs_phone_fetch = BooleanField(default=False) #TODO ASK

    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)

class AgentMetadata(EmbeddedDocument):
    agent_id = StringField()
    branch_id = StringField()
    workflow_node_id = StringField()


class TranscriptMessage(EmbeddedDocument):
    role = StringField(choices=("agent", "user", "system"), required=True)

    message = StringField()
    original_message = StringField()

    interrupted = BooleanField()
    time_in_call_secs = IntField()

    agent_metadata = EmbeddedDocumentField(AgentMetadata)

    tool_calls = ListField(DictField())
    tool_results = ListField(DictField())

    feedback = DictField()
    llm_override = DictField()

    conversation_turn_metrics = DictField()
    rag_retrieval_info = DictField()
    llm_usage = DictField()

    source_medium = StringField()
    performance = DictField()
    created_at = IntField()

class ConversationMetadata(EmbeddedDocument):
    start_time_unix_secs = IntField()
    accepted_time_unix_secs = IntField()
    call_duration_secs = IntField()

    cost = IntField()

    termination_reason = StringField()
    error = DictField()
    warnings = ListField(StringField())

    main_language = StringField()
    text_only = BooleanField()

    authorization_method = StringField()
    agent_created_from = StringField()
    agent_last_updated_from = StringField()

    deletion_settings = DictField()
    feedback = DictField()
    charging = DictField()

    phone_call = DictField()
    batch_call = DictField()

    features_usage = DictField()
    rag_usage = DictField()

    timezone = StringField()

class SipMetadata(EmbeddedDocument):
    participant_sid = StringField()
    trunk_id = StringField()
    dispatch_rule = StringField()
    call_to = StringField()
    call_from = StringField()
    local_participant_sid = StringField()
    remote_participant_sid = StringField()

class STTModel(EmbeddedDocument):
    model_provider = StringField()
    model_name = StringField()
class LLMModel(EmbeddedDocument):
    model_provider = StringField()
    model_name = StringField()
class TTSModel(EmbeddedDocument):
    model_provider = StringField()
    model_name = StringField()
    voice_id = StringField()

class Models(EmbeddedDocument):
    stt = EmbeddedDocumentField(STTModel)
    llm = EmbeddedDocumentField(LLMModel)
    tts = EmbeddedDocumentField(TTSModel)
class LivekitMetadata(EmbeddedDocument):
    sip = DictField()
    usage = EmbeddedDocumentField(UsageSummaryEmbedded)
    models  = EmbeddedDocumentField(Models)

class Analysis(EmbeddedDocument):
    intent = StringField()
    emotion = StringField()
    score = FloatField()
    objection_analysis = StringField()
    summary = StringField()
    follow_up_trigger = BooleanField()
    lead_type = StringField()
    title = StringField()

class VoiceCallDetails(Document):
    meta = {
        "collection": "voice-calls-detail",
        "indexes": [
            "call_id",
            "user_id",
        ],
    }

    call_id = ObjectIdField(required=True, unique=True)
    user_id = ObjectIdField()

    agent_id = ObjectIdField()
    agent_name = StringField()
    room_name = StringField()
    call_type = StringField()

    branch_id = StringField()
    version_id = StringField()

    status = StringField()
    call_duration_secs = IntField()

    transcript = EmbeddedDocumentListField(TranscriptMessage)

    metadata = EmbeddedDocumentField(ConversationMetadata)
    lk_metadata = EmbeddedDocumentField(LivekitMetadata)
    analysis = EmbeddedDocumentField(Analysis)

    conversation_initiation_client_data = DictField()

    has_audio = BooleanField()
    has_user_audio = BooleanField()
    has_response_audio = BooleanField()
    recording_url = StringField()
    voice_summary = StringField()

    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)

class LeadsSummary(EmbeddedDocument):
    meta = {
        "collection": "leads_summary",
        "indexes": [
            "user_id",
        ],
    }

    user_id = ObjectIdField(required=True)
    agent_id = ObjectIdField()
    summary = StringField()

class VoiceCallLeads(Document):
    meta = {
        "collection": "voice-call-leads",
        "indexes": [
            "user_id",
        ],
    }

    user_id = ObjectIdField(required=True)
    admin_id = ObjectIdField()
    last_call_id = ObjectIdField()

    phone = StringField()  #TODO QUERY Need list?
    first_name = StringField()
    last_name = StringField()
    email = EmailField()

    agents_summary = EmbeddedDocumentListField(LeadsSummary)
    overall_summary = StringField()
    metadata = DictField()

async def on_call_ended(
    session: AgentSession,
    recording_url: str
):
    """
    Called when call completes.
    Uses Mongo VoiceCalls._id as primary key.
    """

    # -------------------------
    # Update VoiceCalls (summary)
    # -------------------------
    chat_messages = [
        item for item in session.history.items
        if isinstance(item, llm.ChatMessage)
    ]

    tool_calls = [
        item for item in session.history.items
        if isinstance(item, llm.FunctionCall)
    ]   

    VoiceCalls.objects(
        id=session.userdata.call_id,
        status="in_progress",  # protects against double-finalization
    ).update_one(
        set__status="completed",
        set__call_duration_secs=time.time() - session._started_at,
        set__message_count=len(chat_messages),
        set__tools_count=len(tool_calls),
        set__call_successful="success",
        set__transcript_summary="", #TODO analysis_raw.get("transcript_summary"),
        set__call_summary_title="", #TODO analysis_raw.get("call_summary_title"),
        set__updated_at=datetime.utcnow(),
    )   

    # -------------------------
    # Build transcript objects
    # -------------------------
    transcript_docs = build_structured_transcript(session.history.items)

    # -------------------------
    # Update VoiceCallDetails
    # -------------------------
    VoiceCallDetails.objects(
        call_id=str(session.userdata.call_id)
    ).update_one(
        set__status="completed",
        set__transcript=transcript_docs,
        set__call_duration_secs=time.time() - session._started_at,
        set__recording_url= recording_url,
        set__updated_at=datetime.utcnow(),
    )

def normalize_chat_content(content) -> str | None: #TODO Move to utils
    if not content:
        return None
    if isinstance(content, list):
        return " ".join(str(c) for c in content if c)
    return str(content)

def build_structured_transcript(history_items: list) -> list[dict]:

    transcript = []
    function_call_map = {}

    current_agent_index = None  # 🔥 track last agent message

    for item in history_items:
        print(f"Item: {type(item)}/n: {item}")
        # ------------------------
        # ChatMessage
        # ------------------------
        if isinstance(item, llm.ChatMessage):
            role = "agent" if item.role == "assistant" else "user"

            if isinstance(item.content, list):
                message = " ".join(p.strip() for p in item.content if p)
            else:
                message = str(item.content).strip()

            m = item.metrics if item.metrics else {}
            performance_data = {k: v for k, v in m.items()}

            transcript.append({
                "role": role,
                "message": message,
                "created_at": item.created_at,
                "performance": performance_data,
                "tool_calls": []
            })

            if role == "agent":
                current_agent_index = len(transcript) - 1  # ✅ track position

            continue

        # ------------------------
        # FunctionCall
        # ------------------------
        if isinstance(item, llm.FunctionCall):
            print(f"Function Call item: \n{item}")
            try:
                args = json.loads(item.arguments) if item.arguments else {}
            except Exception:
                args = item.arguments

            call_obj = {
                "name": item.name,
                "arguments": args,
                "output": None,
                "is_error": False,
            }

            function_call_map[item.call_id] = call_obj

            # ✅ Attach immediately to correct agent message
            if current_agent_index is not None:
                transcript[current_agent_index]["tool_calls"].append(call_obj)

            continue

        # ------------------------
        # FunctionCallOutput
        # ------------------------
        if isinstance(item, llm.FunctionCallOutput):
            print(f"Function Call Result Item: \n{item}")
            if item.call_id in function_call_map:
                try:
                    parsed_output = json.loads(item.output)
                except Exception:
                    parsed_output = item.output

                function_call_map[item.call_id]["output"] = parsed_output
                function_call_map[item.call_id]["is_error"] = item.is_error

            continue

    return transcript

def build_transcript_string(history_items: list) -> str:
    """
    Builds a readable transcript string from session history.
    Includes only ChatMessage items (skips AgentHandoff, tools, etc).
    """

    lines = []

    for item in history_items:
        if not isinstance(item, llm.ChatMessage):
            continue  # skip AgentHandoff, system events, etc

        speaker = "Agent" if item.role == "assistant" else "User"

        # content is usually a list of strings
        if isinstance(item.content, list):
            text = " ".join(part.strip() for part in item.content if part)
        else:
            text = str(item.content).strip()

        if not text:
            continue

        lines.append(f"{speaker}: {text}")

    return "\n\n".join(lines)

async def save_usage_summary(call_id: str, summary: UsageSummary):
    try:
        call_details: VoiceCallDetails = VoiceCallDetails.objects.get(call_id=call_id)
    except DoesNotExist:
        logger.warning(f"Call Details not found for call_id {call_id} when saving usage summary")
        return
    try:
        call_details.lk_metadata.usage = UsageSummaryEmbedded(
            llm_prompt_tokens=summary.llm_prompt_tokens,
            llm_prompt_cached_tokens=summary.llm_prompt_cached_tokens,

            llm_input_audio_tokens=summary.llm_input_audio_tokens,
            llm_input_cached_audio_tokens=summary.llm_input_cached_audio_tokens,

            llm_input_text_tokens=summary.llm_input_text_tokens,
            llm_input_cached_text_tokens=summary.llm_input_cached_text_tokens,

            llm_input_image_tokens=summary.llm_input_image_tokens,
            llm_input_cached_image_tokens=summary.llm_input_cached_image_tokens,

            llm_completion_tokens=summary.llm_completion_tokens,

            llm_output_audio_tokens=summary.llm_output_audio_tokens,
            llm_output_image_tokens=summary.llm_output_image_tokens,
            llm_output_text_tokens=summary.llm_output_text_tokens,

            tts_characters_count=summary.tts_characters_count,
            tts_audio_duration=summary.tts_audio_duration,
            stt_audio_duration=summary.stt_audio_duration,
        )
        call_details.save()
    except Exception as e:
        logger.error("Error saving usage summary:\n %s", e)

async def save_analysis(call_id: str, analysis: schemas.PostCallAnalysis):
    try:
        call_details: VoiceCallDetails = VoiceCallDetails.objects.get(call_id=call_id)
    except DoesNotExist:
        logger.warning(f"CallDetails not found for call_id {call_id} when saving analysis")
        return
    analysis = Analysis(**analysis.model_dump())
    call_details.analysis = analysis
    call_details.save()


async def inbound_handler(config: AgentConfig, session: AgentSession):
    participant = next(iter(session._room_io._room._remote_participants.values()), None)
    sip_attrs = participant.attributes if participant else None
    call = VoiceCalls(
        user_id=config.user_id,
        agent_id=config.agent_id,
        agent_name=config.agent_name,
        call_type=config.call_type,
        # branch_id=getattr(config, "branch_id", None),
        # version_id=getattr(config, "version_id", None),
        start_time_unix_secs=session._started_at,
        status="in_progress",
        agent_phone=config.call_details.call_to,
        customer_phone=config.call_details.call_from,
    ).save()

    VoiceCallDetails(
        call_id=str(call.id),                 # internal linkage
        user_id=config.user_id,
        agent_id=config.agent_id,
        agent_name=config.agent_name,
        room_name=session.room_io.room.name if session.room_io and session.room_io.room else None,
        call_type=config.call_type,
        lk_metadata=LivekitMetadata(
            sip=sip_attrs,
        ),
        # branch_id=getattr(config, "branch_id", None),
        # version_id=getattr(config, "version_id", None),
        status="in_progress",
    ).save()

    session.userdata.call_id = str(call.id)
    return call.id

async def outbound_handler(config, session):
    pass

async def test_inbound_handler(config: AgentConfig, session: AgentSession):
    participant = next(iter(session._room_io._room._remote_participants.values()), None)
    sip_attrs = participant.attributes if participant else None
    call = VoiceCalls(
        user_id=config.user_id,
        agent_id=config.agent_id,
        agent_name=config.agent_name,
        call_type=config.call_type,
        start_time_unix_secs=session._started_at,
        status="in_progress",
    ).save()

    VoiceCallDetails(
        call_id=str(call.id),                 # internal linkage
        user_id=config.user_id,
        agent_id=config.agent_id,
        agent_name=config.agent_name,
        call_type=config.call_type,
        room_name=session.room_io.room.name if session.room_io and session.room_io.room else None,
        lk_metadata=LivekitMetadata(
            sip=sip_attrs
        ),
        status="in_progress",
    ).save()

    session.userdata.call_id = str(call.id)
    return call.id



async def upsert_voice_call_lead(
    *,
    user_id: str,
    admin_id: str,
    agent_id: str,
    phone: str | None = None,
    last_call_id: str | None = None,
    first_name: str | None = None,
    last_name: str | None = None,
    email: str | None = None,
    metadata: dict | None = None,
    agent_summary: str | None = None,
    overall_summary: str | None = None,
):
    """
    Upserts a voice call lead using phone as unique identifier.
    Updates only provided fields.
    """

    lead: VoiceCallLeads = VoiceCallLeads.objects(phone=phone).first()

    if not lead:
        lead:VoiceCallLeads = VoiceCallLeads(
            user_id=ObjectId(user_id),
            admin_id=ObjectId(admin_id),
            phone=phone,
            agents_summary=[]
        )

    # ---- Basic fields ----
    if first_name:
        lead.first_name = first_name

    if last_name:
        lead.last_name = last_name

    if email:
        lead.email = email

    if phone:
        lead.phone = phone

    if metadata:
        if lead.metadata:
            lead.metadata.update(metadata)  # merge with existing
        else:
            lead.metadata = metadata
    
    if last_call_id:
        lead.last_call_id = ObjectId(last_call_id)

    # ---- Agent-specific summary ----
    if agent_summary:
        existing = next(
            (s for s in lead.agents_summary if str(s.agent_id) == agent_id),
            None
        )

        if existing:
            existing.summary = agent_summary
        else:
            lead.agents_summary.append(
                LeadsSummary(
                    user_id=ObjectId(user_id),
                    agent_id=ObjectId(agent_id),
                    summary=agent_summary,
                )
            )

    # ---- Overall summary ----
    if overall_summary:
        lead.overall_summary = overall_summary

    lead.save()

    return {"status": "success"}

async def get_lead_by_phone(phone_number: str, admin_id: str) -> VoiceCallLeads | None:
    lead = VoiceCallLeads.objects(
        phone=phone_number,
        admin_id=ObjectId(admin_id)
    ).first()

    return lead

async def get_summary_by_phone_number(phone_number: str, admin_id: str, agent_id: str) -> LeadSummary:
    lead = await get_lead_by_phone(phone_number, admin_id)
    
    if not lead:
        return None

    last_call_details = await get_latest_completed_call_details(phone_number, agent_id)

    # Try to find agent-specific summary first
    agent_summary = next(
        (s.summary for s in lead.agents_summary if str(s.agent_id) == agent_id),
        None
    )

    # Fallback to overall summary
    return LeadSummary(
        agent_summary=agent_summary if agent_summary else None,
        overall_summary=lead.overall_summary if lead.overall_summary else None,
        last_call_summary=last_call_details.analysis.summary if last_call_details and last_call_details.analysis and last_call_details.analysis.summary else None
    )

async def get_latest_completed_call_details(phone_number: str, agent_id: str) -> VoiceCallDetails | None:
    last_call = VoiceCalls.objects(
        customer_phone=phone_number,
        agent_id=ObjectId(agent_id),
        status="completed"
    ).order_by("-start_time_unix_secs").first()

    if not last_call:
        return None

    call_details = VoiceCallDetails.objects(call_id=str(last_call.id)).first()
    return call_details