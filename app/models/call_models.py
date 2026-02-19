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
    ListField,
    DictField,
)
from livekit.agents import llm, AgentSession
from livekit.agents.metrics import UsageSummary

from app.shared import schemas
from app.agent import AgentConfig
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

    # identifiers
    agent_id = StringField()
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

class LivekitMetadata(EmbeddedDocument):
    sip = EmbeddedDocumentField(SipMetadata)
    usage = EmbeddedDocumentField(UsageSummaryEmbedded)

class Analysis(EmbeddedDocument):
    intent = StringField()
    emotion = StringField()
    score = FloatField()
    objection_analysis = StringField()
    summary = StringField()
    follow_up_trigger = BooleanField()
    lead_type = StringField()

class VoiceCallDetails(Document):
    meta = {
        "collection": "voice-calls-detail",
        "indexes": [
            "call_id",
            "user_id",
        ],
    }

    call_id = StringField(required=True, unique=True)
    user_id = ObjectIdField()

    agent_id = StringField()
    agent_name = StringField()

    branch_id = StringField()
    version_id = StringField()

    status = StringField()

    transcript = EmbeddedDocumentListField(TranscriptMessage)

    metadata = EmbeddedDocumentField(ConversationMetadata)
    lk_metadata = EmbeddedDocumentField(LivekitMetadata)
    analysis = EmbeddedDocumentField(Analysis)

    conversation_initiation_client_data = DictField()

    has_audio = BooleanField()
    has_user_audio = BooleanField()
    has_response_audio = BooleanField()

    voice_summary = StringField()

    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)


async def on_call_arrived(config: AgentConfig, session: AgentSession) -> ObjectId:
    """
    Called when a call starts.
    Extracts minimal data from config + session.
    """

    call = VoiceCalls(
        user_id=config.user_id,
        agent_id=config.agent_id,
        agent_name=config.agent_name,
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
        lk_metadata=LivekitMetadata(
            sip=SipMetadata(
                local_participant_sid=session.room_io.room.local_participant.sid,
                remote_participant_sid=list(session.room_io.room.remote_participants.values())[0].sid,
                trunk_id=config.call_details.trunk_id,
                dispatch_rule=config.call_details.dispatch_rule,
                call_to=config.call_details.call_to,
                call_from=config.call_details.call_from,
            )
        ),
        # branch_id=getattr(config, "branch_id", None),
        # version_id=getattr(config, "version_id", None),
        status="in_progress",
    ).save()

    session.userdata.call_id = str(call.id)
    return call.id

async def on_call_ended(
    config: AgentConfig,
    session: AgentSession,
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
        # set__metadata=ConversationMetadata(**metadata_raw)
        # if metadata_raw
        # else None,
        set__updated_at=datetime.utcnow(),
    )

def normalize_chat_content(content) -> str | None: #TODO Move to utils
    if not content:
        return None
    if isinstance(content, list):
        return " ".join(str(c) for c in content if c)
    return str(content)

def build_structured_transcript(history_items: list) -> list[dict]:
    """
    Builds structured transcript with:
    - role
    - message
    - function_calls (if any)
    """

    transcript = []

    # Temporary store of function calls by call_id
    function_call_map = {}

    for item in history_items:

        # ------------------------
        # Capture FunctionCall
        # ------------------------
        if isinstance(item, llm.FunctionCall):
            try:
                args = json.loads(item.arguments) if item.arguments else {}
            except Exception:
                args = item.arguments

            function_call_map[item.call_id] = {
                "name": item.name,
                "arguments": args,
                "output": None,
                "is_error": False,
            }
            continue

        # ------------------------
        # Capture FunctionCallOutput
        # ------------------------
        if isinstance(item, llm.FunctionCallOutput):
            if item.call_id in function_call_map:
                try:
                    parsed_output = json.loads(item.output)
                except Exception:
                    parsed_output = item.output

                function_call_map[item.call_id]["output"] = parsed_output
                function_call_map[item.call_id]["is_error"] = item.is_error
            continue

        # ------------------------
        # Capture ChatMessage
        # ------------------------
        if isinstance(item, llm.ChatMessage):
            role = "agent" if item.role == "assistant" else "user"

            # Normalize content
            if isinstance(item.content, list):
                message = " ".join(p.strip() for p in item.content if p)
            else:
                message = str(item.content).strip()

            transcript.append({
                "role": role,
                "message": message,
                "tool_calls": []
            })

    # ------------------------
    # Attach function calls to last agent message
    # (Most frameworks call functions immediately after assistant turn)
    # ------------------------
    for call in function_call_map.values():
        # attach to most recent agent message
        for entry in reversed(transcript):
            if entry["role"] == "agent":
                print("Entry:\n", entry)
                entry["tool_calls"].append(call)
                break

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
        logger.warning(f"CallDetails not found for call_id {call_id} when saving usage summary")
        return

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

async def save_analysis(call_id: str, analysis: schemas.PostCallAnalysis):
    try:
        call_details: VoiceCallDetails = VoiceCallDetails.objects.get(call_id=call_id)
    except DoesNotExist:
        logger.warning(f"CallDetails not found for call_id {call_id} when saving analysis")
        return
    analysis = Analysis(**analysis.model_dump())
    call_details.analysis = analysis
    call_details.save()

