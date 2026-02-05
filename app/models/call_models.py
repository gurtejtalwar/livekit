import time
from bson import ObjectId
from datetime import datetime

from mongoengine import (
    Document,
    StringField,
    IntField,
    BooleanField,
    ObjectIdField,
    DateTimeField,
    EmbeddedDocument,
    EmbeddedDocumentField,  
    EmbeddedDocumentListField,
    ListField,
    DictField,
)
from livekit.agents import AgentSession

from app.services.agent import AgentConfig
from app.models import db

class Calls(Document):
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

    # status
    status = StringField(choices=("completed", "in_progress", "failed"))
    call_successful = StringField(choices=("success", "failure", "unknown"))

    transcript_summary = StringField()
    call_summary_title = StringField()

    direction = StringField()
    rating = IntField()

    agent_phone = StringField()
    customer_phone = StringField()

    escalation_transfer = StringField(choices=("yes", "no"), default="no")
    needs_phone_fetch = BooleanField(default=False)

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

class CallDetails(Document):
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

    analysis = DictField()

    conversation_initiation_client_data = DictField()

    has_audio = BooleanField()
    has_user_audio = BooleanField()
    has_response_audio = BooleanField()

    voice_summary = StringField()

    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)


def on_call_arrived(config: AgentConfig, session: AgentSession) -> ObjectId:
    """
    Called when a call starts.
    Extracts minimal data from config + session.
    """

    call = Calls(
        user_id=config.user_id,
        agent_id=config.agent_id,
        agent_name=config.agent_name,
        # branch_id=getattr(config, "branch_id", None),
        # version_id=getattr(config, "version_id", None),
        start_time_unix_secs=session._started_at,
        status="in_progress",
    ).save()

    CallDetails(
        call_id=str(call.id),                 # internal linkage
        user_id=config.user_id,
        agent_id=config.agent_id,
        agent_name=config.agent_name,
        # branch_id=getattr(config, "branch_id", None),
        # version_id=getattr(config, "version_id", None),
        status="in_progress",
    ).save()

    return call.id

def on_call_ended(
    call_id: ObjectId,
    config: AgentConfig,
    session: AgentSession,
):
    """
    Called when call completes.
    Uses Mongo Calls._id as primary key.
    """

    transcript_raw = ""#TODO session.transcript or []
    metadata_raw = {}#TODO session.metadata or {}
    analysis_raw = ""#TODO session.analysis or {}

    # -------------------------
    # Update Calls (summary)
    # -------------------------
    Calls.objects(
        id=call_id,
        status="in_progress",  # protects against double-finalization
    ).update_one(
        set__status="completed",
        set__call_duration_secs=time.time() - session._started_at,
        set__message_count=len(session.history.items),
        set__call_successful="success",
        set__transcript_summary="", #TODO analysis_raw.get("transcript_summary"),
        set__call_summary_title="", #TODO analysis_raw.get("call_summary_title"),
        set__updated_at=datetime.utcnow(),
    )   

    # -------------------------
    # Build transcript objects
    # -------------------------
    transcript_docs = build_transcript_messages(session.history.items)

    # -------------------------
    # Update CallDetails
    # -------------------------
    CallDetails.objects(
        call_id=str(call_id)
    ).update_one(
        set__status="completed",
        set__transcript=transcript_docs,
        set__metadata=ConversationMetadata(**metadata_raw)
        if metadata_raw
        else None,
        set__updated_at=datetime.utcnow(),
    )

def normalize_chat_content(content) -> str | None: #TODO Move to utils
    if not content:
        return None
    if isinstance(content, list):
        return " ".join(str(c) for c in content if c)
    return str(content)

def build_transcript_messages(raw_events: list) -> list[TranscriptMessage]:
    """
    Converts raw session transcript events into MongoEngine TranscriptMessage objects.
    Skips non-chat events (AgentHandoff, etc.)
    """

    transcript: list[TranscriptMessage] = []

    for event in raw_events:
        # -------------------------
        # Filter only ChatMessage
        # -------------------------
        if getattr(event, "type", None) != "message":
            continue

        # -------------------------
        # Normalize fields
        # -------------------------
        message_text = normalize_chat_content(event.content)

        transcript.append(
            TranscriptMessage(
                role="agent" if event.role == "assistant" else event.role,
                message=message_text,
                original_message=message_text,
                interrupted=event.interrupted,
                time_in_call_secs=(
                    int(event.created_at - raw_events[0].created_at)
                    if event.created_at and raw_events
                    else None
                ),
                agent_metadata=AgentMetadata(
                    agent_id=None,              # optional / fill if available
                    branch_id=None,
                    workflow_node_id=None,
                )
                if event.role == "assistant"
                else None,
                tool_calls=[],
                tool_results=[],
                feedback={
                    "transcript_confidence": event.transcript_confidence
                }
                if event.transcript_confidence is not None
                else None,
                llm_override=None,
                conversation_turn_metrics=event.metrics or {},
                rag_retrieval_info=None,
                llm_usage=None,
                source_medium=None,
            )
        )

    return transcript
