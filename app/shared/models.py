from mongoengine import (
    Document, EmbeddedDocument,
    StringField, BooleanField, DateTimeField,
    ListField, DictField, ReferenceField,
    FloatField, IntField, ObjectIdField
)
from bson import ObjectId


class VoiceAgent(Document):
    meta = {
        "collection": "voice-agent",
        "indexes": ["userId", "providerType", "type"],
        "strict": False  # Allow mongoose extra fields

    }

    userId = ObjectIdField(required=True)
    adminId = ObjectIdField(required=True)
    agentName = StringField(required=True)

    type = StringField(default="voice")
    providerType = StringField(required=True, default="elevenlabs")
    status = StringField(default="deploying")
    isActive = BooleanField(default=True)

    knowledgeBaseId = StringField()
    resourceCentreName = StringField()

    assignedPhoneNumberId = StringField()
    assignedPhoneNumber = StringField()

    mainGoal = StringField()
    role = StringField()

    livekitAgentId = StringField()
    livekitProjectSubdomain = StringField()


    inboundTrunkId = StringField()
    outboundTrunkId = StringField()
    inboundDispatchRuleId = StringField()

    identity = DictField(default=dict)
    agentConfig = DictField(default=dict)
    voiceConfig = DictField(default=dict)
    escalation = DictField(default=dict)
    config = DictField(default=dict)
    advancedSettings = DictField(default=dict)
    deployment = DictField(default=dict)
    metrics = DictField(default=dict)

    lastError = StringField()

    lastUpdatedTypes = ListField(StringField(), default=list)

class VoiceAgentConfig(Document):
    meta = {
        "collection": "voice-agent-config",
        "indexes": [
            "userId",
            "agentId",
            {"fields": ["userId", "agentId"], "unique": True}
        ],
        "strict": False  # Allow mongoose extra fields

    }

    userId = ObjectIdField(required=True)
    agentId = ObjectIdField(required=True)

    purposes = ListField(StringField(), default=list)
    welcomeMessage = StringField()
    welcomeMessageLocked = BooleanField(default=False)
    isWelcomeMessageEdited = BooleanField()
    gptCustomizationEnabled = BooleanField()
    customErrorMessageEnabled = BooleanField()
    customErrorMessage = StringField()
    systemPrompt = StringField()
    systemPromptConfidenceScore = StringField(choices=["low", "medium", "high"])
    dataToCollect = DictField()
    purposeConfigs = DictField()
    additionalLanguages = ListField(StringField(), default=list)
    llm = DictField()
    tools = ListField(StringField(), default=list)
    timezone = StringField(default='Asia/Calcutta')
    isWorkflowEnabled = BooleanField(default=False)
    workflowS3Url = StringField()
class VoiceAgentIdentity(Document):
    meta = {
        "collection": "voice-identity",
        "indexes": ["userId", "agentId"],
        "strict": False  # Allow mongoose extra fields
    }

    userId = ObjectIdField(required=True)
    agentId = ObjectIdField(required=True)

    agentName = StringField()
    greetingName = StringField()

    businessName = StringField()
    businessWebsite = StringField()
    businessAddress = StringField()
    businessEmail = StringField()

    businessPhone = DictField()

    industry = StringField()
    useCase = StringField()
    callType = ListField(StringField())

    resourceCentreId = StringField()
    resourceCentreName = StringField()

    assignedPhoneNumberId = StringField()
    assignedPhoneNumber = StringField()

    mainGoal = StringField()
    roleDescription = StringField()
    agentDescription = StringField()
    shareBusinessDetails = BooleanField()

class VoiceAgentVoiceConfig(Document):
    meta = {
        "collection": "voice-config",
        "indexes": [
            "userId",
            "agentId",
            {"fields": ["userId", "agentId"], "unique": True}
        ],
        "strict": False  # Allow mongoose extra fields
    }

    userId = ObjectIdField(required=True)
    agentId = ObjectIdField(required=True)

    # ── Voice / personality fields (original) ──
    language = StringField()
    welcomeMessage = StringField()
    isWelcomeMessageEdited = BooleanField()
    systemPrompt = StringField()
    autoSwitchLanguage = BooleanField()
    tone = StringField()  # enum VoiceTone
    speakingSpeed = FloatField()
    stability = FloatField()
    similarity = FloatField()
    emotionAwareResponse = BooleanField()
    callerMemory = BooleanField()
    additionalLanguages = ListField(StringField(), default=list)
    audioFormat = StringField(choices=['telephony', 'widget'], default='widget')

    tts = DictField()
    stt = DictField()

    # ── LiveKit-only fields (merged from voice-agent-config) ──
    # Stored here for LiveKit agents so we only need one collection

    purposes = ListField(StringField(), default=list)
    gptCustomizationEnabled = BooleanField()
    customErrorMessageEnabled = BooleanField()
    customErrorMessage = StringField()
    systemPromptConfidenceScore = StringField()
    dataToCollect = DictField()
    purposeConfigs = DictField()
    llm = DictField()
    tools = ListField(StringField(), default=list)
    timezone = StringField(default='Asia/Calcutta')

    # LiveKit workflow toggle
    isWorkflowEnabled = BooleanField(default=False)



class VoiceAgentEscalation(Document):
    meta = {
        "collection": "voice-escalation",
        "indexes": ["userId", "agentId"],
        "strict": False  # Allow mongoose extra fields
    }

    userId = ObjectIdField(required=True)
    agentId = ObjectIdField(required=True)

    humanEscalationEnabled = BooleanField(default=False)
    escalationRules = DictField()
    escalationBehavior = DictField()

    teamId = StringField()
    teamName = StringField()
    roleType = StringField()
    availability = StringField()
    escalationPriority = StringField()

    teamMembers = ListField(DictField())
    escalationPrompt = StringField()

class VoiceAgentWorkflow(Document):
    meta = {
        "collection": "voice-agent-workflow",
        "indexes": ["userId", "agentId"],
        "strict": False  # Allow mongoose extra fields
    }

    userId = ObjectIdField(required=True)
    agentId = ObjectIdField(required=True)

    mode = StringField()
    workflow_config = DictField()

class VoiceAgentAdvancedSettings(Document):
    meta = {
        "collection": "voice-advanced-settings",
        "indexes": ["userId", "agentId"],
        "strict": False  # Allow mongoose extra fields
    }

    userId = ObjectIdField(required=True)
    agentId = ObjectIdField(required=True)

    privacy = DictField()
    startSpeakingPlan = DictField()
    stopSpeakingPlan = DictField()
    voicemailDetection = DictField()
    webhook = DictField()
    inboundTimeout = DictField()
    outboundTimeout = DictField()
    systemTools = DictField()

    conversationalBehavior = DictField()
    softTimeout = DictField()

    smartAnalysis = BooleanField()
    enableBursting = DictField()
    callBack = DictField()

class VoiceBotSettings(Document):
    meta = {
        "collection": "voicebotsettings",
        "indexes": [
            {
                "fields": ["userId"],
                "unique": True
            }
        ],
        "strict": False
    }

    userId = ObjectIdField(required=True)
    agentId = ObjectIdField(required=True)

    phone_number = StringField()
    startTime = StringField()
    endTime = StringField()
    timezone = StringField()

    handleIncomingCall = BooleanField()