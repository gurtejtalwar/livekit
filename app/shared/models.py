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

class VoiceAgentConfigLivekit(Document):
    meta = {
        "collection": "voice-agent-config-livekit",
        "indexes": ["userId", "agentId"],
        "strict": False  # Allow mongoose extra fields

    }

    userId = ObjectIdField(required=True)
    agentId = ObjectIdField(required=True)

    voiceType = StringField()
    language = StringField()
    autoSwitchLanguage = BooleanField()

    tone = StringField()
    speakingSpeed = FloatField()

    stability = FloatField()
    similarity = FloatField()

    emotionAwareResponse = BooleanField()
    callerMemory = BooleanField()

    additionalLanguages = ListField(StringField())

    tts = DictField()
    stt = DictField()

    purposes = ListField(StringField())
    welcomeMessage = StringField()
    isWelcomeMessageEdited = BooleanField()
    gptCustomizationEnabled = BooleanField()
    customErrorMessageEnabled = BooleanField()
    customErrorMessage = StringField()

    systemPrompt = StringField()
    systemPromptConfidenceScore = StringField()

    dataToCollect = DictField()
    purposeConfigs = DictField()

    additionalLanguages = ListField(StringField())
    llm = DictField()
    tools = ListField(StringField())

    timezone = StringField()

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
        "indexes": ["userId", "agentId"],
        "strict": False  # Allow mongoose extra fields
    }

    userId = ObjectIdField(required=True)
    agentId = ObjectIdField(required=True)



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