from app.agents import AgentConfig, LeadContext
from app.agents.prompt import inbound

CALL_TYPE_MAP = {
    "test-inbound": "inbound",
    "test-outbound": "outbound",
    "campaign-outbound": "campaign_outbound",
    "test-campaign-outbound": "campaign_outbound",
    "test-campaign-callback": "callback_outbound",
    "campaign-callback-outbound": "callback_outbound",
    "callback-inbound": "callback_inbound",
}

CALL_BEHAVIOR = {
    "inbound": {
        "mode": "reactive",
        "has_previous_context": False,
        "opener": "neutral",
    },
    "outbound": {
        "mode": "proactive",
        "has_previous_context": False,
        "opener": "intro",
    },
    "campaign_outbound": {
        "mode": "scripted",
        "has_previous_context": False,
        "opener": "campaign",
    },
    "callback_outbound": {
        "mode": "contextual_proactive",
        "has_previous_context": True,
        "opener": "callback",
    },
    "callback_inbound": {
        "mode": "contextual_reactive",
        "has_previous_context": True,
        "opener": "callback",
    },
}

class PromptBuilder:
    def __init__(
        self, 
        config: AgentConfig,
        lead: LeadContext,
        call_type: str, 
    ):
        self.config = config
        self.lead_data = lead.data
        self.summary = lead.summary
        self.call_type = call_type
        # self.behavior = CALL_BEHAVIOR[call_type]
        self.session_context = lead.summary or {}

    def build(self):
        return "\n\n".join([
            self._system_prompt(),
            self._admin_goal(),
            self._user_context(),
            self._call_context(),
            self._callback_context(),
            # self._tool_context(),
        ])
    
    def _system_prompt(self):
        return inbound.lk_base_prompt.format(
            agent_name=self.config.agent_name,
            language=self.config.models.stt.language if self.config.models and self.config.models.stt and self.config.models.stt.language else "en",
            additional_languages=", ".join(self.config.additional_languages) if self.config and self.config.additional_languages else [],
            time=self.lead_data.user_current_time,
            timezone=self.config.call_details.timezone if self.config.call_details and self.config.call_details.timezone else "UTC",
        )

    # lk_base_prompt = inbound.lk_base_prompt.format(
    #     agent_name=agent.agentName,
    #     admin_goal=system_prompt,
    #     language=voice_config_doc.language if voice_config_doc and voice_config_doc.language else "English",
    #     additional_languages=", ".join(config_doc.additionalLanguages) if config_doc and config_doc.additionalLanguages else [],
    #     time=get_time_in_timezone(config_doc.timezone),
    #     timezone=config_doc.timezone
    # )  
    def _admin_goal(self):
        return f"""ADMIN GOAL: \n{self.config.system_prompt}"""

    def _user_context(self):
        u = self.lead_data

        return f"""
            User Details:
            Name: {u.name}
            Email: {u.email}
            Phone: {u.phone}
            Current Time: {u.user_current_time}
            Timezone: {u.user_timezone}
            """
    
    def _capability_rules(self):
        return """
    Keep responses short and conversational.
    Do not use bullet points.
    Speak naturally like a human on a phone call.
    """
        
    def _call_context(self):
        call_type = CALL_TYPE_MAP.get(self.config.call_type)

        if call_type == "inbound":
            return "You are handling an inbound call. Let the user lead."

        if call_type == "campaign_outbound":
            return "You are making a campaign call. Lead the conversation."

        if call_type == "callback_outbound":
            return "This is a scheduled callback to the user."

        if call_type == "callback_inbound":
            return "The user is calling back after a previous interaction."

        return ""
    
    def _callback_outbound_context(self):
        last_call = self.session_context.last_call_summary if self.session_context.last_call_summary else ""

        return f"""
        This is a scheduled callback to the user.

        Context from previous interaction:
        {last_call}

        You MUST:
        - Acknowledge this is a follow-up
        - Continue naturally from where the last conversation ended
        - Do NOT repeat the full introduction
        """
    
    def _callback_context(self):
        if "callback" not in self.config.call_type:
            return ""

        last_call = self.session_context.last_call_summary if self.session_context.last_call_summary else ""

        if not last_call:
            return ""

        return f"""
        Previous Interaction Summary:
        {last_call}

        You MUST:
        - Continue from previous conversation
        - Do NOT restart from scratch
        """

    def _conversation_context(self):
        lead_name = self.session_context.get("lead_name")

        if not lead_name:
            return ""

        return f"The user's name is {lead_name}. Use it naturally."
        
    def _tool_context(self):
        if "schedule_callback" in self.config.tools:
            return "You may schedule callbacks if needed."

        return ""