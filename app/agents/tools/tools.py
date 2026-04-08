import os
import pickle
import faiss
import torch
import asyncio
import logging
import gc

import time
from datetime import datetime, timezone
from dotenv import load_dotenv
from typing import Optional, List, Literal
from pydantic import BaseModel
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

from livekit import api
from livekit.agents import llm, get_job_context, RunContext

from app.utils.timer import Timer
from app.shared.settings import get_settings
from app.utils.requests import _request, APIClient
from app.agents.workflows import WarmTransferTask

load_dotenv(override=True)
settings = get_settings()
logger = logging.getLogger("TOOLS")

from opentelemetry import trace
import json as json_lib

tracer = trace.get_tracer(__name__)

class ToolExecutor:
    @staticmethod
    async def run(tool_name: str, fn, *, args: dict):
        span = trace.get_current_span()

        span.update_name(tool_name)

        span.set_attribute("tool.name", tool_name)
        span.set_attribute("tool.args", json_lib.dumps(args))

        try:
            result = await fn()

            span.set_attribute(
                "tool.response",
                json_lib.dumps(result)[:2000]
            )
            span.set_attribute("tool.status", "success")

            return result

        except Exception as e:
            span.set_attribute("tool.status", "error")
            span.set_attribute("tool.error", str(e))
            raise
def tool(name: str):
    def decorator(fn):
        @llm.function_tool
        async def wrapper(*args, **kwargs):
            return await ToolExecutor.run(
                name,
                lambda: fn(*args, **kwargs),
                args=kwargs
            )
        return wrapper
    return decorator

n3_api = APIClient(settings.N3_ISC_URL, settings.N3_ISC_API_KEY)
n1_api = APIClient(settings.N1_ISC_URL, settings.N1_ISC_API_KEY)

class KnowledgeBase:
    def __init__(self, index, chunks):
        self.index = index
        self.chunks = chunks
        self.last_used = time.time()

    def search(self, query_emb, k=3):
        dist, idx = self.index.search(query_emb, k)
        indices = idx[0]
        return [
            self.chunks[i] if 0 <= i < len(self.chunks) else "[INVALID INDEX]"
            for i in indices
        ]

class KBManager:
    def __init__(self):
        self._kbs: dict[str, KnowledgeBase] = {}
        self._lock = asyncio.Lock()

async def get_kb(self, agent_id: str) -> KnowledgeBase:
    async with self._lock:
        kb = self._kbs.get(agent_id)
        if kb:
            kb.last_used = time.time()
            return kb

        index = faiss.read_index(f"kb/{agent_id}/faiss.index")
        with open(f"kb/{agent_id}/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

        kb = KnowledgeBase(index, chunks)
        self._kbs[agent_id] = kb
        return kb

async def unload_kb(self, agent_id: str):
    async with self._lock:
        kb = self._kbs.pop(agent_id, None)
        if not kb:
            return

        del kb.index
        del kb.chunks
        del kb

    gc.collect()

KB_CACHE = {} #TODO HAZARD use redis

#TODO Use resource centre id instead of agent id
def load_knowledge_base(resource_centre_id: str) -> KnowledgeBase: 
    if resource_centre_id in KB_CACHE:
        return KB_CACHE[resource_centre_id]

    with Timer(f"Load KB for {resource_centre_id}"):
        index = faiss.read_index(f"app/knowledge_base/{resource_centre_id}_faiss.index")
        with open(f"app/knowledge_base/{resource_centre_id}_chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

    kb = KnowledgeBase(index=index, chunks=chunks)
    KB_CACHE[resource_centre_id] = kb
    return kb

def make_ask_knowledge_base_tool(kb: KnowledgeBase):

    @tool("ask_knowledge_base")
    async def ask_knowledge_base(question: str):
        with Timer("KB Tool Total"):
            with Timer("Embed Query"):
                q_emb = embed(question)

            with Timer("FAISS Search"):
                results = kb.search(q_emb, k=3)

            return "\n".join(results)

    return ask_knowledge_base

#TODO Pre call tasks
def get_faiss_index_and_chunks():
    """Load FAISS index and text chunks from disk."""
    with open("dev_scripts/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    if kb not in KB_CACHE:
        index = faiss.read_index("dev_scripts/faiss.index")
        KB_CACHE[kb]=True
    return index, chunks

def embed(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

#TODO Deficit
KB_CACHE={}
MODEL_CACHE={}
kb = "test"
model = "test"
with Timer("Load Embedding Model"):

    if model not in MODEL_CACHE:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        model = ORTModelForFeatureExtraction.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        MODEL_CACHE[model]=True


@tool("get_current_time")
async def get_current_time(input: str) -> str:
    """Get the current time."""
    from datetime import datetime
    return f"The current time is {datetime.now().strftime('%I:%M %p')}" 

async def hangup_call(ctx: RunContext):
    # Ensure any pending agent speech is finished before killing the room
    await ctx.wait_for_playout()
    await api.room.delete_room(
        api.DeleteRoomRequest(room=ctx.room.name)
    )

@tool("end_call")
async def end_call(ctx: RunContext, reason: str = ""):
    """
    Gracefully terminates the ongoing voice call/session.

    This tool is invoked by the agent when it determines that the conversation
    has reached a natural conclusion or when a specific condition requires
    ending the call (e.g., user request, task completion, escalation, etc.).
    
    Args:
        reason (str, optional): A short description of why the call is being
            ended (e.g., "user requested", "task completed", "no response").
            Defaults to an empty string.

    Usage Guidelines (for LLM):
    - Only call this function when the conversation is clearly complete.
    - Do NOT call this function on partial acknowledgements (e.g., "thanks").
    - Prefer confirming with the user before ending the call unless explicitly requested.

    Returns:
        None
    """
    session = ctx.session

    # 🗣️ Step 1: speak
    speech = await session.say(
        "Alright, thanks for your time. Have a great day!"
    )
    logger.info(
        "TOOL: END CALL Message: %s",
        speech.chat_items[-1].content if speech.chat_items else "No chat items"
    )
    # 🧠 Step 2: wait OR get interrupted
    await speech.wait_if_not_interrupted([])

    # ❌ Step 3: user interrupted → abort ending
    if speech.interrupted:
        return

    # ✅ Step 4: safe to end
    job_ctx = get_job_context()
    if job_ctx:
        await session.aclose()
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(room=job_ctx.room.name)
        )

@tool("detected_voicemail")
async def detected_voicemail(ctx: RunContext, dummy: str=""):
    """Call this tool if you have detected a voicemail system, AFTER hearing the voicemail greeting"""
    await ctx.session.generate_reply(
        instructions="Leave a voicemail message letting the user know you'll call back later."
    )
    await asyncio.sleep(0.5) # Add a natural gap to the end of the voicemail message
    await hangup_call()


#/ -- Book Appointment Tool --/

class CustomField(BaseModel):
    key: str
    value: str

#/ -- Book Appointment Tool --/
@tool("book_appointment")
async def book_appointment(
    name: str,
    date: str,
    time: str,
    agentId: str,
    ctx: RunContext,
    customFields: Optional[List[CustomField]] = None,
):
    """
    Called only when the user confirms the date and time for booking a new appointment.
    Do not call this tool without confirming with the user first. 
    """
    payload = {
        "conversation_id": ctx.session.userdata.call_id,
        "caller_name": name,
        "date": date,
        "time": time,
        "agentId": ctx.session.userdata.agent_id,
        "customFields": customFields or {}
    }

    return await n3_api.post("/book-appointment", payload)

#/ -- Cancel Appointment Tool --/
@tool("cancel_appointment")
async def cancel_appointment(booking_id: str):
    """
    Cancel an existing appointment using booking ID.
    """
    headers = {
    "x-agent-secret": settings.N1_ISC_API_KEY,
    "Content-Type": "application/json"
    }

    return await n3_api.delete("/book-appointment", 
                               params={"booking_id": booking_id}
                               )
                               
#/ -- Get Available Slots Tool --/
@tool("get_available_slots")
async def get_available_slots(
    city: str,
    ctx: RunContext,
):
    """
    Get available appointment slots for a given agent and date.
    """
    params={
        "city": city,
        "contact_phone": ctx.session.userdata.phone,
        "timezoneName": "Asia/Calcutta", #TODO HAZARD
    }
    return await n1_api.get(f"/timeslot/get-voicebot-slots/{ctx.session.userdata.admin_id}", params=params)


async def get_available_slots_DEPRECATED(
    agentId: str,
    date: str,
    status: str = "available"
):
    """
    Get available appointment slots for a given agent and date.
    """
    headers = {
    "x-agent-secret": settings.N3_ISC_API_KEY,
    "Content-Type": "application/json"
    }
    result = await _request(
        "GET",
        f"{settings.N3_ISC_URL}/available-slot",
        headers=headers,
        params={
            # "agentId": agentId,
            "date": date,
            "status": status
        }
    )
    print("Available slots result:\n", result)
    return result

#/ -- Reschedule Appointment Tool --/
@tool("reschedule_appointment")
async def reschedule_appointment(
    booking_id: str,
    time: str,
    ctx: RunContext,
    # customFields: dict | None = None
):
    """
    Reschedule an existing appointment.
    """
    payload = {
        "booking_id": booking_id,
        "contact_phone": ctx.session.userdata.phone, #TODO QUERY - Feature for callback on different number? Misuse implications?,
        "time": time,
        "agentId": ctx.session.userdata.agent_id,
        # "customFields": customFields or {}
    }

    return await n3_api.patch("/book-appointment", payload)

#/ -- Create CRM Lead Tool --/
@tool("create_crm_lead")
async def create_crm_lead(
    first_name: str,
    email: str,
    phone: str,
    company: str,
    admin_id: str,
):
    """
    Create a lead in the external CRM system.
    Call this when a user wants to be contacted by sales or provides contact details.
    """
    headers = {
    "x-agent-secret": settings.N1_ISC_API_KEY,
    "Content-Type": "application/json"
    }
    payload = {
        "firstName": first_name,
        "email": email,
        "phone": phone,
        "company": company,
        "adminId": admin_id,
    }
    return n1_api.post("/api/crm/external/lead-create", payload)

@tool("customer_support")
async def customer_support(
    caller_name: str,
    contact_email: str,
    issue_category: str,
    issue_description: str,
    ctx: RunContext,
):
    """
    Create a customer support ticket.

    This tool must ONLY be used after:
    - The caller explicitly agrees to create a support ticket
    - All required information has been collected

    ----------------------------
    PARAMETER DEFINITIONS
    ----------------------------

    caller_name:
        Name of the caller creating the support ticket.
        Extract from the conversation if provided.
        If missing, ask the user before calling the tool.

    contact_email:
        Email address of the caller.
        Must be explicitly provided by the user.
        Do NOT infer or guess. Ask if missing.

    issue_category:
        Category of the issue. Must be one of:
        - "technical" → bugs, errors, API issues, integrations, audio/call problems
        - "general" → billing, account, subscription, general inquiries

        Determine this based on the user's problem description.

    issue_description:
        Clear and concise description of the issue.
        Use the caller’s own words where possible.
        If multiple issues are mentioned, summarize them into one.
        Do NOT ask again if already clearly described.

    ----------------------------
    EXECUTION RULES
    ----------------------------

    - Do NOT call this tool without explicit user consent
    - Ask for missing fields ONE AT A TIME
    - Do NOT ask for phone number
    - Do NOT re-ask for already provided information
    - As soon as all required fields are available, call the tool immediately
    - Do NOT continue conversation after all fields are collected

    ----------------------------
    FAILURE HANDLING
    ----------------------------

    - If the tool fails:
        - Do NOT retry automatically
        - Do NOT ask the user to repeat inputs
        - Apologize once
        - Inform the user the request cannot be completed right now
        - Offer a fallback (manual follow-up or retry later)
    """
    payload = {
        "caller_name": caller_name,
        "contact_email": contact_email,
        "contact_phone": ctx.session.userdata.phone, #TODO QUERY - Feature for callback on different number? Misuse implications?
        "conversation_id": ctx.session.userdata.call_id,
        "issue_category": issue_category,
        "issue_description": issue_description,
        "agentId": ctx.session.userdata.agent_id,
    }

    return await n3_api.post("/customer-ticket/create", payload)

@tool("sales_lead_generation")
async def sales_lead_generation(
    name: str,
    email: str,
    company: str,
    ctx: RunContext,
):
    """
    Create a sales lead from the conversation.

    This tool must ONLY be used after:
    - The caller explicitly agrees to be contacted or followed up
    - All required information has been collected

    ----------------------------
    PARAMETER DEFINITIONS
    ----------------------------

    name:
        Name of the caller/lead.
        Extract from the conversation if provided.
        If missing, ask the user before calling the tool.

    email:
        Email address of the caller.
        Must be explicitly provided by the user.
        Do NOT infer or guess.
        Confirm if unclear.

    company:
        Company name of the caller.
        Extract if mentioned.
        If not provided, ask the user.

    ----------------------------
    EXECUTION RULES
    ----------------------------

    - Do NOT call this tool without explicit user consent
    - Ask for missing fields ONE AT A TIME
    - Do NOT ask for phone number
    - Do NOT ask for adminId
    - Do NOT re-ask for already provided information
    - As soon as all required fields are available, call the tool immediately
    - Do NOT continue conversation after all fields are collected

    ----------------------------
    FAILURE HANDLING
    ----------------------------

    - If the tool fails:
        - Do NOT retry automatically
        - Do NOT ask the user to repeat inputs
        - Apologize once
        - Inform the user the request cannot be completed right now
        - Offer a fallback (manual follow-up or retry later)
    """
    payload = {
        "name": name,
        "email": email,
        "phone": ctx.session.userdata.phone,
        "company": company,
        "conversationId": ctx.session.userdata.call_id,
        "adminId": ctx.session.userdata.admin_id,
    }

    return await n3_api.post("/api/lead/voicebot/lead-create", payload)


@tool("feedback_review_collection")
async def feedback_review_collection(
    rating: int,
    ctx: RunContext,
):
    """
    Collect customer service feedback rating.

    This tool must ONLY be used in outbound calls after:
    - The customer is asked to rate their experience
    - A valid rating (1–5) is explicitly provided

    ----------------------------
    PARAMETER DEFINITIONS
    ----------------------------

    rating:
        Customer rating for the service.
        Must be an integer between 1 and 5.
        Only accept valid numeric responses.
        If the user provides invalid input (e.g., "good", "10", "zero"),
        ask them to provide a rating between 1 and 5.

    ----------------------------
    EXECUTION RULES
    ----------------------------

    - Do NOT call this tool without a valid rating (1–5)
    - Do NOT ask for phone number
    - Do NOT ask for any additional information
    - As soon as a valid rating is received, call the tool immediately
    - Do NOT continue conversation after rating is collected
    - Do NOT use this tool for complaints or support issues

    ----------------------------
    FAILURE HANDLING
    ----------------------------

    - If the tool fails:
        - Do NOT retry automatically
        - Do NOT ask the user to repeat inputs
        - Apologize once
        - Inform the user the request cannot be completed right now
        - Offer a fallback (manual follow-up or retry later)
    """

    payload = {
        "rating": rating,
        "contact_phone": ctx.session.userdata.phone,
        "conversation_id": ctx.session.userdata.call_id,
        "agentId": ctx.session.userdata.agent_id,
    }

    return await n3_api.post("/service-feedback", payload)


@tool("information_faq_mode")
async def information_faq_mode(dummy: str): 
    pass

# ensure the following variables/env vars are set
SIP_TRUNK_ID = os.getenv("LIVEKIT_SIP_OUTBOUND_TRUNK")  # "ST_abcxyz"
SUPERVISOR_PHONE_NUMBER = os.getenv("LIVEKIT_SUPERVISOR_PHONE_NUMBER")  # "+12003004000"
SIP_NUMBER = os.getenv("LIVEKIT_SIP_NUMBER")  # "+15005006000" - caller ID shown to supervisor+
SUMMARY_INSTRUCTIONS = """
Introduce the conversation from your perspective as the AI assistant who participated in this call:

WHO you're talking to (name, role, company if mentioned)
WHY they contacted you (goal, problem, request)
WHY a human agent is requested or needed at this point
Brief summary in 100-200 characters from a first-person perspective"""

@tool("call_back")
async def call_back(city: str,
                    type: Literal["absolute", "relative"],
                    meridiem: Literal["am", "pm"],
                    ctx: RunContext,
                    value: Optional[float] = None,
                    unit: Optional[str] = None,
                    hour: Optional[int] = None):
    """
    Schedule a callback for the user at a later time.

    This tool must ONLY be used when:
    - The user explicitly requests a callback
    - The user agrees to be contacted later
    - A valid callback time can be determined

    ----------------------------
    PARAMETER DEFINITIONS
    ----------------------------

    type:
        Type of time provided by the user:
        - "absolute" → specific time (e.g., "call me at 5 PM")
        - "relative" → relative time (e.g., "call me in 2 hours")

        Choose based on how the user expresses the callback time.

    value:
        Value of time for callback
        Required ONLY when type = "relative".
        Should be omitted (None) for absolute time requests.

        The LLM does NOT control this parameter.

    unit:
        Unit of time provided for callback
        Must be either "hour" or "minutes".
        Should be omitted (None) for absolute time requests.

        The LLM does NOT control this parameter.

    hour:
        Hour of the callback (1–12).
        Required ONLY when type = "absolute".
        Should be omitted (None) for relative time requests.

        The LLM does NOT control this parameter.

    meridiem:
        Indicates whether the time is AM or PM.
        Must be either "am" or "pm".
        Required for absolute time interpretation.
        Should be omitted (None) for relative time requests.

        The LLM does NOT control this parameter.

    city:
        City of the caller.
        Used to help determine or validate the user's timezone.
        Extract from conversation if mentioned, otherwise infer from context if available.
        Required for absolute time interpretation.
        Should be omitted (None) for relative time requests.

        The LLM does NOT control this parameter.
    ----------------------------
    EXECUTION RULES
    ----------------------------

    - Do NOT call this tool unless the user clearly requests a callback
    - Confirm intent if unclear before calling
    - Ensure time information is complete and valid
    - Ask for missing details ONE AT A TIME
    - Do NOT ask for phone number (use system-provided value)
    - Do NOT proceed if time is ambiguous or incomplete
    - Call the tool immediately once all required details are available
    - Do NOT continue conversation after scheduling the callback

    ----------------------------
    TIME HANDLING RULES
    ----------------------------

    - If user provides absolute time:
        → require hour + meridiem
    - If user provides relative time:
        → do NOT use hour
    - Validate all time inputs before execution
    - Avoid assumptions for unclear time expressions

    ----------------------------
    FAILURE HANDLING
    ----------------------------

    - If the tool fails:
        - Do NOT retry automatically
        - Do NOT ask the user to repeat inputs
        - Apologize once
        - Inform the user the request cannot be completed right now
        - Offer a fallback (manual follow-up or retry later)
    """
    payload = {
        "type": type,
        "value": value,
        "unit": unit,
        "hour": hour,
        "meridiem": meridiem,
        "city": city,
        "contact_phone": ctx.session.userdata.phone,
        "conversation_id": ctx.session.userdata.call_id,
        "agentId": ctx.session.userdata.agent_id,
        "current_utc_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    }

    return await api.post("/voice-callback", payload)

@tool("do_not_call")
async def do_not_call(reason: str,
                      ctx: RunContext):
    """Use this tool to mark that the user should not be called back. Only use this tool if the user has explicitly stated that they do not want a callback, or if you have been instructed to do so by the user. Do not use this tool for any other reason.
    """
    headers = {
    "x-agent-secret": settings.N1_ISC_API_KEY,
    }
    payload = {
        "agentId": ctx.session.userdata.agent_id,
        "contact_phone": ctx.session.userdata.phone, #TODO QUERY - Feature for callback on different number? Misuse implications?,
        "conversation_id": ctx.session.userdata.call_id
    }
    
    return await _request(
        "POST",
        f"{settings.N3_ISC_URL}/do-not-call",
        headers=headers,
        json=payload
    )
@tool("transfer_to_human")
async def transfer_to_human(outbound_trunk: str, ctx: RunContext) -> None:
    """Called when the user asks to speak to a human agent. This will put the user on
        hold while the supervisor is connected.

    Ensure that the user has confirmed that they wanted to be transferred. Do not start transfer
    until the user has confirmed. You must use this tool only when you have user's confirmation to speak to a human.
    Examples on when the tool should be called:
    ----
    - User: Can I speak to your supervisor?
    - Assistant: Yes of course.
    ----

    Args:
        outbound_trunk: The unique outbound SIP trunk ID
                        associated with the account.
    """
    logger.info("tool called to transfer to human")
    await ctx.session.say(
        "Please hold while I connect you to a human agent.", allow_interruptions=False
    )
    try:
        assert SIP_TRUNK_ID is not None
        assert SUPERVISOR_PHONE_NUMBER is not None

        result = await WarmTransferTask(
            target_phone_number=ctx.session.userdata.human_escalation_phone,
            sip_trunk_id=ctx.session.userdata.outbound_trunk_id,
            # sip_number=SIP_NUMBER,
            chat_ctx=ctx.session._chat_ctx,
            # add extra instructions for summarization
            # you can also customize the entire instructions by overriding the `get_instructions` method
            extra_instructions=SUMMARY_INSTRUCTIONS,
            stt=ctx.session._agent.stt,
            llm=ctx.session._agent.llm,
            tts=ctx.session._agent.tts,
        )
    except llm.ToolError as e:
        logger.error(f"failed to transfer to supervisor with tool error: {e}")
        raise e
    except Exception as e:
        logger.exception("failed to transfer to supervisor")
        raise llm.ToolError(f"failed to transfer to supervisor with error: {e}") from e

    logger.info(
        "transfer to supervisor successful",
        extra={"supervisor_identity": result.human_agent_identity},
    )
    await ctx.session.say(
        "you are on the line with my supervisor. I'll be hanging up now.",
        allow_interruptions=False,
    )
    
    # Wait for the agent to finish speaking the intro
    await ctx.wait_for_playout()

    ctx.session.shutdown()