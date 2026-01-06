import os
import pickle
import faiss
import torch
import asyncio
import logging

from dotenv import load_dotenv
from typing import Optional, List
from pydantic import BaseModel
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

from livekit import api
from livekit.agents.beta.workflows import WarmTransferTask
from livekit.agents import llm, get_job_context, RunContext

from app.utils.timer import Timer
from app.utils.tools import _request

load_dotenv(override=True)
logger = logging.getLogger("TOOLS")

class KnowledgeBase:
    def __init__(self, index, chunks):
        self.index = index
        self.chunks = chunks

    def search(self, query_emb, k=3):
        dist, idx = self.index.search(query_emb, k)
        indices = idx[0]
        return [
            self.chunks[i] if 0 <= i < len(self.chunks) else "[INVALID INDEX]"
            for i in indices
        ]

KB_CACHE = {} #TODO HAZARD use redis

#TODO Use resource centre id instead of agent id
def load_knowledge_base(resource_centre_id: str) -> KnowledgeBase: 
    if resource_centre_id in KB_CACHE:
        return KB_CACHE[resource_centre_id]

    with Timer(f"Load KB for {resource_centre_id}"):
        index = faiss.read_index(f"kbs/{resource_centre_id}/faiss.index")
        with open(f"kbs/{resource_centre_id}/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

    kb = KnowledgeBase(index=index, chunks=chunks)
    KB_CACHE[resource_centre_id] = kb
    return kb

def make_ask_knowledge_base_tool(kb: KnowledgeBase):

    @llm.function_tool
    async def ask_knowledge_base(question: str):
        with Timer("KB Tool Total"):
            with Timer("Embed Query"):
                q_emb = embed(question)

            with Timer("FAISS Search"):
                results = kb.search(q_emb, k=3)

            return "\n".join(results)

    return ask_knowledge_base

#TODO Pre call tasks
with open("dev_scripts/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

def embed(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

def get_text_from_indices(indices):
    """Return the text chunks for each FAISS result index."""
    result = []
    for idx in indices:
        if 0 <= idx < len(chunks):
            result.append(chunks[idx])
        else:
            result.append("[INVALID INDEX]")
    return result


with Timer("Load Index, Tokenizer and Embedding Model"):
    index = faiss.read_index("dev_scripts/faiss.index")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = ORTModelForFeatureExtraction.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        export=True
    )

@llm.function_tool
async def ask_knowledge_base(question: str):
    """Ultra-fast retrieval with streaming context"""
    with Timer("KB Tool Total:"):
        with Timer("Embed Query"):
            q_emb = embed(question)
        with Timer("FAISS Search"):
            dist, idx = index.search(q_emb, k=3)    # top 3 matches
        indices = idx[0]                        # array of indices
        matched_text = get_text_from_indices(indices)
        context = "\n".join(matched_text)
        return context

@llm.function_tool
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

@llm.function_tool
async def end_call(ctx: RunContext,
                   dummy: str = ""):
    """Use this tool when the user has signaled they wish to end the current call."""
    session = ctx.session
    session.generate_reply(instructions="You/User have chosen to end the call.")
    await ctx.wait_for_playout() # Ensure agent finishes speaking
    job_ctx = get_job_context()
    if job_ctx:
        # Use job_ctx.api to delete the room
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(room=job_ctx.room.name)
        )

@llm.function_tool
async def detected_answering_machine(ctx: RunContext, dummy: str):
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
@llm.function_tool
async def book_appointment(
    name: str,
    date: str,
    time: str,
    agentId: str,
    customFields: Optional[List[CustomField]] = None
):
    """
    Called only when the user confirms the date and time for booking a new appointment.
    Do not call this tool without confirming with the user first. 
    """
    payload = {
        "name": name,
        "date": date,
        "time": time,
        "agentId": "695254c6414ceece8c926513",
        "customFields": customFields or {}
    }

    return await _request(
        "POST",
        "/book-appointment",
        json=payload
    )

#/ -- Cancel Appointment Tool --/
@llm.function_tool
async def cancel_appointment(booking_id: str):
    """
    Cancel an existing appointment using booking ID.
    """
    return await _request(
        "DELETE",
        "/book-appointment",
        params={"booking_id": booking_id}
    )

#/ -- Get Available Slots Tool --/
@llm.function_tool
async def get_available_slots(
    agentId: str,
    date: str,
    status: str = "available"
):
    """
    Get available appointment slots for a given agent and date.
    """
    result = await _request(
        "GET",
        "/available-slot",
        params={
            # "agentId": agentId,
            "date": date,
            "status": status
        }
    )
    print("Available slots result:\n", result)
    return result

#/ -- Reschedule Appointment Tool --/
@llm.function_tool
async def reschedule_appointment(
    booking_id: str,
    name: str,
    date: str,
    time: str,
    agentId: str,
    # customFields: dict | None = None
):
    """
    Reschedule an existing appointment.
    """
    payload = {
        "booking_id": booking_id,
        "name": name,
        "date": date,
        "time": time,
        "agentId": agentId,
        # "customFields": customFields or {}
    }

    return await _request(
        "PATCH",
        "/book-appointment",
        json=payload
    )

#/ -- Create CRM Lead Tool --/
@llm.function_tool
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
    payload = {
        "firstName": first_name,
        "email": email,
        "phone": phone,
        "company": company,
        "adminId": admin_id,
    }

    return await _request(
        method="POST",
        url="/api/crm/external/lead-create",
        json=payload,
    )


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

@llm.function_tool
async def transfer_to_human(dummy: str, ctx: RunContext) -> None:
    """Called when the user asks to speak to a human agent. This will put the user on
        hold while the supervisor is connected.

    Ensure that the user has confirmed that they wanted to be transferred. Do not start transfer
    until the user has confirmed. You must use this tool only when you have user's confirmation to speak to a human.
    Examples on when the tool should be called:
    ----
    - User: Can I speak to your supervisor?
    - Assistant: Yes of course.
    ----
    """
    job_ctx = get_job_context()
    logger.info("tool called to transfer to human")
    await ctx.session.say(
        "Please hold while I connect you to a human agent.", allow_interruptions=False
    )
    try:
        assert SIP_TRUNK_ID is not None
        assert SUPERVISOR_PHONE_NUMBER is not None

        result = await WarmTransferTask(
            target_phone_number=SUPERVISOR_PHONE_NUMBER,
            sip_trunk_id=SIP_TRUNK_ID,
            # sip_number=SIP_NUMBER,
            chat_ctx=ctx.session._chat_ctx,
            # add extra instructions for summarization
            # you can also customize the entire instructions by overriding the `get_instructions` method
            extra_instructions=SUMMARY_INSTRUCTIONS,
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
    # ctx.session.shutdown()