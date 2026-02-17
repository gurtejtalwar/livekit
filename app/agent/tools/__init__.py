from dataclasses import dataclass

from app.agent import AgentConfig
from .tools import (
    end_call,
    make_ask_knowledge_base_tool, 
    load_knowledge_base,
    get_current_time,
    book_appointment,
    cancel_appointment,
    get_available_slots,
    reschedule_appointment,
    transfer_to_human,
    detected_voicemail
)

# TOOL_REGISTRY = {
#     "end_call": end_call,
#     "ask_knowledge_base": ask_knowledge_base,
#     "get_current_time": get_current_time,
#     "book_appointment": book_appointment,
#     "cancel_appointment": cancel_appointment,
#     "get_available_slots": get_available_slots,
#     "reschedule_appointment": reschedule_appointment,
#     "transfer_to_human": transfer_to_human,
#     "detected_voicemail": detected_voicemail,
# }


@dataclass
class ToolContext:
    agent_id: str
    kb: object | None = None

def resolve_ask_knowledge_base(ctx: ToolContext):
    return make_ask_knowledge_base_tool(ctx.kb)

def resolve_get_current_time(ctx: ToolContext):
    return get_current_time

def resolve_book_appointment(ctx: ToolContext):
    return book_appointment

def resolve_cancel_appointment(ctx: ToolContext):
    return cancel_appointment

def resolve_get_available_slots(ctx: ToolContext):
    return get_available_slots

def resolve_reschedule_appointment(ctx: ToolContext):
    return reschedule_appointment

def resolve_end_call(ctx: ToolContext):
    return end_call

def resolve_transfer_to_human(ctx: ToolContext):
    return transfer_to_human

def resolve_detected_voicemail(ctx: ToolContext):
    return detected_voicemail


TOOL_REGISTRY = {
    "ask_knowledge_base": resolve_ask_knowledge_base,
    "get_current_time": resolve_get_current_time,
    "book_appointment": resolve_book_appointment,
    "cancel_appointment": resolve_cancel_appointment,
    "get_available_slots": resolve_get_available_slots,
    "reschedule_appointment": resolve_reschedule_appointment,
    "end_call": resolve_end_call,
    "transfer_to_human": resolve_transfer_to_human,
    "detected_voicemail": detected_voicemail,
}

def resolve_tools(config: AgentConfig) -> list:
    kb=None
    if config.knowledge_base_id:
        kb = load_knowledge_base(config.knowledge_base_id) 
    tool_ctx = ToolContext(agent_id=config.agent_id, kb=kb)
    resolved_tools = []
    for tool_name in config.tools:
        resolver = TOOL_REGISTRY.get(tool_name)
        if not resolver:
            raise ValueError(f"Unknown tool: {tool_name}")
        resolved_tools.append(resolver(tool_ctx))

    return resolved_tools