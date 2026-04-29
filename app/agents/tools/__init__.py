from dataclasses import dataclass

from app.agents import AgentConfig
from .utilities import (
    update_lead_profile,
)
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
    call_back,
    do_not_call,
    detected_voicemail,
    feedback_review_collection,
    sales_lead_generation,
    information_faq_mode,
    customer_support
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

def resolve_ask_knowledge_base(tool_ctx: ToolContext):
    return make_ask_knowledge_base_tool(tool_ctx.kb)

def resolve_get_current_time(tool_ctx: ToolContext):
    return get_current_time

def resolve_book_appointment(tool_ctx: ToolContext):
    return book_appointment

def resolve_cancel_appointment(tool_ctx: ToolContext):
    return cancel_appointment

def resolve_get_available_slots(tool_ctx: ToolContext):
    return get_available_slots

def resolve_reschedule_appointment(tool_ctx: ToolContext):
    return reschedule_appointment

def resolve_end_call(tool_ctx: ToolContext):
    return end_call

def resolve_transfer_to_human(tool_ctx: ToolContext):
    return transfer_to_human

def resolve_detected_voicemail(tool_ctx: ToolContext):
    return detected_voicemail

def resolve_call_back(tool_ctx: ToolContext):
    return call_back

def resolve_do_not_call(tool_ctx: ToolContext):
    return do_not_call

def resolve_sales_lead_generation(tool_ctx: ToolContext):
    return sales_lead_generation

def resolve_feedback_review_collection(tool_ctx: ToolContext):
    return feedback_review_collection

def resolve_information_faq_mode(tool_ctx: ToolContext):
    return information_faq_mode

def resolve_customer_support(tool_ctx: ToolContext):
    return customer_support


def resolve_create_customer_ticket(tool_ctx: ToolContext):
    return None

def resolve_follow_ups_reminders(tool_ctx: ToolContext):
    return None
# def resolve_utility_switch_to_english(tool_ctx: ToolContext):
#     return utility_tools.switch_to_english
# def resolve_utility_switch_to_spanish(tool_ctx: ToolContext):
#     return utility_tools.switch_to_spanish
# def resolve_utility_switch_to_french(tool_ctx: ToolContext):
#     return utility_tools.switch_to_french
# def resolve_utility_switch_to_german(tool_ctx: ToolContext):
#     return utility_tools.switch_to_german
# def resolve_utility_switch_to_italian(tool_ctx: ToolContext):
#     return utility_tools.switch_to_italian
def resolve_utility_update_lead_profile(tool_ctx: ToolContext):
    return update_lead_profile

UTILITY_TOOL_REGISTRY = {
    "utility_update_lead_profile": resolve_utility_update_lead_profile,
    # "utility_switch_to_english": resolve_utility_switch_to_english,
    # "utility_switch_to_spanish": resolve_utility_switch_to_spanish,
    # "utility_switch_to_french": resolve_utility_switch_to_french,
    # "utility_switch_to_german": resolve_utility_switch_to_german,
    # "utility_switch_to_italian": resolve_utility_switch_to_italian
}

TOOL_REGISTRY = {
    "ask_knowledge_base": resolve_ask_knowledge_base,
    "get_current_time": resolve_get_current_time,
    "book_appointment": resolve_book_appointment,
    "cancel_appointment": resolve_cancel_appointment,
    "get_available_slots": resolve_get_available_slots,
    "reschedule_appointment": resolve_reschedule_appointment,
    "end_call": resolve_end_call,
    "transfer_to_human": resolve_transfer_to_human,
    "call_back": resolve_call_back,
    "do_not_call": resolve_do_not_call,
    "detected_voicemail": resolve_detected_voicemail,
    "sales_lead_generation": resolve_sales_lead_generation,
    "customer_support": resolve_customer_support,
    "feedback_review_collection": resolve_feedback_review_collection,
    "information_faq_mode": resolve_information_faq_mode,
    "follow_ups_reminders": resolve_follow_ups_reminders, #TODO HAZARD BACKLOG
    "create_customer_ticket": resolve_create_customer_ticket, #TODO HAZARD BACKLOG
}

def resolve_tools(config: AgentConfig) -> list:
    kb=None
    if config.knowledge_base_id:
        kb = load_knowledge_base(config.knowledge_base_id) 
    tool_ctx = ToolContext(agent_id=config.agent_id, kb=kb)
    resolved_tools = []
    # config.tools.extend([name for name in TOOL_REGISTRY.keys() if name.startswith("utility_") and name not in config.tools]) 
    for tool_name in config.tools:
        resolver = TOOL_REGISTRY.get(tool_name)
        if not resolver:
            raise ValueError(f"Unknown tool: {tool_name}")
        resolved_tools.append(resolver(tool_ctx))

    resolved_tools.extend([resolver(tool_ctx) for name, resolver in UTILITY_TOOL_REGISTRY.items()])
    return resolved_tools