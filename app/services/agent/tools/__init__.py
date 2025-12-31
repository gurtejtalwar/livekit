from .kb import load_knowledge_base, make_ask_knowledge_base_tool, get_current_time, ask_knowledge_base, end_call
from .appointments import book_appointment, cancel_appointment, get_available_slots, reschedule_appointment

TOOL_REGISTRY = {
    "end_call": end_call,
    "ask_knowledge_base": ask_knowledge_base,
    "get_current_time": get_current_time,
    "book_appointment": book_appointment,
    "cancel_appointment": cancel_appointment,
    "get_available_slots": get_available_slots,
    "reschedule_appointment": reschedule_appointment,
}

