from .kb import get_current_time, ask_knowledge_base
from .appointments import book_appointment, cancel_appointment, get_available_slots, reschedule_appointment

TOOL_REGISTRY = {
    "ask_knowledge_base": ask_knowledge_base,
    "get_current_time": get_current_time,
    "book_appointment": book_appointment,
    "cancel_appointment": cancel_appointment,
    "get_available_slots": get_available_slots,
    "reschedule_appointment": reschedule_appointment,
}