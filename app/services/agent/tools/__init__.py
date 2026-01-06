from .tools import (
    end_call,
    ask_knowledge_base,
    get_current_time,
    book_appointment,
    cancel_appointment,
    get_available_slots,
    reschedule_appointment,
    transfer_to_human,
)

TOOL_REGISTRY = {
    "end_call": end_call,
    "ask_knowledge_base": ask_knowledge_base,
    "get_current_time": get_current_time,
    "book_appointment": book_appointment,
    "cancel_appointment": cancel_appointment,
    "get_available_slots": get_available_slots,
    "reschedule_appointment": reschedule_appointment,
    "transfer_to_human": transfer_to_human,
}


