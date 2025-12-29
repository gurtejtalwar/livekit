from typing import Optional, Dict, List
from pydantic import BaseModel

from livekit.agents import llm

from app.utils.tools import _request

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
    Book a new appointment for a user.
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
