from livekit.agents import llm, RunContext

from app.models.call_models import upsert_voice_call_lead

@llm.function_tool
async def update_lead_profile(
    first_name: str | None,
    last_name: str | None,
    phone: str | None,
    email: str | None,
    # agent_summary: str | None,
    ctx: RunContext,
):
    """
    Silently update or enrich lead information in the background.

    This tool runs WITHOUT informing the user.

    ----------------------------
    PURPOSE
    ----------------------------

    - Capture lead data progressively during conversation
    - Maintain updated CRM records
    - Improve future personalization and sales context

    ----------------------------
    PARAMETER DEFINITIONS
    ----------------------------

    first_name:
        First name of the caller.
        Extract only if clearly provided.
        Do NOT guess.

    last_name:
        Last name of the caller.
        Extract only if clearly provided.
        Do NOT guess.
    
    phone:
        Phone number of the caller.
        Extract only if explicitly stated.
        Do NOT guess.

    email:
        Email of the caller.
        Extract only if explicitly stated.
        Do NOT infer.

    agent_summary:
        Short summary of the current conversation.
        Include:
        - intent
        - interest level
        - key discussion points

        Keep it concise (1–2 lines max).
    ----------------------------
    EXECUTION RULES
    ----------------------------

    - This tool runs silently (DO NOT mention it to the user)
    - Call whenever new useful information is discovered
    - Do NOT call if no new data is available
    - Do NOT overwrite good data with worse/empty data
    - Prefer incremental enrichment

    ----------------------------
    FAILURE HANDLING
    ----------------------------

    - Fail silently
    - Do NOT surface errors to the user
    """

    try:
        await upsert_voice_call_lead(
            phone=phone,
            user_id=ctx.session.userdata.user_id,
            admin_id=ctx.session.userdata.admin_id,
            agent_id=ctx.session.userdata.agent_id,
            first_name=first_name,
            last_name=last_name,
            email=email,
            # agent_summary=agent_summary, #TODO Save from external API
        )
    except Exception:
        pass

    return {"status": "captured"}