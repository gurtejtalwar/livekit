from dotenv import load_dotenv
from livekit import api
load_dotenv(override=True)

async def run():
    lkapi = api.LiveKitAPI()
    try:
        # Create a dispatch rule to place each caller in a separate room
        rule = api.SIPDispatchRule(
        dispatch_rule_individual = api.SIPDispatchRuleIndividual(
            room_prefix = 'call-',
        )
        )

        request = api.CreateSIPDispatchRuleRequest(
        dispatch_rule = api.SIPDispatchRuleInfo(
            rule = rule,
            name = 'Inbound Dispatch Rule',
            trunk_ids = ["ST_NdWz5EhNfSEA"],
            room_config=api.RoomConfiguration(
                agents=[api.RoomAgentDispatch(
                    agent_name="inbound-agent",
                    metadata="job dispatch metadata",
                )]
            )
        )
        )

        dispatch =  await lkapi.sip.create_sip_dispatch_rule(request)
        print("created dispatch", dispatch)
    finally:
        await lkapi.aclose()

import asyncio
asyncio.run(run())