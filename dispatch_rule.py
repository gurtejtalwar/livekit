from livekit import api

lkapi = api.LiveKitAPI()

async def run():
    # Create a dispatch rule to place each caller in a separate room
    rule = api.SIPDispatchRule(
    dispatch_rule_individual = api.SIPDispatchRuleIndividual(
        room_prefix = 'call-',
    )
    )

    request = api.CreateSIPDispatchRuleRequest(
    dispatch_rule = api.SIPDispatchRuleInfo(
        rule = rule,
        name = 'Local Inbound Dispatch Rule',
        trunk_ids = [],
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
    await lkapi.aclose()

import asyncio
asyncio.run(run())