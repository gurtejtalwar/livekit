import asyncio
from dotenv import load_dotenv

from livekit import api

load_dotenv(dotenv_path=".env",override=True)
async def main():
  livekit_api = api.LiveKitAPI()
  # livekit_api = api.LiveKitAPI(url="ws://localhost:7880", api_key="devkey", api_secret="secret")

  trunk = api.SIPInboundTrunkInfo(
    name = "My trunk",
    numbers = ["+12294713457"],
    krisp_enabled = True,
  )

  request = api.CreateSIPInboundTrunkRequest(
    trunk = trunk
  )

  trunk = await livekit_api.sip.create_inbound_trunk(request)
  print("created inbound trunk", trunk)

  await livekit_api.aclose()

asyncio.run(main())

# async def main():
#     lkapi = api.LiveKitAPI()
#     try:
#         dispatch_req = api.ListSIPDispatchRuleRequest(page=api.Pagination(limit=20))
#         dispatch = await lkapi.sip.list_dispatch_rule(dispatch_req)
#         trunk_req = api.ListSIPInboundTrunkRequest(page=api.Pagination(limit=20))
#         trunks = await lkapi.sip.list_inbound_trunk(trunk_req)
#         print("Dispatch rules: \n",dispatch)
#         print("Trunks: \n", trunks)
#     finally:
#         await lkapi.aclose()


# async def main():
#     lkapi = api.LiveKitAPI()
#     try:
#         update_req_1 = api.UpdateSIPDispatchRuleRequest(
#             sip_dispatch_rule_id="SDR_EvNDWDoDdXuH",
#             update=api.SIPDispatchRuleUpdate(metadata="eminence")
#         )
#         update_req_2 = api.UpdateSIPDispatchRuleRequest(
#             sip_dispatch_rule_id="SDR_SeSjLZ2Z9Ptn",
#             update=api.SIPDispatchRuleUpdate(metadata="perceptyne")
#         )
#         update_req_3 = api.UpdateSIPDispatchRuleRequest(
#             sip_dispatch_rule_id="SDR_sxxxWMGKCZv9",
#             update=api.SIPDispatchRuleUpdate(metadata="perceptyne")
#         )
#         dispatch_1 = await lkapi.sip.update_dispatch_rule_fields(rule_id="SDR_EvNDWDoDdXuH",
#                                                               metadata="eminence")
#         dispatch_2 = await lkapi.sip.update_dispatch_rule_fields(rule_id="SDR_SeSjLZ2Z9Ptn",
#                                                               metadata="perceptyne")
#         dispatch_3 = await lkapi.sip.update_dispatch_rule_fields(rule_id="SDR_sxxxWMGKCZv9",
#                                                               metadata="perceptyne")
#     finally:
#         await lkapi.aclose()

# asyncio.run(main())


# async def main():
#     lkapi = api.LiveKitAPI()
#     try:
#         request = api.CreateSIPOutboundTrunkRequest(
#             trunk=api.SIPOutboundTrunkInfo(
#                 name="Outbound Trunk",
#                 address="livekit-itsbot2.pstn.twilio.com",
#                 transport=api.SIPTransport.SIP_TRANSPORT_TLS,
#                 numbers=["+12294713457"],
#                 auth_username="gtbt",
#                 auth_password="Admin@1234567",
#             )
#         )
#         trunk = await lkapi.sip.create_outbound_trunk(request)
#         print("created outbound trunk", trunk)
#     except Exception as e:
#         print("error creating outbound trunk", e)
#     finally:
#         await lkapi.aclose()

# asyncio.run(main())