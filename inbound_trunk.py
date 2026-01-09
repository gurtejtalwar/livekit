import asyncio
from dotenv import load_dotenv

from livekit import api

load_dotenv(dotenv_path=".env",override=True)
# async def main():
#   livekit_api = api.LiveKitAPI()
#   # livekit_api = api.LiveKitAPI(url="ws://localhost:7880", api_key="devkey", api_secret="secret")

#   trunk = api.SIPInboundTrunkInfo(
#     name = "My trunk",
#     numbers = ["+12294713457"],
#     krisp_enabled = True,
#   )

#   request = api.CreateSIPInboundTrunkRequest(
#     trunk = trunk
#   )

#   trunk = await livekit_api.sip.create_inbound_trunk(request)
#   print("created inbound trunk", trunk)

#   await livekit_api.aclose()

# asyncio.run(main())

async def run():
    lkapi = api.LiveKitAPI()
    try:
        trunks = api.ListSIPDispatchRuleRequest()
        print(trunks)
    finally:
        await lkapi.aclose()