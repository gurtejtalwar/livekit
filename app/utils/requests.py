import httpx

API_BASE = "https://2831d7c36859.ngrok-free.app"
AGENT_SECRET = "kndsalfn221lfsa204ncodsa023459"

headers = {
    "x-agent-secret": AGENT_SECRET,
    "Content-Type": "application/json"
}

async def _request(method: str, url: str, *, headers: dict = headers, params=None, json=None):

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.request(
            method,
            f"{url}",
            headers=headers,
            params=params,
            json=json
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {"status": "ok"}
