import httpx

API_BASE = "http://localhost:3001"
AGENT_SECRET = "YOUR_AGENT_SECRET"

async def _request(method: str, url: str, *, params=None, json=None):
    headers = {
        "x-agent-secret": AGENT_SECRET,
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.request(
            method,
            f"{API_BASE}{url}",
            headers=headers,
            params=params,
            json=json
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {"status": "ok"}
