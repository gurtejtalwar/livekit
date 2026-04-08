import httpx

API_BASE = "https://2831d7c36859.ngrok-free.app"
AGENT_SECRET = "kndsalfn221lfsa204ncodsa023459"


import json as json_lib
import time
import httpx

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer(__name__)


async def _request(method: str, url: str, *, headers: dict, params=None, json=None):
    with tracer.start_as_current_span("http_request") as span:
        start_time = time.time()

        # ---- Request metadata ----
        span.set_attribute("http.method", method)
        span.set_attribute("http.url", url)

        if params:
            span.set_attribute("http.params", json_lib.dumps(params))

        if json:
            span.set_attribute("http.request_body", json_lib.dumps(json))

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=json
                )

            latency_ms = int((time.time() - start_time) * 1000)

            # ---- Response metadata ----
            span.set_attribute("http.status_code", resp.status_code)
            span.set_attribute("http.latency_ms", latency_ms)

            resp.raise_for_status()

            response_data = resp.json() if resp.content else {"status": "ok"}

            # ⚠️ Avoid huge payloads
            span.set_attribute(
                "http.response_body",
                json_lib.dumps(response_data)[:2000]  # truncate
            )

            span.set_status(Status(StatusCode.OK))

            return response_data

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)

            span.set_attribute("http.latency_ms", latency_ms)
            span.set_attribute("error", str(e))
            span.set_status(Status(StatusCode.ERROR))

            raise