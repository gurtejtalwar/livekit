import logging
from dotenv import load_dotenv

from livekit.agents import (
    cli
)

from app.services.agent.templates.outbound_agent.entrypoint import outbound_server # noqa: F401

load_dotenv(override=True)

if __name__ == "__main__":
    # Configure logging for better debugging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asc   time)s - %(name)s - %(levelname)s - %(message)s'
    )
    cli.run_app(outbound_server)