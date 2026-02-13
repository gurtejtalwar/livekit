import logging
from dotenv import load_dotenv

from livekit.agents import (
    cli
)

logger = logging.getLogger(__name__)

for noisy_logger in ["pymongo", "pymongo.topology", "pymongo.connection"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


load_dotenv(override=True)


# Import entrypoints to register RTC sessions
from app.agent.templates.inbound_agent.entrypoint import inbound_server  # noqa: F401
from app.agent.templates.outbound_agent.entrypoint import outbound_server # noqa: F401

if __name__ == "__main__":
    # Configure logging for better debugging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    cli.run_app(inbound_server)
    cli.run_app(outbound_server)
