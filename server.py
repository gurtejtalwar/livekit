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
from app.agents.templates.inbound_agent.entrypoint import inbound_server  # noqa: F401

if __name__ == "__main__":
    cli.run_app(inbound_server)
