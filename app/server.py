import logging
from dotenv import load_dotenv

from livekit.agents import (
    AgentServer,
    cli
)

logger = logging.getLogger(__name__)

for noisy_logger in ["pymongo", "pymongo.topology", "pymongo.connection"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


load_dotenv(override=True)

agent_server = AgentServer()

if __name__ == "__main__":
    # Configure logging for better debugging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    cli.run_app(agent_server)
