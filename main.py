import logging
from livekit.agents import cli, WorkerOptions

from app.services.agent.inbound_agent import inbound_entrypoint, agent_server

logger = logging.getLogger("inbound-agent")

if __name__ == "__main__":
    # Configure logging for better debugging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    cli.run_app(agent_server)
